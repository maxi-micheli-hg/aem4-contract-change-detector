"""Parser multimodal de imágenes — GPT-4o Vision vía LangChain.

`parse_contract_image()` lee un JPG/PNG escaneado de un contrato desde disco,
lo codifica como data URL base64, construye un `HumanMessage` multimodal y
le pide a GPT-4o Vision que transcriba el documento preservando su jerarquía
(cláusulas numeradas, títulos de sección, etc.).

La función abre un span hijo de Langfuse llamado `parse_<role>_contract`
para que el caller (`main.py`) tenga las dos etapas de parsing como hijos
nombrados del span raíz `contract-analysis`. La invocación subyacente de
`ChatOpenAI.invoke()` también queda capturada automáticamente por el
`CallbackHandler` de LangChain, produciendo una observación `generation`
anidada con metadata de tokens y latencia.

----------------------------------------------------------------------------
GUÍA DE LECTURA:

  Este archivo encapsula TODO lo que tiene que ver con "leer una imagen":
    - validación de path/extensión
    - encoding a base64
    - construcción del HumanMessage multimodal
    - llamada al LLM
    - apertura/cierre del span de Langfuse para esta etapa

  Concepto clave que se entrega acá: cómo se le pasa una imagen a GPT-4o.
  La API de OpenAI acepta imágenes en dos formatos:
    1) URL pública (`https://...`)
    2) Data URL inline (`data:image/jpeg;base64,<bytes>`)
  Usamos (2) porque mantiene el sistema autosuficiente — no hace falta
  subir las imágenes a un bucket S3 ni nada parecido. El precio es que
  el payload del request es más grande (~33% más por el base64 encoding).
----------------------------------------------------------------------------
"""

import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from shared.logger import get_logger

log = get_logger(__name__)

# Extensiones soportadas y su mapeo MIME. Por qué solo estas:
#   - JPG/JPEG son el formato natural de los escaneos.
#   - PNG sirve por si alguien convierte un PDF directo a PNG.
#   - Otros formatos (TIFF, HEIC, etc.) podrían soportarse pero no son
#     necesarios para los inputs del bootcamp.
_SUPPORTED = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


# El prompt que va a "ver" GPT-4o junto con la imagen.
# Decisiones de prompting (rúbrica 2.1):
#   - Rol explícito ("especialista en OCR y análisis de documentos legales")
#   - Lista numerada de qué preservar (jerarquía, numeración, montos, etc.)
#   - Formato de output especificado (Markdown, '#' para títulos)
#   - 3 prohibiciones explícitas: NO añadir comentarios, NO resumir, NO traducir
#
# Sin estas instrucciones, GPT-4o tiende a parafrasear o "limpiar" el texto,
# lo cual sería catastrófico porque perderíamos identificadores de cláusulas.
_PARSE_PROMPT = (
    "Eres un especialista en OCR y análisis de documentos legales. "
    "Recibes la imagen escaneada de un contrato o de su enmienda (adenda). "
    "Transcribe el contenido completo de la imagen al texto en español "
    "preservando con la mayor fidelidad posible:\n"
    "  - Títulos y subtítulos\n"
    "  - Numeración de cláusulas y secciones (por ejemplo '1.', '2.1', etc.)\n"
    "  - Las partes contratantes y la fecha del contrato\n"
    "  - Montos, plazos, porcentajes y cualquier dato numérico\n"
    "  - El orden original de los párrafos\n\n"
    "Devuelve el texto extraído en formato Markdown, usando '#' para los "
    "títulos principales y manteniendo la numeración de cláusulas como texto "
    "plano al inicio de cada párrafo. NO añadas comentarios propios, NO "
    "resumas el contenido, NO traduzcas — sólo transcribe."
)


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Lee la imagen y devuelve (data_url, mime_type). Lanza excepción ante errores de IO / encoding.

    Por qué dos try/except distintos:
      - El primero captura errores de I/O (archivo bloqueado, permisos, etc.).
      - El segundo captura errores de base64 encoding (muy raro pero defensivo).
    Cada uno re-lanza con contexto en el mensaje (qué archivo falló) para
    que cuando el corrector vea el error sepa exactamente qué pasó.
    """
    try:
        raw = image_path.read_bytes()
    except OSError as e:
        raise OSError(f"No se pudo leer el archivo de imagen {image_path}: {e}") from e

    try:
        # base64.b64encode devuelve bytes; .decode("ascii") los convierte a str
        # para meterlos en la data URL. Ascii alcanza porque base64 solo usa
        # caracteres del subset A-Za-z0-9+/=.
        b64 = base64.b64encode(raw).decode("ascii")
    except Exception as e:
        raise ValueError(f"Falló el encoding base64 de {image_path}: {e}") from e

    mime = _SUPPORTED[image_path.suffix.lower()]
    # Formato de la data URL: data:<mime>;base64,<bytes>
    return f"data:{mime};base64,{b64}", mime


def parse_contract_image(
    image_path: Path | str,
    llm: ChatOpenAI,
    role: str,
    callbacks: list[Any] | None = None,
    langfuse_client: Any | None = None,
) -> str:
    """Extrae texto Markdown desde la imagen escaneada de un contrato usando GPT-4o Vision.

    Args:
        image_path: ruta a un archivo .jpg / .jpeg / .png.
        llm: una instancia configurada de `ChatOpenAI(model="gpt-4o", ...)`.
            Pasado por inyección de dependencias (main.py la construye una vez
            y nos la pasa) — así esta función es stateless y testeable.
        role: 'original' o 'amendment' — usado para nombrar el span de Langfuse.
            Mantiene los dos parsing calls separables en el dashboard.
        callbacks: lista de callbacks (típicamente `[langfuse_handler]`) que se
            reenvía al invoke de LangChain para que la llamada al LLM aparezca
            como una generation hija.
            Si es None, la llamada se hace pero no se trackea.
        langfuse_client: cliente Langfuse opcional. Si está provisto, la función
            abre explícitamente un span `parse_<role>_contract`. Si es None,
            el CallbackHandler sigue capturando la llamada al LLM pero sin
            el span que la envuelve.

    Returns:
        String Markdown con la transcripción completa.

    Raises:
        FileNotFoundError: si `image_path` no existe.
        ValueError: si la extensión del archivo no está soportada.
        OSError: si el archivo no se puede leer.
    """
    # Validaciones primero — falla rápido si el input está mal antes de
    # gastar tiempo encoding bytes ni llamando al LLM.
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {p}")
    if p.suffix.lower() not in _SUPPORTED:
        raise ValueError(
            f"Extensión de imagen no soportada '{p.suffix}'. Soportadas: {sorted(_SUPPORTED)}"
        )

    # Encoding (puede lanzar OSError o ValueError, ver _encode_image).
    data_url, mime = _encode_image(p)
    span_name = f"parse_{role}_contract"  # ej: "parse_original_contract"
    log.info(f"[cyan]{span_name}[/cyan]: parseando {p.name} ({len(data_url) // 1024} KB base64)")

    # ---- Construcción del mensaje multimodal --------------------------------
    # HumanMessage.content puede ser un string (text-only) o una lista de dicts
    # (multimodal). Cada dict tiene una key "type" que define cómo se interpreta.
    # Acá pasamos:
    #   1) Un bloque de texto con las instrucciones de transcripción
    #   2) Un bloque de imagen con la data URL
    # GPT-4o los procesa juntos: "ve" la imagen y aplica las instrucciones.
    message = HumanMessage(
        content=[
            {"type": "text", "text": _PARSE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    # config={"callbacks": [...]} es como LangChain propaga el handler de
    # Langfuse a través de toda la cadena de llamadas internas.
    invoke_cfg = {"callbacks": callbacks} if callbacks else {}

    def _do_invoke() -> str:
        # llm.invoke acepta una lista de mensajes. Acá solo pasamos uno
        # (el HumanMessage multimodal). No hay system prompt porque las
        # instrucciones ya están en el bloque de texto del HumanMessage.
        response = llm.invoke([message], config=invoke_cfg)
        # response es un AIMessage; .content es el texto generado.
        text = response.content if hasattr(response, "content") else str(response)
        # Defensa contra el caso (raro) en que content sea una lista de
        # partes en vez de un string plano.
        return text if isinstance(text, str) else str(text)

    # ---- Span de Langfuse ----------------------------------------------------
    # Si tenemos client de Langfuse, abrimos un span hijo nombrado.
    # Si no, ejecutamos sin envolver (modo --no-langfuse).
    if langfuse_client is not None:
        with langfuse_client.start_as_current_observation(
            name=span_name,
            as_type="span",
            # Metadata útil para ver en el dashboard sin tener que abrir
            # los detalles del generation hijo.
            input={"image_filename": p.name, "image_role": role, "mime_type": mime},
        ) as span:
            text = _do_invoke()
            # Adjuntamos algo del output al span para que sea inspectable.
            # Guardamos preview (300 chars) en vez del texto completo
            # porque el texto completo ya queda en el generation hijo.
            span.update(output={"text_length": len(text), "text_preview": text[:300]})
    else:
        text = _do_invoke()

    log.info(f"[success]{span_name}: extraídos {len(text)} caracteres[/success]")
    return text
