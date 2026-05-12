"""Multimodal image parser — GPT-4o Vision via LangChain.

`parse_contract_image()` reads a scanned contract JPG/PNG from disk, encodes
it to a base64 data URL, builds a multimodal `HumanMessage`, and asks
GPT-4o Vision to transcribe the document while preserving its hierarchy
(numbered clauses, section titles, etc.).

The function opens a child Langfuse span named `parse_<role>_contract` so the
caller (`main.py`) gets the two parsing steps as named children of the root
`contract-analysis` span. The underlying `ChatOpenAI.invoke()` is also
captured automatically by the LangChain Langfuse `CallbackHandler`,
producing a nested `generation` observation with token + latency metadata.
"""

import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from shared.logger import get_logger

log = get_logger(__name__)

_SUPPORTED = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

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
    """Read the image and return (data_url, mime_type). Raises on IO / encoding errors."""
    try:
        raw = image_path.read_bytes()
    except OSError as e:
        raise OSError(f"Could not read image file {image_path}: {e}") from e

    try:
        b64 = base64.b64encode(raw).decode("ascii")
    except Exception as e:
        raise ValueError(f"Failed to base64-encode {image_path}: {e}") from e

    mime = _SUPPORTED[image_path.suffix.lower()]
    return f"data:{mime};base64,{b64}", mime


def parse_contract_image(
    image_path: Path | str,
    llm: ChatOpenAI,
    role: str,
    callbacks: list[Any] | None = None,
    langfuse_client: Any | None = None,
) -> str:
    """Extract Markdown text from a scanned contract image using GPT-4o Vision.

    Args:
        image_path: path to a .jpg / .jpeg / .png file.
        llm: a configured `ChatOpenAI(model="gpt-4o", ...)` instance.
        role: 'original' or 'amendment' — used to name the Langfuse span.
        callbacks: callback list (typically `[langfuse_handler]`) forwarded to
            the LangChain invoke so the LLM call appears as a child generation.
        langfuse_client: optional Langfuse client. If provided, the function
            opens a `parse_<role>_contract` span explicitly. If None, the
            CallbackHandler still captures the LLM call but without the
            wrapping span.

    Returns:
        Markdown string with the full transcription.

    Raises:
        FileNotFoundError: if `image_path` does not exist.
        ValueError: if the file extension is not supported.
        OSError: if the file cannot be read.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() not in _SUPPORTED:
        raise ValueError(
            f"Unsupported image extension '{p.suffix}'. Supported: {sorted(_SUPPORTED)}"
        )

    data_url, mime = _encode_image(p)
    span_name = f"parse_{role}_contract"
    log.info(f"[cyan]{span_name}[/cyan]: parsing {p.name} ({len(data_url) // 1024} KB base64)")

    message = HumanMessage(
        content=[
            {"type": "text", "text": _PARSE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    invoke_cfg = {"callbacks": callbacks} if callbacks else {}

    def _do_invoke() -> str:
        response = llm.invoke([message], config=invoke_cfg)
        text = response.content if hasattr(response, "content") else str(response)
        return text if isinstance(text, str) else str(text)

    if langfuse_client is not None:
        with langfuse_client.start_as_current_observation(
            name=span_name,
            as_type="span",
            input={"image_filename": p.name, "image_role": role, "mime_type": mime},
        ) as span:
            text = _do_invoke()
            span.update(output={"text_length": len(text), "text_preview": text[:300]})
    else:
        text = _do_invoke()

    log.info(f"[success]{span_name}: extracted {len(text)} chars[/success]")
    return text
