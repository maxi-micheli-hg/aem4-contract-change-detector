"""ContextualizationAgent — Agente 1: "Analista Senior de Contratos".

Recibe el texto completo del contrato original y su enmienda, y produce un
mapa estructural (Markdown) que alinea las secciones entre ambos documentos.
Ese mapa se pasa después al ExtractionAgent para que pueda enfocarse en
identificar cambios en lugar de tener que re-establecer las correspondencias
estructurales.

Crucialmente este agente NO enumera cambios — ese es el trabajo del auditor.
Su único output es la alineación estructural + un resumen ejecutivo de una
sola línea con el tipo de contrato y las partes.

Abre un span hijo de Langfuse llamado `contextualization_agent` para que la
traza muestre límites claros del handoff entre los dos agentes.

----------------------------------------------------------------------------
GUÍA DE LECTURA — ¿por qué un agente que NO extrae cambios?

  La pregunta más común en la defensa va a ser "¿por qué 2 agentes en
  vez de uno?". La respuesta más fuerte es ESTA:

      Este primer agente NO devuelve cambios. Devuelve un mapa
      estructural. Sirve como CONTEXTO al segundo agente, no como
      conclusión.

  Imitamos el flujo de un equipo legal real:
    - Analista Senior lee ambos documentos y arma una tabla:
        "Sección 1 del original corresponde a Sección 1 de la enmienda
         y ESTÁN distintas/iguales/nueva/eliminada"
    - Auditor recién entonces va sección por sección con el mapa
      en la mano y dice exactamente QUÉ cambió.

  Beneficio práctico: el segundo agente tiene menos cosas en mente al
  mismo tiempo. Menos cosas en mente = menos alucinaciones. Menos
  alucinaciones = mejor rúbrica.
----------------------------------------------------------------------------
"""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from shared.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# El system prompt (las "instrucciones permanentes" del agente).
#
# Anatomía:
#   1. Role priming: "Eres un Analista Senior de Contratos..."
#   2. Lista numerada de responsabilidades (5)
#   3. Una prohibición explícita ("NO TE CORRESPONDE...") — esto es
#      lo que evita que el agente se desvíe y empiece a hacer el trabajo
#      del Auditor.
#   4. Formato de salida fijado (intro + tabla Markdown)
#
# Por qué importa el role priming: GPT-4o ajusta tono, vocabulario y
# rigor según el rol que le des. "Analista Senior" -> respuestas más
# técnicas y formales que si dijéramos solo "asistente".
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """Eres un Analista Senior de Contratos en LegalMove, una empresa de tecnología legal. Tu trabajo es revisar contratos originales y sus enmiendas (adendas) para producir un MAPA ESTRUCTURAL que será usado por el equipo de auditoría legal.

Tu responsabilidad es:
1. Identificar el tipo de contrato (licencia, servicios, SaaS, confidencialidad, etc.) y las partes contratantes.
2. Listar TODAS las secciones presentes en cada documento, conservando los identificadores originales (números, títulos).
3. Alinear las secciones del original con las de la enmienda (¿cuál corresponde a cuál?).
4. Marcar qué secciones existen sólo en uno de los dos documentos (añadidas o eliminadas).
5. Para cada par de secciones, citar BREVEMENTE (1-2 frases) lo que dice cada documento.

NO TE CORRESPONDE describir los cambios en detalle ni cuantificarlos — eso lo hace el Auditor Legal en el siguiente paso. Tu output es sólo un mapa estructural.

Formato de salida: Markdown con dos partes:
1. Un párrafo introductorio (tipo de contrato, partes, fechas).
2. Una tabla Markdown con columnas: `ID Sección | Título | Cita Original | Cita Enmienda | Estado`, donde Estado ∈ {{`Sin cambios`, `Modificada`, `Nueva (sólo enmienda)`, `Eliminada (sólo original)`}}.
"""


# El user template — la parte variable del prompt que se rellena con
# los textos extraídos por los dos parse_contract_image() calls.
# Los `{original_text}` y `{amendment_text}` son placeholders de
# ChatPromptTemplate.
_USER_TEMPLATE = """### CONTRATO ORIGINAL

{original_text}

---

### ENMIENDA (ADENDA)

{amendment_text}

---

Produce el mapa estructural siguiendo el formato indicado en las instrucciones de sistema."""


class ContextualizationAgent:
    """Agente 1 — produce el mapa estructural comparado entre ambos documentos."""

    def __init__(self, llm: ChatOpenAI, langfuse_client: Any | None = None) -> None:
        """Inyección de dependencias: el LLM y el cliente Langfuse vienen de afuera.

        Este patrón se usa porque:
          1. Facilita testing (podés inyectar mocks).
          2. Permite compartir un solo LLM entre todos los agentes
             (lo construye main.py una vez).
          3. Hace explícito de qué depende cada agente.
        """
        self.llm = llm
        self.langfuse_client = langfuse_client

        # Construimos la chain LCEL: prompt | llm | parser
        # ChatPromptTemplate.from_messages convierte la lista [("system", ...),
        # ("human", ...)] en un Runnable que sabe formatear los placeholders.
        # StrOutputParser extrae el .content del AIMessage final.
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _USER_TEMPLATE)]
        )
        # El operador | crea un RunnableSequence (lo vas a ver en Langfuse
        # como el wrapper que contiene los pasos).
        self._chain = prompt | self.llm | StrOutputParser()

    def run(
        self,
        original_text: str,
        amendment_text: str,
        callbacks: list[Any] | None = None,
    ) -> str:
        """Genera el mapa estructural. Devuelve texto Markdown.

        Args:
            original_text: texto extraído por parse_contract_image del original.
            amendment_text: texto extraído por parse_contract_image de la enmienda.
            callbacks: lista de callbacks (CallbackHandler de Langfuse) para
                propagar a través de la cadena LCEL.

        Returns:
            String Markdown con el mapa estructural (intro + tabla).
        """
        # Diccionario con los valores para los placeholders del prompt.
        # Las keys tienen que coincidir con los {original_text} y
        # {amendment_text} de _USER_TEMPLATE.
        invoke_cfg = {"callbacks": callbacks} if callbacks else {}
        inputs = {"original_text": original_text, "amendment_text": amendment_text}

        def _do_invoke() -> str:
            return self._chain.invoke(inputs, config=invoke_cfg)

        # Si hay Langfuse, abrimos span hijo nombrado. Si no, ejecutamos
        # directo. Mismo patrón que en image_parser.py.
        if self.langfuse_client is not None:
            with self.langfuse_client.start_as_current_observation(
                name="contextualization_agent",
                as_type="span",
                # Adjuntamos solo los tamaños como metadata; los textos
                # completos quedan en la generation hija (el ChatOpenAI).
                input={
                    "original_chars": len(original_text),
                    "amendment_chars": len(amendment_text),
                },
            ) as span:
                result = _do_invoke()
                span.update(
                    output={
                        "map_chars": len(result),
                        "map_preview": result[:400],
                    }
                )
        else:
            result = _do_invoke()

        log.info(
            f"[success]contextualization_agent: mapa de {len(result)} caracteres producido[/success]"
        )
        return result
