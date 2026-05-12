"""ContextualizationAgent — Agente 1: "Analista Senior de Contratos".

Receives the full text of the original contract and its amendment, and
produces a structural map (Markdown) that aligns sections between the two
documents. This map is then passed to the ExtractionAgent so it can focus on
identifying changes rather than re-establishing structural correspondences.

Crucially this agent DOES NOT enumerate changes — that is the auditor's job.
Its only output is the structural alignment + a one-paragraph executive
summary of the contract type and parties.

Opens a child Langfuse span named `contextualization_agent` so the trace
shows clear handoff boundaries between the two agents.
"""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from shared.logger import get_logger

log = get_logger(__name__)

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
        self.llm = llm
        self.langfuse_client = langfuse_client
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _USER_TEMPLATE)]
        )
        self._chain = prompt | self.llm | StrOutputParser()

    def run(
        self,
        original_text: str,
        amendment_text: str,
        callbacks: list[Any] | None = None,
    ) -> str:
        """Generate the structural map. Returns Markdown text."""
        invoke_cfg = {"callbacks": callbacks} if callbacks else {}
        inputs = {"original_text": original_text, "amendment_text": amendment_text}

        def _do_invoke() -> str:
            return self._chain.invoke(inputs, config=invoke_cfg)

        if self.langfuse_client is not None:
            with self.langfuse_client.start_as_current_observation(
                name="contextualization_agent",
                as_type="span",
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

        log.info(f"[success]contextualization_agent: produced {len(result)}-char map[/success]")
        return result
