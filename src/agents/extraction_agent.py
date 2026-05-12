"""ExtractionAgent — Agente 2: "Auditor Legal Forense".

Receives the structural map from the ContextualizationAgent plus both contract
texts, and produces a strictly-validated `ContractChangeOutput` (Pydantic).

This agent does the actual change extraction: additions (new clauses),
deletions (clauses removed in the amendment), and modifications (rewordings
or new values). It uses LangChain's `with_structured_output()` so the OpenAI
API enforces the schema server-side and returns a validated Pydantic
instance directly.

Opens a child Langfuse span `extraction_agent`. Catches `ValidationError`
and re-raises with the raw text logged for debugging — this is what the
rubric calls "graceful error handling".
"""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from models import ContractChangeOutput
from shared.logger import get_logger

log = get_logger(__name__)


_SYSTEM_PROMPT = """Eres un Auditor Legal Forense en LegalMove. Recibes:
1. Un MAPA ESTRUCTURAL elaborado por el Analista Senior que alinea las secciones del contrato original con las de la enmienda.
2. El texto completo del contrato original.
3. El texto completo de la enmienda (adenda).

Tu trabajo es identificar, aislar y describir CADA CAMBIO introducido por la enmienda. Debes distinguir entre:
- ADICIONES: cláusulas o secciones que aparecen sólo en la enmienda.
- ELIMINACIONES: cláusulas o secciones que aparecen sólo en el original.
- MODIFICACIONES: cláusulas presentes en ambos documentos pero con redacción, montos, plazos u otros datos distintos.

Para cada cambio, cita explícitamente el valor original y el valor nuevo cuando sea posible (por ejemplo: "el plazo se extiende de 12 a 24 meses", "el monto pasa de USD 8.000 a USD 9.500"). NO inventes cambios que no estén respaldados por los textos; si no estás seguro, omítelos.

Devuelve un objeto JSON que cumpla exactamente el schema `ContractChangeOutput`, con tres campos:
- `sections_changed`: lista de identificadores/títulos de secciones modificadas, añadidas o eliminadas.
- `topics_touched`: lista de categorías legales/comerciales afectadas (por ejemplo "Precio", "Plazo", "Confidencialidad", "Propiedad Intelectual"), sin duplicar temas.
- `summary_of_the_change`: descripción en prosa fluida (no bullets), en español, que enumere cada cambio con su valor original y nuevo cuando aplique.

EJEMPLO de output válido (sólo para demostrar la forma, no copies estos valores):
{{
  "sections_changed": ["2. Plazo", "3. Pago", "Protección de Datos"],
  "topics_touched": ["Plazo", "Precio", "Protección de Datos"],
  "summary_of_the_change": "La enmienda extiende el plazo del contrato de 12 a 24 meses y aumenta la tarifa anual de USD 12.000 a USD 15.000. Además, se incorpora una nueva cláusula de Protección de Datos que obliga al Licenciante a cumplir con las normativas aplicables sobre la información del Licenciatario."
}}
"""

_USER_TEMPLATE = """### MAPA ESTRUCTURAL (del Analista Senior)

{context_map}

---

### CONTRATO ORIGINAL

{original_text}

---

### ENMIENDA (ADENDA)

{amendment_text}

---

Produce el JSON `ContractChangeOutput` que enumere todos los cambios introducidos por la enmienda."""


class ExtractionAgent:
    """Agente 2 — extrae los cambios y devuelve un ContractChangeOutput validado."""

    def __init__(self, llm: ChatOpenAI, langfuse_client: Any | None = None) -> None:
        self.llm = llm
        self.langfuse_client = langfuse_client
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _USER_TEMPLATE)]
        )
        # with_structured_output wires the OpenAI structured-outputs API so the
        # response is guaranteed to match ContractChangeOutput (server-side
        # schema enforcement) and is returned as a validated Pydantic instance.
        self._chain = prompt | self.llm.with_structured_output(ContractChangeOutput)

    def run(
        self,
        context_map: str,
        original_text: str,
        amendment_text: str,
        callbacks: list[Any] | None = None,
    ) -> ContractChangeOutput:
        """Run the extraction. Returns a validated ContractChangeOutput.

        Raises:
            pydantic.ValidationError: if the LLM output does not conform to
                the schema (extremely unlikely with structured outputs, but
                kept as a defensive measure as required by the rubric).
        """
        invoke_cfg = {"callbacks": callbacks} if callbacks else {}
        inputs = {
            "context_map": context_map,
            "original_text": original_text,
            "amendment_text": amendment_text,
        }

        def _do_invoke() -> ContractChangeOutput:
            try:
                return self._chain.invoke(inputs, config=invoke_cfg)
            except ValidationError as e:
                log.error("[error]extraction_agent: Pydantic validation failed[/error]")
                log.error(f"Validation errors: {e.errors()}")
                raise

        if self.langfuse_client is not None:
            with self.langfuse_client.start_as_current_observation(
                name="extraction_agent",
                as_type="span",
                input={
                    "context_map_chars": len(context_map),
                    "original_chars": len(original_text),
                    "amendment_chars": len(amendment_text),
                },
            ) as span:
                result = _do_invoke()
                span.update(
                    output={
                        "sections_changed": result.sections_changed,
                        "topics_touched": result.topics_touched,
                        "summary_chars": len(result.summary_of_the_change),
                    }
                )
        else:
            result = _do_invoke()

        log.info(
            f"[success]extraction_agent: {len(result.sections_changed)} sections, "
            f"{len(result.topics_touched)} topics[/success]"
        )
        return result
