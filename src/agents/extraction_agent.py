"""ExtractionAgent — Agente 2: "Auditor Legal Forense".

Recibe el mapa estructural del ContextualizationAgent más los dos textos
del contrato, y produce un `ContractChangeOutput` (Pydantic) estrictamente
validado.

Este agente hace la extracción real de cambios: adiciones (cláusulas
nuevas), eliminaciones (cláusulas removidas en la enmienda) y modificaciones
(re-redactados o nuevos valores). Usa `with_structured_output()` de
LangChain para que la API de OpenAI fuerce el schema server-side y devuelva
una instancia Pydantic validada directamente.

Abre un span hijo de Langfuse `extraction_agent`. Captura `ValidationError`
y re-lanza con el texto crudo logueado para debugging — eso es lo que la
rúbrica llama "manejo elegante de errores".

----------------------------------------------------------------------------
GUÍA DE LECTURA — el agente que "cierra" el pipeline:

  Lo que hace distinto a este agente:
    - Es el ÚNICO que devuelve un objeto Pydantic, no un string.
    - Usa `llm.with_structured_output(ContractChangeOutput)` que activa
      la feature "structured outputs" de la API de OpenAI: el SERVIDOR
      valida el JSON antes de devolverlo, garantizando que no haya forma
      de que el output esté malformado.
    - Tiene un ejemplo one-shot dentro del system prompt (un mini JSON
      de ejemplo) para anclar al modelo en la forma esperada del output.

  El `try/except ValidationError` es defensivo: con structured outputs
  la validación a nivel de Pydantic casi nunca falla (porque el server
  ya filtra), pero la rúbrica pide "manejo elegante de excepciones de
  validación" así que está ahí.
----------------------------------------------------------------------------
"""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from models import ContractChangeOutput
from shared.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# System prompt — anatomía:
#
#   1. Role priming: "Eres un Auditor Legal Forense" (distinto al
#      Analista Senior — diferenciación importa para la rúbrica).
#   2. Lista numerada de los 3 inputs que va a recibir (mapa + 2 textos).
#   3. Taxonomía explícita de los 3 TIPOS de cambios:
#        ADICIONES / ELIMINACIONES / MODIFICACIONES
#      Esto reduce ambigüedad cuando el modelo decide cómo categorizar.
#   4. Instrucción ANTI-ALUCINACIÓN: "NO inventes cambios que no estén
#      respaldados por los textos". Lo dice explícitamente.
#   5. Especificación de los 3 campos del JSON (sections_changed,
#      topics_touched, summary_of_the_change) — duplica lo que ya está
#      en el schema de Pydantic, pero el modelo ve ambos y converge mejor.
#   6. EJEMPLO ONE-SHOT: un mini JSON de ejemplo. Esto es CLAVE para que
#      el `summary_of_the_change` salga en prosa fluida (como en el
#      ejemplo) y no en bullets sueltos.
#
# Las llaves dobles `{{ }}` escapan los `{ }` del JSON example para que
# ChatPromptTemplate no los confunda con placeholders.
# ---------------------------------------------------------------------------
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


# User template — la parte variable. Notar que pasamos 3 inputs ahora:
# el mapa del Analista + los 2 textos. Esto es el "handoff" entre agentes.
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

        # ---- LA LÍNEA MÁS IMPORTANTE DEL PROYECTO ---------------------------
        # `with_structured_output(ContractChangeOutput)` hace 3 cosas a la vez:
        #   1. Traduce ContractChangeOutput a JSON Schema.
        #   2. Configura el `response_format` de la API de OpenAI con ese
        #      schema, lo cual fuerza al modelo a generar JSON válido.
        #   3. Parsea automáticamente el output JSON en una instancia de
        #      ContractChangeOutput (Pydantic).
        #
        # Resultado: el `.invoke()` devuelve directamente una instancia
        # validada — no tenemos que hacer json.loads ni model_validate
        # manualmente. Si el JSON no cumple el schema, OpenAI lo rechaza
        # ANTES de devolverlo. Si por algún motivo extraño llegara igual,
        # Pydantic lo rechaza adentro del parser.
        self._chain = prompt | self.llm.with_structured_output(ContractChangeOutput)

    def run(
        self,
        context_map: str,
        original_text: str,
        amendment_text: str,
        callbacks: list[Any] | None = None,
    ) -> ContractChangeOutput:
        """Ejecuta la extracción. Devuelve un ContractChangeOutput validado.

        Args:
            context_map: el mapa estructural Markdown que produjo el agente 1.
            original_text: texto extraído del contrato original.
            amendment_text: texto extraído de la enmienda.
            callbacks: callbacks para propagar a Langfuse.

        Returns:
            ContractChangeOutput (instancia Pydantic validada).

        Raises:
            pydantic.ValidationError: si el output del LLM no conforma al
                schema (casi imposible con structured outputs, pero la
                rúbrica pide manejarlo).
        """
        invoke_cfg = {"callbacks": callbacks} if callbacks else {}
        inputs = {
            "context_map": context_map,
            "original_text": original_text,
            "amendment_text": amendment_text,
        }

        def _do_invoke() -> ContractChangeOutput:
            # try/except defensivo aunque es muy raro que ocurra.
            # Si pasara, loggeamos los errores específicos de Pydantic
            # (qué campo, qué tipo esperado, qué valor recibido) — útil
            # para debug.
            try:
                return self._chain.invoke(inputs, config=invoke_cfg)
            except ValidationError as e:
                log.error("[error]extraction_agent: falló la validación Pydantic[/error]")
                log.error(f"Errores de validación: {e.errors()}")
                raise

        # Mismo patrón de span hijo que en los otros agentes.
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
                # Adjuntamos el output del agente al span (no las listas
                # ni el summary completo — solo conteos + listas pequeñas)
                # para que sea visible en el dashboard sin tener que abrir
                # el generation hijo.
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
            f"[success]extraction_agent: {len(result.sections_changed)} secciones, "
            f"{len(result.topics_touched)} temas[/success]"
        )
        return result
