"""Pydantic schema for the final pipeline output.

`ContractChangeOutput` is the strict contract between the ExtractionAgent and
any downstream consumer (compliance dashboards, CRMs, audit logs). The three
required fields are mandated by the project consigna.

The `description` strings on each Field are intentionally rich: when this
model is used via `ChatOpenAI(...).with_structured_output(ContractChangeOutput)`,
LangChain forwards them to the OpenAI structured-outputs API as the JSON
schema `description` for the corresponding property, and GPT-4o reads them
as part of its generation instructions.

----------------------------------------------------------------------------
GUÍA DE LECTURA — el truco más sutil del proyecto:

  Pydantic NORMALMENTE se usa para validar datos en runtime. Pero acá lo
  estamos usando para algo más interesante:

      1. Definimos ContractChangeOutput con `Field(..., description="...")`
      2. Llamamos a `llm.with_structured_output(ContractChangeOutput)`
      3. LangChain traduce nuestro modelo Pydantic a JSON Schema
      4. Inyecta ese schema en el request a la API de OpenAI como
         parámetro `response_format`
      5. OpenAI FUERZA al modelo a devolver JSON que cumpla el schema
      6. Los `description=...` aparecen en el schema como hints
      7. GPT-4o los LEE y los usa como parte del prompting

  Por eso nuestros `description` están escritos como instrucciones (qué
  formato usar, qué ejemplos, qué evitar), no como docs para developers.
  Son DOBLE PROPÓSITO: docs + prompt engineering.

  Resultado: el extraction agent no necesita decirle al modelo "devolveme
  un JSON con estos 3 campos" en su system prompt — el schema enforcement
  + las descriptions ya hacen ese trabajo a nivel de API.
----------------------------------------------------------------------------
"""

from pydantic import BaseModel, Field


class ContractChangeOutput(BaseModel):
    """Structured result of comparing an original contract to its amendment.

    Los 3 campos son obligatorios (el `...` en Field = required). Los nombres
    están literalmente especificados en la consigna del bootcamp y no se pueden
    cambiar — fallaría la rúbrica 1.3 (validación Pydantic) si renombráramos.
    """

    sections_changed: list[str] = Field(
        ...,
        description=(
            "Lista de identificadores o títulos de las secciones del contrato "
            "que fueron modificadas, añadidas o eliminadas en la enmienda. "
            "Usa el formato que aparece en el documento (por ejemplo "
            "'2. Plazo', 'Cláusula de Confidencialidad', 'Protección de Datos'). "
            "Incluye también las secciones completamente nuevas que aparecen "
            "sólo en la enmienda."
        ),
    )
    topics_touched: list[str] = Field(
        ...,
        description=(
            "Lista de categorías legales o comerciales afectadas por la "
            "enmienda (por ejemplo 'Precio', 'Plazo', 'Confidencialidad', "
            "'Propiedad Intelectual', 'Soporte Técnico', 'Protección de "
            "Datos', 'Alcance del Servicio'). Cada tema debe aparecer una "
            "única vez incluso si afecta varias secciones."
        ),
    )
    summary_of_the_change: str = Field(
        ...,
        description=(
            "Resumen detallado en español que describe cada cambio "
            "introducido por la enmienda. Para cada cambio, indica el valor "
            "original y el nuevo valor cuando aplique (por ejemplo 'el plazo "
            "se extiende de 12 a 24 meses'), distinguiendo claramente entre "
            "adiciones (cláusulas nuevas), eliminaciones (cláusulas "
            "removidas) y modificaciones (cláusulas con redacción cambiada). "
            "Escribe en prosa fluida, no en bullets."
        ),
    )
