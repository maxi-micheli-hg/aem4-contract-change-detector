"""Schema Pydantic para el output final del pipeline.

`ContractChangeOutput` es el contrato estricto entre el ExtractionAgent y
cualquier consumidor downstream (dashboards de compliance, CRMs, logs de
auditoría). Los tres campos requeridos están mandados por la consigna del
proyecto.

Los strings de `description` de cada Field son intencionalmente ricos:
cuando este modelo se usa vía `ChatOpenAI(...).with_structured_output(ContractChangeOutput)`,
LangChain los reenvía a la API de structured outputs de OpenAI como la
`description` del JSON schema para la propiedad correspondiente, y GPT-4o
las lee como parte de sus instrucciones de generación.

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
    """Resultado estructurado de comparar un contrato original con su enmienda.

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
