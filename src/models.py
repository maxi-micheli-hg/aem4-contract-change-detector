"""Pydantic schema for the final pipeline output.

`ContractChangeOutput` is the strict contract between the ExtractionAgent and
any downstream consumer (compliance dashboards, CRMs, audit logs). The three
required fields are mandated by the project consigna.

The `description` strings on each Field are intentionally rich: when this
model is used via `ChatOpenAI(...).with_structured_output(ContractChangeOutput)`,
LangChain forwards them to the OpenAI structured-outputs API as the JSON
schema `description` for the corresponding property, and GPT-4o reads them
as part of its generation instructions.
"""

from pydantic import BaseModel, Field


class ContractChangeOutput(BaseModel):
    """Structured result of comparing an original contract to its amendment."""

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
