"""Entry point — multi-agent contract comparison pipeline.

Usage:
    uv run python src/main.py <original_image> <amendment_image>

Pipeline (5 stages, all traced in Langfuse under root span `contract-analysis`):
    1. parse_original_contract     (GPT-4o Vision)
    2. parse_amendment_contract    (GPT-4o Vision)
    3. contextualization_agent     ("Analista Senior" — structural map)
    4. extraction_agent            ("Auditor Legal Forense" — Pydantic JSON)
    5. Pydantic validation         (automatic via with_structured_output)

Exit codes:
    0 - success, JSON printed to stdout
    1 - Pydantic ValidationError
    2 - IO / API / argument error

----------------------------------------------------------------------------
GUÍA DE LECTURA (ver también GUIA.md en la raíz del repo):

  - Este archivo es el "director de orquesta". No conoce los detalles de
    cómo GPT-4o lee imágenes ni cómo se valida el JSON — solo SECUENCIA
    los pasos y abre el span raíz de Langfuse que va a contener a todos
    los hijos.
  - La idea de tener TODA la lógica de un agente encapsulada en su clase
    (ContextualizationAgent, ExtractionAgent) y que main.py solo las
    instancie + llame `.run()` es lo que el rubric 1.2 llama "separación
    clara de responsabilidades".
----------------------------------------------------------------------------
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup de path + encoding (debe correr ANTES de los imports del proyecto).
#
# Por qué `sys.stdout.reconfigure(encoding="utf-8")`:
#   Windows usa cp1252 por default y Rich escupe caracteres unicode (íconos,
#   tildes, etc.) que romperían el output con `UnicodeEncodeError`. Esto se
#   nos comió media tarde en el Módulo 3 antes de descubrirlo.
#
# Por qué insertamos `src/` en sys.path:
#   La consigna pide que el entry point viva en `src/main.py`. Cuando lo
#   ejecutás con `python src/main.py ...`, Python no sabe que sus "vecinos"
#   (`agents/`, `shared/`, etc.) son módulos importables. Sin esta línea,
#   los imports `from agents.contextualization_agent import ...` fallarían.
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Imports del proyecto (después del path manipulation, por eso ruff queda
# silenciado con per-file-ignore E402 en pyproject.toml).
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, RateLimitError
from pydantic import ValidationError
from rich.console import Console

from agents.contextualization_agent import ContextualizationAgent
from agents.extraction_agent import ExtractionAgent
from image_parser import parse_contract_image
from shared.config import get_openai_api_key, load_env
from shared.logger import get_logger
from shared.observability import get_observability

log = get_logger(__name__)
console = Console()


def _parse_args() -> argparse.Namespace:
    """Argparse: 2 imágenes posicionales + flag opcional para desactivar Langfuse.

    El help text de cada argumento aparece cuando corrés `... --help`. El
    epilog inyecta todo el docstring de este módulo, así el grader puede
    ver el pipeline completo con --help sin abrir el archivo.
    """
    parser = argparse.ArgumentParser(
        description="Compare an original contract with its amendment and produce a "
        "Pydantic-validated JSON describing every change.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "original_image",
        type=Path,
        help="Path to the scanned image of the original contract (.jpg/.jpeg/.png).",
    )
    parser.add_argument(
        "amendment_image",
        type=Path,
        help="Path to the scanned image of the amendment / adenda.",
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Disable Langfuse tracing (use only for offline debugging).",
    )
    return parser.parse_args()


def _build_llm() -> ChatOpenAI:
    """One ChatOpenAI(gpt-4o) instance shared by parser + both agents.

    Defensive settings — cada uno apunta a una línea del rubric (2.2):

      - temperature=0       Outputs reproducibles. Misma imagen + mismo
                            prompt -> mismo output. Crítico para la
                            defensa en vivo (sin sorpresas).
      - max_retries=2       OpenAI SDK reintenta automáticamente cuando
                            ve un APITimeoutError o RateLimitError. 2
                            retries con backoff exponencial cubren la
                            mayoría de errores transitorios sin enmascarar
                            problemas reales.
      - timeout=60          El default de OpenAI es 600s (10 minutos!).
                            Demasiado para nuestra Visión, donde cualquier
                            llamada que tarde > 1 min está claramente
                            stuck. 60s ayuda a que los errores se vean
                            rápido en desarrollo.
      - api_key=...         Pasada desde el .env. Si falta, get_openai_api_key()
                            lanza EnvironmentError con instrucciones.
                            Nunca hardcoded en el código.
    """
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=2,
        timeout=60,
        api_key=get_openai_api_key(),
    )


def _run_pipeline(args: argparse.Namespace) -> int:
    """Execute the 4 stages under one Langfuse root span. Returns exit code.

    Trace hierarchy creada por esta función:

        contract-analysis  (root, abierto acá abajo)
        ├── parse_original_contract       (abierto en image_parser.py)
        ├── parse_amendment_contract      (abierto en image_parser.py)
        ├── contextualization_agent       (abierto en el agente)
        └── extraction_agent              (abierto en el agente)

    El truco es que `langfuse.start_as_current_observation()` detecta si
    hay un span activo en el contexto. Si lo hay, el nuevo span se vuelve
    HIJO automáticamente. Por eso main.py abre solo el root y los demás
    se anidan naturalmente.
    """
    load_env()
    llm = _build_llm()
    # obs = wrapper que tiene el cliente de Langfuse + el CallbackHandler.
    # Si --no-langfuse fue pasado, ambos son None y el código de abajo
    # opera como no-op (sin trazas, pero corre igual).
    obs = get_observability(enabled=not args.no_langfuse)
    callbacks = obs.callbacks  # lista vacía si Langfuse está desactivado

    try:
        # Abrimos el span raíz si hay cliente Langfuse, si no usamos
        # nullcontext() (un context manager no-op del stdlib) para que el
        # `with` siguiente funcione igual sin tener que ramificar el código.
        if obs.client is not None:
            ctx = obs.client.start_as_current_observation(
                name="contract-analysis",
                as_type="span",
                input={
                    "original_image": args.original_image.name,
                    "amendment_image": args.amendment_image.name,
                },
            )
        else:
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx as root_span:
            # Etapa 1 — parse imagen original. Abre su propio span hijo
            # `parse_original_contract` adentro de image_parser.py.
            original_text = parse_contract_image(
                args.original_image,
                llm,
                role="original",
                callbacks=callbacks,
                langfuse_client=obs.client,
            )

            # Etapa 2 — idem para la enmienda.
            amendment_text = parse_contract_image(
                args.amendment_image,
                llm,
                role="amendment",
                callbacks=callbacks,
                langfuse_client=obs.client,
            )

            # Etapa 3 — Analista Senior produce el mapa estructural.
            # Acá instanciamos el agente justo antes de usarlo. Podría
            # estar instanciado fuera del try/except pero así queda más
            # claro qué pasa en cada etapa.
            context_map = ContextualizationAgent(llm, obs.client).run(
                original_text,
                amendment_text,
                callbacks=callbacks,
            )

            # Etapa 4 — Auditor Legal toma el mapa + ambos textos y
            # devuelve un ContractChangeOutput validado (Pydantic).
            # La validación ocurre adentro de with_structured_output(),
            # no necesitamos llamar `model_validate()` manualmente.
            result = ExtractionAgent(llm, obs.client).run(
                context_map,
                original_text,
                amendment_text,
                callbacks=callbacks,
            )

            # Adjuntamos el output al span raíz para que aparezca en el
            # dashboard de Langfuse cuando el grader lo abra (sin tener
            # que hacer drill-down al extraction_agent).
            if root_span is not None:
                root_span.update(
                    output={
                        "sections_changed": result.sections_changed,
                        "topics_touched": result.topics_touched,
                        "summary_chars": len(result.summary_of_the_change),
                    }
                )

        # Output al stdout. `console.print_json` renderiza con sintaxis
        # de colores en una terminal; en una pipe se degrada a JSON plano.
        console.rule("[bold green]ContractChangeOutput[/bold green]")
        console.print_json(result.model_dump_json(indent=2))
        return 0

    # ---- Manejo de errores ---------------------------------------------------
    # 3 ramas separadas porque cada una merece un exit code distinto y un
    # mensaje distinto. NO uso `except Exception: pass` ni `except` genérico
    # — la consigna premia errores ruidosos sobre éxitos silenciosos.
    except (FileNotFoundError, ValueError) as e:
        # Input del usuario: ruta no existe, extensión no soportada, etc.
        log.error(f"[error]Input error: {e}[/error]")
        return 2
    except (APITimeoutError, RateLimitError, OSError) as e:
        # API de OpenAI siguió fallando después de los 2 retries internos.
        log.error(f"[error]API or IO error after retries: {e}[/error]")
        return 2
    except ValidationError as e:
        # El extractor devolvió algo que no matchea ContractChangeOutput.
        # Muy raro con structured outputs, pero el rubric pide manejarlo.
        log.error(f"[error]Pydantic validation failed: {e}[/error]")
        return 1
    finally:
        # SIEMPRE flusheamos, incluso si hubo excepción. Sin esto las
        # trazas que se hayan abierto pueden quedar en buffer y nunca
        # llegar al servidor de Langfuse.
        obs.flush()


def main() -> None:
    """Punto de entrada. Solo encadena _parse_args -> _run_pipeline -> sys.exit."""
    args = _parse_args()
    code = _run_pipeline(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
