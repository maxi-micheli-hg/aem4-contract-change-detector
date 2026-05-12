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
"""

import argparse
import sys
from pathlib import Path

# Allow `uv run python src/main.py ...` from the project root by ensuring src/
# is on sys.path so the sibling imports below resolve.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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

    Defensive settings per rubric line 2.2:
      - temperature=0          → reproducible outputs
      - max_retries=2          → handle transient APITimeoutError / RateLimitError
      - timeout=60             → fail fast on stuck vision calls (default is 600s)
    """
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=2,
        timeout=60,
        api_key=get_openai_api_key(),
    )


def _run_pipeline(args: argparse.Namespace) -> int:
    """Execute the 4 stages under one Langfuse root span. Returns exit code."""
    load_env()
    llm = _build_llm()
    obs = get_observability(enabled=not args.no_langfuse)
    callbacks = obs.callbacks

    try:
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
            original_text = parse_contract_image(
                args.original_image,
                llm,
                role="original",
                callbacks=callbacks,
                langfuse_client=obs.client,
            )
            amendment_text = parse_contract_image(
                args.amendment_image,
                llm,
                role="amendment",
                callbacks=callbacks,
                langfuse_client=obs.client,
            )
            context_map = ContextualizationAgent(llm, obs.client).run(
                original_text,
                amendment_text,
                callbacks=callbacks,
            )
            result = ExtractionAgent(llm, obs.client).run(
                context_map,
                original_text,
                amendment_text,
                callbacks=callbacks,
            )
            if root_span is not None:
                root_span.update(
                    output={
                        "sections_changed": result.sections_changed,
                        "topics_touched": result.topics_touched,
                        "summary_chars": len(result.summary_of_the_change),
                    }
                )

        console.rule("[bold green]ContractChangeOutput[/bold green]")
        console.print_json(result.model_dump_json(indent=2))
        return 0

    except (FileNotFoundError, ValueError) as e:
        log.error(f"[error]Input error: {e}[/error]")
        return 2
    except (APITimeoutError, RateLimitError, OSError) as e:
        log.error(f"[error]API or IO error after retries: {e}[/error]")
        return 2
    except ValidationError as e:
        log.error(f"[error]Pydantic validation failed: {e}[/error]")
        return 1
    finally:
        obs.flush()


def main() -> None:
    args = _parse_args()
    code = _run_pipeline(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
