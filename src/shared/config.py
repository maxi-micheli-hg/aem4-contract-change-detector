"""Environment + credentials loader.

Loads `.env` from the project root (one level up from `src/`). Provides
typed accessors for OpenAI and Langfuse credentials that raise loud,
actionable errors when a key is missing.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _MODULE_ROOT / ".env"


def load_env() -> None:
    """Load `.env` once. Subsequent calls are no-ops thanks to dotenv."""
    load_dotenv(_ENV_FILE, override=False)


def get_module_root() -> Path:
    return _MODULE_ROOT


def get_openai_api_key() -> str:
    load_env()
    key = os.getenv("OPENAI_API_KEY")
    if not key or key.startswith("sk-..."):
        raise EnvironmentError(
            f"OPENAI_API_KEY is missing.\n  Add it to {_ENV_FILE} (copy from .env.example)."
        )
    return key


def get_langfuse_credentials() -> dict[str, str]:
    load_env()
    creds = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
        "host": os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
    }
    if not creds["public_key"] or not creds["secret_key"]:
        raise EnvironmentError(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are missing.\n"
            f"  Add them to {_ENV_FILE}, or call get_observability(enabled=False) "
            f"to run without trace capture."
        )
    return creds
