"""Loader de entorno y credenciales.

Carga `.env` desde la raíz del proyecto (un nivel arriba de `src/`). Provee
accesores tipados para las credenciales de OpenAI y Langfuse que lanzan
errores claros y accionables cuando falta una key.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _MODULE_ROOT / ".env"


def load_env() -> None:
    """Carga `.env` una vez. Las llamadas subsiguientes son no-op gracias a dotenv."""
    load_dotenv(_ENV_FILE, override=False)


def get_module_root() -> Path:
    return _MODULE_ROOT


def get_openai_api_key() -> str:
    """Devuelve la OPENAI_API_KEY desde el `.env`. Lanza EnvironmentError si falta."""
    load_env()
    key = os.getenv("OPENAI_API_KEY")
    if not key or key.startswith("sk-..."):
        raise EnvironmentError(
            f"Falta OPENAI_API_KEY.\n  Agregala a {_ENV_FILE} (copiar desde .env.example)."
        )
    return key


def get_langfuse_credentials() -> dict[str, str]:
    """Devuelve un dict con las 3 credenciales de Langfuse. Lanza si faltan las keys."""
    load_env()
    creds = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
        "host": os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
    }
    if not creds["public_key"] or not creds["secret_key"]:
        raise EnvironmentError(
            "Faltan LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY.\n"
            f"  Agregalas a {_ENV_FILE}, o llamá a get_observability(enabled=False) "
            f"para correr sin captura de trazas."
        )
    return creds
