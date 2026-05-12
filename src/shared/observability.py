"""Langfuse plumbing.

One `Langfuse` client + one `CallbackHandler` created at startup. The handler
is injected into every LangChain `.invoke()` so all LLM calls become
`generation` observations under whatever span is currently active. The client
is used by `main.py` to open the root span and to flush traces at exit.
"""

from typing import Any

from .config import get_langfuse_credentials, load_env
from .logger import get_logger

log = get_logger(__name__)


class Observability:
    """Holds the Langfuse handler + client.

    Pass `handler` as a callback to every LangChain `invoke()` so the LLM
    call is captured as a child `generation` observation. Use `client` in
    `main.py` to open the root span via `start_as_current_observation`.
    """

    def __init__(self, handler: Any | None, client: Any | None) -> None:
        self.handler = handler
        self.client = client

    @property
    def callbacks(self) -> list[Any]:
        return [self.handler] if self.handler is not None else []

    def flush(self) -> None:
        if self.client is not None:
            self.client.flush()
            log.info("[success]Langfuse trace flushed[/success]")


def get_observability(enabled: bool = True) -> Observability:
    """Build the Langfuse client + callback handler, or a no-op pair if disabled.

    Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from `.env`.
    """
    load_env()

    if not enabled:
        log.info("Langfuse disabled — running without trace capture")
        return Observability(handler=None, client=None)

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    creds = get_langfuse_credentials()
    client = Langfuse(
        public_key=creds["public_key"],
        secret_key=creds["secret_key"],
        host=creds["host"],
    )
    handler = CallbackHandler()
    log.info(f"Langfuse enabled - host=[cyan]{creds['host']}[/cyan]")
    return Observability(handler=handler, client=client)
