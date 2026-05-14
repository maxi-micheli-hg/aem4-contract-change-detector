"""Langfuse plumbing.

One `Langfuse` client + one `CallbackHandler` created at startup. The handler
is injected into every LangChain `.invoke()` so all LLM calls become
`generation` observations under whatever span is currently active. The client
is used by `main.py` to open the root span and to flush traces at exit.

----------------------------------------------------------------------------
GUÍA DE LECTURA — la "magia" de Langfuse en este proyecto:

  Langfuse usa DOS mecanismos distintos para capturar trazas:

    1) CallbackHandler (automático) — instrumentación pasiva.
       Lo pasás como `config={"callbacks": [handler]}` en cada
       llamada de LangChain. El handler intercepta los eventos
       internos de LangChain (start_llm, end_llm, error, etc.)
       y los traduce a observations en Langfuse. SIN tocar
       código de negocio.

    2) start_as_current_observation (manual) — instrumentación activa.
       En tu código creás un span con nombre explícito:

           with client.start_as_current_observation(name="my_step"):
               ... do work ...

       Esos spans son los nodos NOMBRADOS que ves en el árbol
       (contract-analysis, parse_original_contract, etc.).

  Por qué los dos: el callback handler te da los LLM calls "gratis"
  pero no sabe nada de tu lógica de negocio (no sabe que esos 2 LLM
  calls son "parsing del original" y "parsing de la enmienda"). Los
  spans manuales le ponen NOMBRES a esa lógica y los anidan en una
  jerarquía que tiene sentido para un humano.

  La INTERACCIÓN entre ambos:
    Cuando abrís un span manual con `start_as_current_observation`,
    queda "activo" en el contexto. Si DENTRO de ese context manager
    el callback handler crea una generation (por un llm.invoke), esa
    generation se anida automáticamente como HIJA del span activo.
    Por eso vemos `ChatOpenAI` adentro de `parse_original_contract`,
    no flotando suelto.
----------------------------------------------------------------------------
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

    La clase también funciona con handler=None / client=None — eso es
    el modo "Langfuse disabled" (e.g., cuando se corre con --no-langfuse).
    El resto del código no tiene que ramificar; los métodos `.callbacks`
    y `.flush()` son no-op en ese caso.
    """

    def __init__(self, handler: Any | None, client: Any | None) -> None:
        self.handler = handler
        self.client = client

    @property
    def callbacks(self) -> list[Any]:
        """Lista para pasar como `config={"callbacks": obs.callbacks}` en cada invoke().

        Si Langfuse está desactivado, devuelve `[]` y los invokes corren
        sin trazado pero sin error.
        """
        return [self.handler] if self.handler is not None else []

    def flush(self) -> None:
        """Fuerza el envío de todas las observations pendientes al servidor.

        Langfuse bufferea las observations y las manda en batches. Sin
        flush(), si el proceso termina rápido podés perder las últimas
        trazas. Por eso main.py hace `obs.flush()` en `finally`.
        """
        if self.client is not None:
            self.client.flush()
            log.info("[success]Langfuse trace flushed[/success]")


def get_observability(enabled: bool = True) -> Observability:
    """Build the Langfuse client + callback handler, or a no-op pair if disabled.

    Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from `.env`.

    Args:
        enabled: True (default) construye y conecta a Langfuse.
            False salta la conexión y devuelve un Observability con
            handler=None / client=None — útil para debugging offline
            cuando no querés gastar quota de Langfuse.

    Returns:
        Instancia de Observability lista para usar.
    """
    load_env()

    if not enabled:
        log.info("Langfuse disabled — running without trace capture")
        return Observability(handler=None, client=None)

    # LAZY IMPORT: solo importamos langfuse cuando enabled=True.
    # Esto permite correr el proyecto con --no-langfuse incluso si
    # langfuse no está instalado, o si las keys del .env no están.
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    creds = get_langfuse_credentials()
    # Cliente principal — lo usa main.py para abrir el span raíz y
    # para .flush() al final.
    client = Langfuse(
        public_key=creds["public_key"],
        secret_key=creds["secret_key"],
        host=creds["host"],
    )
    # Handler que pasamos como callback a cada llm.invoke() — captura
    # las generations automáticamente.
    handler = CallbackHandler()
    log.info(f"Langfuse enabled - host=[cyan]{creds['host']}[/cyan]")
    return Observability(handler=handler, client=client)
