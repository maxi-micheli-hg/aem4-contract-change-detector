"""Logger basado en Rich con un theme custom.

Provee `get_logger(name)` que devuelve un Logger estándar de Python pero
ruteado a través de un `RichHandler` con un theme custom que entiende los
tags `[success]`, `[warn]` y `[error]` para colorear las líneas.
"""

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_THEME = Theme(
    {
        "success": "bold green",
        "warn": "yellow",
        "error": "bold red",
    }
)

console = Console(theme=_THEME)

_handler = RichHandler(
    console=console,
    show_time=True,
    show_path=False,
    rich_tracebacks=True,
    markup=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[_handler],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
