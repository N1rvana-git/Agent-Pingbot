"""Logging and console helpers."""
from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure root logging once and return a named logger.

    Args:
        level: Logging level.
        name: Logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    return logging.getLogger(name)


def configure_stdio() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to a legacy code page (GBK on zh-CN), which raises
    ``UnicodeEncodeError`` as soon as a standard id or product name contains a
    non-GBK character. Reconfiguring is a no-op on streams that do not support
    it (pytest capture, redirected pipes).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
