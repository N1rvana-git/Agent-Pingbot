"""Logging utilities."""
from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger.

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
