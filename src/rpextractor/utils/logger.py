"""
Custom Logger for Research Paper Extractor
==========================================
- Plain text format (human readable)
- File handler (writes to logs/rpextractor.log)
- Default level: DEBUG

Usage:
    from rpextractor.utils.logger import get_logger

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Processing started")
    # Output: 2026-03-28 18:30:00 | INFO | rpextractor.ingestion.pubmed_client | Processing started
"""

import logging
import os
from pathlib import Path
from configs.config import LOG_DIR

def get_logger(name: str, level= logging.DEBUG) -> logging.Logger:
    """Get a configured logger for the given module.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A logging.Logger instance with file handler
        and correlation ID support.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Log format ──
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── File handler ──
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, f"{Path(name).stem}.log")
    file_handler = logging.FileHandler(
        filename=log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # ── Attach handler ──
    logger.addHandler(file_handler)

    return logger