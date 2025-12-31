"""
Logging utilities for CoMLRL training.

Provides a logging system that works seamlessly with tqdm progress bars,
ensuring that log messages don't interfere with the progress bar display.
"""

import logging
import sys
from typing import Optional

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that writes to tqdm.write() to avoid disrupting progress bars.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(
    name: str = "comlrl",
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with TqdmLoggingHandler.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create handler
    handler = TqdmLoggingHandler()
    handler.setLevel(numeric_level)

    # Set format
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "comlrl") -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # If no handlers, set up with default INFO level
        return setup_logger(name, "INFO")
    return logger

