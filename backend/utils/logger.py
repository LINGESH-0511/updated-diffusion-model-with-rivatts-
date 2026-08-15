"""
Centralized logger for the project.
"""
import logging
import sys
from .config import LOG_LEVEL

logger = logging.getLogger("A2F")
logger.setLevel(getattr(logging, LOG_LEVEL))
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler("bridge.log")
file_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def get_logger(name: str = "A2F") -> logging.Logger:
    """
    Returns a named child logger under the 'A2F' namespace, inheriting
    the level and console handler configured above.
    """
    return logging.getLogger(f"A2F.{name}") if name != "A2F" else logger