"""
Myth Museum - Logging

Centralized logging configuration using Rich for console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Global console instance
console = Console()

# Log directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "myth_museum.log"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Get a configured logger with Rich console output and optional file output.
    
    Args:
        name: Name of the logger (usually __name__)
        level: Logging level (default: INFO)
        log_file: Optional path to log file (default: logs/myth_museum.log)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
    )
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOG_FILE
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def get_console() -> Console:
    """Get the global Rich console instance."""
    return console


# Convenience function for quick logging setup
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup root logger with Rich output.
    
    Args:
        level: Logging level
    
    Returns:
        Root logger
    """
    return get_logger("myth_museum", level=level)


# Module-level logger for this file
_logger: Optional[logging.Logger] = None


def log() -> logging.Logger:
    """Get the module logger (lazy initialization)."""
    global _logger
    if _logger is None:
        _logger = get_logger("myth_museum")
    return _logger
