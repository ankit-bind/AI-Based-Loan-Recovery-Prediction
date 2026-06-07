# ============================================================
# LOAN RECOVERY PROJECT — ENTERPRISE LOGGING UTILITY
# ============================================================
"""
Logger module for loan recovery project.

Creates timestamped log files in logs/ directory.
Supports both file and console logging.
"""

import os
import logging
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = __name__,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup enterprise-grade logger.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        Configured logger instance
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Log file with timestamp
    log_file = os.path.join(
        log_dir,
        f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    )

    # Log format
    log_format = (
        "[ %(asctime)s ] %(name)s "
        "%(levelname)s - %(message)s"
    )

    # Root logger config
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            # File handler
            logging.FileHandler(log_file),
            # Console handler
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(name)
    return logger


# ── Module level logger ──────────────────────────────────────
logger = setup_logger(
    name="loanrecovery",
    log_dir="logs"
)


# ── Utility functions ────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Args:
        name: Name for the logger
              (usually __name__ of calling module)

    Returns:
        Named logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Data loaded successfully")
    """
    return logging.getLogger(name)


def log_dataframe_info(
    logger: logging.Logger,
    df,
    name: str = "DataFrame"
) -> None:
    """
    Log basic DataFrame information.

    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name to identify the DataFrame
    """
    logger.info(f"{name} Shape    : {df.shape}")
    logger.info(f"{name} Columns  : {df.shape[1]}")
    logger.info(f"{name} Rows     : {df.shape[0]:,}")
    logger.info(
        f"{name} Memory   : "
        f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB"
    )


def log_step(
    logger: logging.Logger,
    step: str,
    status: str = "STARTED"
) -> None:
    """
    Log a pipeline step with formatting.

    Args:
        logger: Logger instance
        step: Step name
        status: Step status (STARTED/COMPLETED/FAILED)
    """
    separator = "=" * 50
    logger.info(separator)
    logger.info(f"{status} : {step}")
    logger.info(separator)
