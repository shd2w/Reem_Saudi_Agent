from loguru import logger
import sys
from typing import Optional

from .config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    logger.remove()

    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"
        if not settings.log_json
        else "{message}"
    )

    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        backtrace=False,
        diagnose=False,
        format=log_format,
        enqueue=True,
    )


def get_logger(name: Optional[str] = None):
    return logger if name is None else logger.bind(logger_name=name)


