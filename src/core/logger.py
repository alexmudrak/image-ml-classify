import logging
from logging.handlers import RotatingFileHandler

from core.settings import DEBUG


def app_logger(name):
    formatter = logging.Formatter(
        fmt="[%(levelname)s] [%(asctime)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        "./logs/logger.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(file_handler)

    return logger
