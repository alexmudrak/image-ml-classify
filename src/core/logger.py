import logging

from core.settings import DEBUG


def app_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="[%(levelname)s] [%(asctime)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    logger.addHandler(handler)

    return logger
