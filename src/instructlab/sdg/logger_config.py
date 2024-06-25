import logging
from rich.logging import RichHandler

def setup_logger(name):
    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )
    logger = logging.getLogger(name)
    return logger
