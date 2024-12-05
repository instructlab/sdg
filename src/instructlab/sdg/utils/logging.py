# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
from rich.logging import RichHandler


def setup_logger(level="DEBUG"):
    """
    Setup a logger - ONLY to be used when running CLI commands in
    SDG directly. DO NOT call this from regular library code, and only
    call it from __main__ entrypoints in the instructlab.sdg.cli
    package
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
