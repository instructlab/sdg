# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC
from typing import Any, Dict, Union
import logging
import os.path

# Third Party
import yaml

logger = logging.getLogger(__name__)


# This is part of the public API.
class Block(ABC):
    def __init__(self, ctx, pipe, block_name: str) -> None:
        self.ctx = ctx
        self.pipe = pipe
        self.block_name = block_name

    def _load_config(self, config_path: str) -> Union[Dict[str, Any], None]:
        """
        Load the configuration file for this block.

        If the supplied configuration file is a relative path, it is assumed
        to be part of this Python package.

        :param config_path: The path to the configuration file.
        :return: The loaded configuration.
        """
        if not os.path.isabs(config_path):
            config_path = os.path.join(
                os.path.dirname(self.pipe.config_path), config_path
            )
        with open(config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)
