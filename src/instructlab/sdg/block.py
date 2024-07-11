# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC
from collections import ChainMap
from typing import Any, Dict, Union
import os.path

# Third Party
import yaml

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)


class Block(ABC):
    def __init__(self, ctx, block_name: str) -> None:
        self.ctx = ctx
        self.block_name = block_name

    @staticmethod
    def _validate(prompt_template: str, input_dict: Dict[str, Any]) -> bool:
        """
        Validate the input data for this block. This method should be implemented by subclasses
        to define how the block validates its input data.

        :return: True if the input data is valid, False otherwise.
        """

        class Default(dict):
            def __missing__(self, key: str) -> None:
                raise KeyError(key)

        try:
            prompt_template.format_map(ChainMap(input_dict, Default()))
            return True
        except KeyError as e:
            logger.error("Missing key: {}".format(e))
            return False

    def _load_config(self, config_path: str) -> Union[Dict[str, Any], None]:
        """
        Load the configuration file for this block.

        If the supplied configuration file is a relative path, it is assumed
        to be part of this Python package.

        :param config_path: The path to the configuration file.
        :return: The loaded configuration.
        """
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.ctx.sdg_base, config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)
