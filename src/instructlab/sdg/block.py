# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from collections import ChainMap
from typing import Any, Dict, Union

# Third Party
from datasets import Dataset
import yaml

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)


class Block(ABC):
    def __init__(self, block_name: str) -> None:
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

        :param config_path: The path to the configuration file.
        :return: The loaded configuration.
        """
        with open(config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)
