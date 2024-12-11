# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC
from typing import Any, Dict, Union
import logging
import os.path

# Third Party
from jinja2 import Template, UndefinedError
import yaml

# Local
from ..registry import BlockRegistry

logger = logging.getLogger(__name__)


# This is part of the public API.
@BlockRegistry.register("Block")
class Block(ABC):
    def __init__(self, ctx, pipe, block_name: str) -> None:
        self.ctx = ctx
        self.pipe = pipe
        self.block_name = block_name

    def _validate(self, prompt_template: Template, input_dict: Dict[str, Any]) -> bool:
        """
        Validate the input data for this block. This method validates whether all required
        variables in the Jinja template are provided in the input_dict.

        Args:
            prompt_template (Template): The Jinja2 template object.
            input_dict (Dict[str, Any]): A dictionary of input values to check against
                                         the template.

        Returns:
            True if the input data is valid (i.e., no missing variables), False otherwise.
        """

        try:
            # Try rendering the template with the input_dict
            prompt_template.render(input_dict)
            return True
        except UndefinedError as e:
            # Jinja throws an UndefinedError for any undefnined template variables,
            # assuming the prompt_template was created using StrictUndefined. This
            # is the case for anything using PromptRegistry.template_from_string.
            logger.error(f"Missing key: {e}")
            return False

    def _load_config(self, config_path: str) -> Union[Dict[str, Any], None]:
        """
        Load the configuration file for this block.

        If the supplied configuration file is a relative path, it is assumed
        to be part of this Python package.

        Args:
            config_path (str): The path to the configuration file.

        Returns:
            The loaded configuration.
        """
        if not os.path.isabs(config_path):
            config_path = os.path.join(
                os.path.dirname(self.pipe.config_path), config_path
            )
        with open(config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)


# This is part of the public API.
class BlockConfigParserError(Exception):
    """An exception raised while parsing a block's configuration."""
