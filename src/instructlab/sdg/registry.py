# Standard
from typing import Dict
import logging

# Third Party
from jinja2 import StrictUndefined, Template

logger = logging.getLogger(__name__)


class BlockRegistry:
    """Registry for block classes to avoid manual additions to block type map."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, block_name: str):
        """
        Decorator to register a block class under a specified name.

        :param block_name: Name under which to register the block.
        """

        def decorator(block_class):
            cls._registry[block_name] = block_class
            logger.debug(
                f"Registered block '{block_name}' with class '{block_class.__name__}'"
            )
            return block_class

        return decorator

    @classmethod
    def get_registry(cls):
        """
        Retrieve the current registry map of block types.

        :return: Dictionary of registered block names and classes.
        """
        logger.debug("Fetching the block registry map.")
        return cls._registry


class PromptRegistry:
    """Registry for managing Jinja2 prompt templates."""

    _registry: Dict[str, Template] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a Jinja2 template function by name.

        :param name: Name of the template to register.
        :return: A decorator that registers the Jinja2 template function.
        """

        def decorator(func):
            template_str = func()
            cls._registry[name] = Template(template_str, undefined=StrictUndefined)
            logger.debug(f"Registered prompt template '{name}'")
            return func

        return decorator

    @classmethod
    def get_template(cls, name: str) -> Template:
        """Retrieve a Jinja2 template by name.

        :param name: Name of the template to retrieve.
        :return: The Jinja2 template instance.
        """
        if name not in cls._registry:
            raise KeyError(f"Prompt template '{name}' not found.")
        logger.debug(f"Retrieving prompt template '{name}'")
        return cls._registry[name]

    @classmethod
    def get_registry(cls):
        """
        Retrieve the current registry map of block types.

        :return: Dictionary of registered block names and classes.
        """
        logger.debug("Fetching the block registry map.")
        return cls._registry
