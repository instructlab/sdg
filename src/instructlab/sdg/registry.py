# Standard
from typing import Dict
import logging

# Third Party
from jinja2 import Environment, StrictUndefined, Template

logger = logging.getLogger(__name__)


class BlockRegistry:
    """Registry for block classes to avoid manual additions to block type map."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, block_name: str):
        """
        Decorator to register a block class under a specified name.

        Args:
            block_name (str): Name under which to register the block.
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

        Returns:
            Dictionary of registered block names and classes.
        """
        return cls._registry


class PromptRegistry:
    """Registry for managing Jinja2 prompt templates."""

    _registry: Dict[str, Template] = {}
    _template_env: Environment = Environment(undefined=StrictUndefined)

    @classmethod
    def register(cls, *names: str):
        """Decorator to register Jinja2 template functions by name.

        Args:
            names (str): Names of the templates to register.

        Returns:
            A decorator that registers the Jinja2 template functions.
        """

        def decorator(func):
            template_str = func()
            template = cls.template_from_string(template_str)
            for name in names:
                cls._registry[name] = template
                logger.debug(f"Registered prompt template '{name}'")
            return func

        return decorator

    @classmethod
    def get_template(cls, name: str) -> Template:
        """Retrieve a Jinja2 template by name.

        Args:
            name (str): Name of the template to retrieve.

        Returns:
            The Jinja2 template instance.
        """
        if name not in cls._registry:
            raise KeyError(f"Prompt template '{name}' not found.")
        return cls._registry[name]

    @classmethod
    def get_registry(cls):
        """
        Retrieve the current registry map of block types.

        Returns:
            Dictionary of registered block names and classes.
        """
        return cls._registry

    @classmethod
    def template_from_string(cls, template_str):
        """
        Create a Jinja Template using our Environment from the source string

        Args:
            template_str: The template source, as a string-like thing

        Returns:
            Jinja Template
        """
        return cls._template_env.from_string(template_str)
