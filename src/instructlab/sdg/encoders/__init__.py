# Standard
import importlib


def get_encoder_class(encoder_type: str):
    """Get the encoder class based on the encoder type."""
    try:
        # Convert encoder_type to class name (e.g., 'arctic' -> 'ArcticEmbedEncoder')
        class_name = f"{encoder_type.capitalize()}EmbedEncoder"

        # Use absolute import instead of relative
        module_name = f"sdg.src.instructlab.sdg.encoders.{encoder_type}_encoder"

        module = importlib.import_module(module_name)

        # Get the class from the module
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Unsupported encoder type: '{encoder_type}'") from e
