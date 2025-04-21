# Import all encoder classes directly
# Local
from .arctic_encoder import ArcticEmbedEncoder

# Create a mapping of encoder types to their classes
ENCODER_REGISTRY = {
    "arctic": ArcticEmbedEncoder,
}


def get_encoder_class(encoder_type: str):
    """Get the encoder class based on the encoder type."""
    try:
        if encoder_type not in ENCODER_REGISTRY:
            supported_encoders = list(ENCODER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported encoder type: '{encoder_type}'. "
                f"Supported types are: {supported_encoders}"
            )
        return ENCODER_REGISTRY[encoder_type]
    except Exception as e:
        raise ValueError(f"Error getting encoder class: {str(e)}") from e
