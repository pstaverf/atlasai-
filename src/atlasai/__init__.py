"""Atlas AI: a local-first agent system with its own language model."""

from atlasai.config import AtlasConfig, ConfigError, load_config

__version__ = "0.1.0"
__all__ = ["AtlasConfig", "ConfigError", "__version__", "load_config"]
