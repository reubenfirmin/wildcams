"""Configuration management package for wildlife video processing."""

from .processing_config import ProcessingConfig
from .configuration_manager import ConfigurationManager

__all__ = [
    'ProcessingConfig',
    'ConfigurationManager'
]