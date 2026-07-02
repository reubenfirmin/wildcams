"""
Core application classes for wildlife video processing.

This package contains the main application classes that orchestrate
the video processing pipeline using composition instead of inheritance.
"""

from .batch_processor import BatchVideoProcessor
from .session_manager import ProcessingSessionManager
from .wildlife_processor import WildlifeVideoProcessor

__all__ = ["WildlifeVideoProcessor", "ProcessingSessionManager", "BatchVideoProcessor"]
