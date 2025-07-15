"""
Core application classes for wildlife video processing.

This package contains the main application classes that orchestrate
the video processing pipeline using composition instead of inheritance.
"""

from .wildlife_processor import WildlifeVideoProcessor
from .session_manager import ProcessingSessionManager  
from .batch_processor import BatchVideoProcessor

__all__ = [
    'WildlifeVideoProcessor',
    'ProcessingSessionManager', 
    'BatchVideoProcessor'
]