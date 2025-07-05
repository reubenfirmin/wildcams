"""Motion detection package for wildlife video processing."""

from .motion_detector import MotionDetector
from .motion_tracker import MotionTracker
from .background_subtractor import BackgroundSubtractorFactory

__all__ = [
    'MotionDetector',
    'MotionTracker', 
    'BackgroundSubtractorFactory'
]