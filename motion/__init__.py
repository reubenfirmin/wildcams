"""Motion detection package for wildlife video processing."""

from .background_subtractor import BackgroundSubtractorFactory
from .motion_detector import MotionDetector
from .motion_tracker import MotionTracker

__all__ = ["MotionDetector", "MotionTracker", "BackgroundSubtractorFactory"]
