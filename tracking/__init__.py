"""
Temporal tracking system for wildlife video processing.

This package provides pluggable tracking implementations for maintaining
temporal consistency of detections across video frames.
"""

from .tracking_interface import TemporalTracker
from .track_data import Detection, Track, TrackingInfo
from .tracking_factory import TrackerFactory

__all__ = [
    'TemporalTracker',
    'Detection', 
    'Track',
    'TrackingInfo',
    'TrackerFactory'
]