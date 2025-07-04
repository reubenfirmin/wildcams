"""Video I/O operations package for wildlife video processing."""

from .video_reader import VideoReader
from .frame_extractor import FrameExtractor
from .analysis_writer import AnalysisWriter
from .processed_tracker import ProcessedVideoTracker

__all__ = [
    'VideoReader',
    'FrameExtractor', 
    'AnalysisWriter',
    'ProcessedVideoTracker'
]