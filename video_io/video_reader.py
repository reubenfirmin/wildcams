"""Video reading operations for wildlife video processing."""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


logger = logging.getLogger(__name__)


class VideoReader:
    """Handles video file reading and metadata extraction."""
    
    def __init__(self, video_path: Path):
        """Initialize video reader for a specific video file."""
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames = 0
        self._fps = 0.0
        self._duration = 0.0
        self._is_opened = False
        
    def open(self) -> bool:
        """Open the video file with fallback backends."""
        # Try different video reading backends for corrupted files
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            self._cap = cv2.VideoCapture(str(self.video_path), backend)
            
            if self._cap.isOpened():
                self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._fps = self._cap.get(cv2.CAP_PROP_FPS)
                self._duration = self._total_frames / self._fps if self._fps > 0 else 0
                self._is_opened = True
                return True
                
            self._cap.release()
        
        logger.error(f"❌ Could not open video with any backend: {self.video_path}")
        return False
    
    def close(self) -> None:
        """Close the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False
    
    def is_opened(self) -> bool:
        """Check if video is successfully opened."""
        return self._is_opened
    
    def get_frame_at_index(self, frame_index: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Get a frame at specific index."""
        if not self._is_opened or self._cap is None:
            return False, None
            
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        
        if ret and frame is not None and frame.size > 0:
            return True, frame
        return False, None
    
    def get_metadata(self) -> dict:
        """Get video metadata."""
        return {
            'total_frames': self._total_frames,
            'fps': self._fps,
            'duration': self._duration,
            'path': str(self.video_path)
        }
    
    @property
    def total_frames(self) -> int:
        """Get total number of frames."""
        return self._total_frames
    
    @property
    def fps(self) -> float:
        """Get frames per second."""
        return self._fps
    
    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        return self._duration
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()