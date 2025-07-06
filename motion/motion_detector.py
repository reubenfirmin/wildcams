"""Motion detector for wildlife video processing."""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .background_subtractor import BackgroundSubtractorFactory
from config import ProcessingConfig
from constants import DEFAULT_MOTION_METHOD

logger = logging.getLogger('wildcams')


class MotionDetector:
    """Detects motion regions in video frames using background subtraction."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize motion detector with configuration.
        
        Args:
            config: ProcessingConfig object
        """
        self.bg_subtractor = None
        self.analysis_subtractor = None
        self._init_motion_detector(config)
    
    def _init_motion_detector(self, config: ProcessingConfig):
        """Initialize motion detection algorithm."""
        method = config.motion_method
        
        # Create main background subtractor
        self.bg_subtractor = BackgroundSubtractorFactory.create_subtractor(method, config)
        
        logger.info(f"🔍 Motion method: {method}")
        logger.info(f"🎯 Motion area range: {config.min_motion_area}-{config.max_motion_area} pixels")
        logger.info(f"📐 Region size: min {config.min_region_width}x{config.min_region_height}")
        logger.info(f"📏 Max aspect ratio: {config.max_aspect_ratio}")
    
    def create_analysis_motion_detector(self, config: ProcessingConfig) -> Any:
        """Create a separate motion detector for analysis (if needed)."""
        if self.analysis_subtractor is None:
            method = config.motion_method
            self.analysis_subtractor = BackgroundSubtractorFactory.create_subtractor(method, config)
            logger.info(f"🔍 Created separate analysis motion detector: {method}")
        return self.analysis_subtractor
    
    def open_video_stream(self, video_path: Path) -> Optional[cv2.VideoCapture]:
        """Open video stream with fallback backends."""
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(str(video_path), backend)
            if cap.isOpened():
                return cap
            cap.release()
        
        logger.error(f"❌ Could not open video with any backend: {video_path}")
        return None
    
    def detect_motion_regions(self, frame: np.ndarray, config: ProcessingConfig) -> List[Tuple[int, int, int, int]]:
        """
        Detect motion regions in frame using background subtraction.
        
        Args:
            frame: Input frame
            config: ProcessingConfig object
            
        Returns:
            List of motion regions as (x1, y1, x2, y2) tuples
        """
        if self.bg_subtractor is None:
            return []
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < config.min_motion_area or area > config.max_motion_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by dimensions and aspect ratio
            if (w < config.min_region_width or 
                h < config.min_region_height):
                continue
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > config.max_aspect_ratio:
                continue
            
            # Expand region with smart margin for better ML context
            base_margin = config.motion_margin
            
            # Smart margin calculation: larger context for better crops
            # Use percentage-based expansion with minimum absolute margin
            percentage_margin_w = max(base_margin, w * 0.75)  # 75% of motion width or base margin
            percentage_margin_h = max(base_margin, h * 0.75)  # 75% of motion height or base margin
            
            # Ensure minimum crop size for ML effectiveness (at least 150x150)
            min_crop_width = 150
            min_crop_height = 150
            
            # Calculate expanded region
            frame_h, frame_w = frame.shape[:2]
            center_x, center_y = x + w // 2, y + h // 2
            
            # Initial expansion with smart margins
            x1_expanded = max(0, x - int(percentage_margin_w))
            y1_expanded = max(0, y - int(percentage_margin_h))
            x2_expanded = min(frame_w, x + w + int(percentage_margin_w))
            y2_expanded = min(frame_h, y + h + int(percentage_margin_h))
            
            # Ensure minimum crop dimensions by expanding from center if needed
            current_width = x2_expanded - x1_expanded
            current_height = y2_expanded - y1_expanded
            
            if current_width < min_crop_width:
                needed_expansion = (min_crop_width - current_width) // 2
                x1_expanded = max(0, x1_expanded - needed_expansion)
                x2_expanded = min(frame_w, x2_expanded + needed_expansion)
            
            if current_height < min_crop_height:
                needed_expansion = (min_crop_height - current_height) // 2
                y1_expanded = max(0, y1_expanded - needed_expansion)
                y2_expanded = min(frame_h, y2_expanded + needed_expansion)
            
            x1, y1, x2, y2 = x1_expanded, y1_expanded, x2_expanded, y2_expanded
            
            motion_regions.append((x1, y1, x2, y2))
            
            if len(motion_regions) >= config.max_regions_per_frame:
                break
        
        return motion_regions
    
    def detect_motion_regions_with_subtractor(self, frame: np.ndarray, bg_subtractor, config: ProcessingConfig) -> List[Tuple[int, int, int, int]]:
        """Detect motion regions using a specific background subtractor and config."""
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < config.min_motion_area or area > config.max_motion_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by dimensions and aspect ratio
            if (w < config.min_region_width or 
                h < config.min_region_height):
                continue
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > config.max_aspect_ratio:
                continue
            
            # Expand region with smart margin for better ML context
            base_margin = config.motion_margin
            
            # Smart margin calculation: larger context for better crops
            percentage_margin_w = max(base_margin, w * 0.75)
            percentage_margin_h = max(base_margin, h * 0.75)
            
            # Ensure minimum crop size for ML effectiveness
            min_crop_width = 150
            min_crop_height = 150
            
            # Calculate expanded region
            frame_h, frame_w = frame.shape[:2]
            center_x, center_y = x + w // 2, y + h // 2
            
            # Initial expansion with smart margins
            x1_expanded = max(0, x - int(percentage_margin_w))
            y1_expanded = max(0, y - int(percentage_margin_h))
            x2_expanded = min(frame_w, x + w + int(percentage_margin_w))
            y2_expanded = min(frame_h, y + h + int(percentage_margin_h))
            
            # Ensure minimum crop size by expanding around center if needed
            current_width = x2_expanded - x1_expanded
            current_height = y2_expanded - y1_expanded
            
            if current_width < min_crop_width:
                expansion_needed = (min_crop_width - current_width) // 2
                x1_expanded = max(0, x1_expanded - expansion_needed)
                x2_expanded = min(frame_w, x2_expanded + expansion_needed)
                
            if current_height < min_crop_height:
                expansion_needed = (min_crop_height - current_height) // 2
                y1_expanded = max(0, y1_expanded - expansion_needed)
                y2_expanded = min(frame_h, y2_expanded + expansion_needed)
            
            motion_regions.append((x1_expanded, y1_expanded, x2_expanded, y2_expanded))
            
            # Limit number of regions to avoid processing too many
            if len(motion_regions) >= config.max_regions_per_frame:
                break
        
        return motion_regions