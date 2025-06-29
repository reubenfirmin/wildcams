#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "python-dotenv>=1.0.0",
#   "opencv-python>=4.8.0",
#   "ultralytics>=8.0.0",
#   "scikit-learn>=1.3.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
#   "tqdm>=4.66.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "transformers>=4.35.0",
#   "pybioclip>=0.1.0",
#   "pytorchwildlife>=1.0.0"
# ]
# ///
"""
Next Generation Wildlife Video Processor.
Combines motion detection with temporal consistency tracking and full-frame validation.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from tqdm import tqdm
from dataclasses import dataclass
import math

# Import base processor
from video_processor_base import VideoProcessorBase

# Get loggers
logger = logging.getLogger('wildcams')
analysis_logger = logger

@dataclass
class Detection:
    """A single animal detection with metadata."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frame_idx: int
    timestamp: float
    source: str
    motion_area: Optional[float] = None

@dataclass
class TrackingInfo:
    """Tracking information for temporal consistency."""
    track_id: int
    detections: List[Detection]
    start_frame: int
    end_frame: int
    duration_seconds: float
    consistency_score: float
    
    def center(self, detection: Detection) -> Tuple[float, float]:
        """Get center point of detection bbox."""
        x1, y1, x2, y2 = detection.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def movement_distance(self) -> float:
        """Calculate total movement distance of track."""
        if len(self.detections) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.detections)):
            prev_center = self.center(self.detections[i-1])
            curr_center = self.center(self.detections[i])
            distance = math.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            total_distance += distance
        return total_distance

class NextGenVideoProcessor(VideoProcessorBase):
    """Next generation processor with temporal consistency and full-frame validation."""
    
    def __init__(self):
        super().__init__()
        
        # Temporal consistency parameters
        self.min_track_duration = float(os.getenv('MIN_TRACK_DURATION', '2.0'))  # seconds
        self.max_skip_frames = int(os.getenv('MAX_SKIP_FRAMES', '3'))  # frames
        self.tracking_distance_threshold = float(os.getenv('TRACKING_DISTANCE_THRESHOLD', '100.0'))  # pixels
        self.full_frame_validation_frames = int(os.getenv('FULL_FRAME_VALIDATION_FRAMES', '5'))  # consecutive frames
        
        # Motion detection configuration - inherited from motion detection processor
        self.motion_config = {
            'method': os.getenv('MOTION_METHOD', 'MOG2'),
            'var_threshold': int(os.getenv('MOTION_VAR_THRESHOLD', '32')),
            'min_area': int(os.getenv('MIN_MOTION_AREA', '2000')),
            'max_area': int(os.getenv('MAX_MOTION_AREA', '80000')),
            'detect_shadows': True,
            'history': int(os.getenv('MOTION_HISTORY', '100')),
            'max_regions_per_frame': int(os.getenv('MAX_REGIONS_PER_FRAME', '10')),
            'min_region_width': int(os.getenv('MIN_REGION_WIDTH', '30')),
            'min_region_height': int(os.getenv('MIN_REGION_HEIGHT', '30')),
            'max_aspect_ratio': float(os.getenv('MAX_ASPECT_RATIO', '5.0')),
            'motion_margin': int(os.getenv('MOTION_MARGIN', '30'))
        }
        
        # Initialize motion detection algorithm
        self.bg_subtractor = None
        self.init_motion_detector()
        
        # Tracking state
        self.active_tracks = []
        self.next_track_id = 0
        
        logger.info(f"🎯 Next Generation video processor initialized")
        logger.info(f"🕒 Temporal consistency: min {self.min_track_duration}s, skip {self.max_skip_frames} frames")
        logger.info(f"🔍 Motion method: {self.motion_config['method']}")
        logger.info(f"✅ Full-frame validation: {self.full_frame_validation_frames} consecutive frames")
    
    def init_motion_detector(self):
        """Initialize motion detection algorithm."""
        if self.motion_config['method'] == 'MOG2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=self.motion_config['detect_shadows'],
                varThreshold=self.motion_config['var_threshold'],
                history=self.motion_config['history']
            )
        elif self.motion_config['method'] == 'KNN':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                detectShadows=self.motion_config['detect_shadows'],
                dist2Threshold=400,
                history=self.motion_config['history']
            )
        else:
            raise ValueError(f"Unknown motion detection method: {self.motion_config['method']}")
        
        analysis_logger.info(f"Motion detector initialized: {self.motion_config['method']}")
    
    def _open_video_stream(self, video_path: Path) -> Optional[cv2.VideoCapture]:
        """Open video stream with fallback backends."""
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(str(video_path), backend)
            if cap.isOpened():
                return cap
            cap.release()
        
        logger.error(f"❌ Could not open video with any backend: {video_path}")
        return None
    
    def detect_motion_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect motion regions in frame using background subtraction."""
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
            if area < self.motion_config['min_area'] or area > self.motion_config['max_area']:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by dimensions and aspect ratio
            if (w < self.motion_config['min_region_width'] or 
                h < self.motion_config['min_region_height']):
                continue
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > self.motion_config['max_aspect_ratio']:
                continue
            
            # Expand region with margin
            margin = self.motion_config['motion_margin']
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame_w, x + w + margin)
            y2 = min(frame_h, y + h + margin)
            
            motion_regions.append((x1, y1, x2, y2))
            
            if len(motion_regions) >= self.motion_config['max_regions_per_frame']:
                break
        
        return motion_regions
    
    def run_ml_on_region(self, frame: np.ndarray, region: Tuple[int, int, int, int], 
                        frame_idx: int, timestamp: float) -> List[Detection]:
        """Run ML ensemble on motion region crop."""
        x1, y1, x2, y2 = region
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop = frame[y1:y2, x1:x2]
        
        # Run ML ensemble on crop
        crop_detections = self.ml_ensemble.run_ensemble_detection(crop, timestamp)
        
        # Convert crop coordinates to full frame coordinates
        detections = []
        for det in crop_detections:
            # Adjust bounding box coordinates
            crop_x1, crop_y1, crop_x2, crop_y2 = det['bbox']
            full_x1 = x1 + crop_x1
            full_y1 = y1 + crop_y1  
            full_x2 = x1 + crop_x2
            full_y2 = y1 + crop_y2
            
            detection = Detection(
                bbox=(full_x1, full_y1, full_x2, full_y2),
                confidence=det['confidence'],
                frame_idx=frame_idx,
                timestamp=timestamp,
                source=det['source'],
                motion_area=(x2-x1) * (y2-y1)
            )
            detections.append(detection)
        
        return detections
    
    def associate_detections_to_tracks(self, detections: List[Detection], fps: float) -> None:
        """Associate new detections to existing tracks or create new tracks."""
        unmatched_detections = list(detections)
        
        # Try to match detections to active tracks
        for track in self.active_tracks:
            if not track.detections:
                continue
                
            last_detection = track.detections[-1]
            last_center = track.center(last_detection)
            
            best_match = None
            best_distance = float('inf')
            
            for detection in unmatched_detections:
                curr_center = track.center(detection)
                distance = math.sqrt(
                    (curr_center[0] - last_center[0])**2 + 
                    (curr_center[1] - last_center[1])**2
                )
                
                if distance < self.tracking_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = detection
            
            if best_match:
                track.detections.append(best_match)
                track.end_frame = best_match.frame_idx
                track.duration_seconds = (track.end_frame - track.start_frame) / fps
                unmatched_detections.remove(best_match)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = TrackingInfo(
                track_id=self.next_track_id,
                detections=[detection],
                start_frame=detection.frame_idx,
                end_frame=detection.frame_idx,
                duration_seconds=0.0,
                consistency_score=1.0
            )
            self.active_tracks.append(new_track)
            self.next_track_id += 1
    
    def cleanup_old_tracks(self, current_frame: int) -> None:
        """Remove tracks that haven't been updated recently."""
        self.active_tracks = [
            track for track in self.active_tracks
            if current_frame - track.end_frame <= self.max_skip_frames
        ]
    
    def get_valid_tracks(self) -> List[TrackingInfo]:
        """Get tracks that meet temporal consistency requirements."""
        valid_tracks = []
        for track in self.active_tracks:
            if (track.duration_seconds >= self.min_track_duration and 
                len(track.detections) >= 2):
                valid_tracks.append(track)
        return valid_tracks
    
    def find_consistent_motion_sequences(self, video_path: Path, fps: float, total_frames: int) -> List[Dict]:
        """STEP 1: Find sequences with consistent motion/tracking."""
        cap = self._open_video_stream(video_path)
        if not cap:
            return []
        
        # Reset motion detector for this video
        self.init_motion_detector()
        
        motion_sequences = []
        current_sequence = None
        
        analysis_logger.info(f"Analyzing motion across {total_frames} frames")
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            motion_regions = self.detect_motion_regions(frame)
            
            if motion_regions:
                if current_sequence is None:
                    # Start new sequence
                    current_sequence = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'start_timestamp': timestamp,
                        'end_timestamp': timestamp,
                        'motion_regions': [motion_regions],
                        'frames': [frame_idx]
                    }
                else:
                    # Continue current sequence
                    current_sequence['end_frame'] = frame_idx
                    current_sequence['end_timestamp'] = timestamp
                    current_sequence['motion_regions'].append(motion_regions)
                    current_sequence['frames'].append(frame_idx)
            else:
                # No motion - end current sequence if it meets duration requirement
                if current_sequence is not None:
                    duration = current_sequence['end_timestamp'] - current_sequence['start_timestamp']
                    if duration >= self.min_track_duration:
                        motion_sequences.append(current_sequence)
                        analysis_logger.info(f"Motion sequence found: frames {current_sequence['start_frame']}-{current_sequence['end_frame']} ({duration:.2f}s)")
                    current_sequence = None
        
        # Handle sequence that extends to end of video
        if current_sequence is not None:
            duration = current_sequence['end_timestamp'] - current_sequence['start_timestamp']
            if duration >= self.min_track_duration:
                motion_sequences.append(current_sequence)
                analysis_logger.info(f"Motion sequence found: frames {current_sequence['start_frame']}-{current_sequence['end_frame']} ({duration:.2f}s)")
        
        cap.release()
        return motion_sequences
    
    def extract_consistent_bboxes(self, video_path: Path, motion_sequences: List[Dict]) -> List[Dict]:
        """STEP 2: Extract all bboxes that track consistently in motion regions."""
        consistent_bbox_sequences = []
        
        for seq_idx, motion_seq in enumerate(motion_sequences):
            analysis_logger.info(f"Extracting bboxes from motion sequence {seq_idx+1}/{len(motion_sequences)}")
            
            # Track bboxes across this motion sequence
            bbox_tracks = []
            
            for frame_idx, motion_regions in zip(motion_seq['frames'], motion_seq['motion_regions']):
                # For each motion region, create a potential bbox
                for region in motion_regions:
                    x1, y1, x2, y2 = region
                    
                    # Try to associate this region to existing bbox tracks
                    associated = False
                    for track in bbox_tracks:
                        last_bbox = track['bboxes'][-1]
                        last_center = ((last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2)
                        curr_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        
                        distance = math.sqrt((curr_center[0] - last_center[0])**2 + (curr_center[1] - last_center[1])**2)
                        
                        if distance < self.tracking_distance_threshold:
                            track['bboxes'].append(region)
                            track['frames'].append(frame_idx)
                            track['end_frame'] = frame_idx
                            associated = True
                            break
                    
                    if not associated:
                        # Create new bbox track
                        bbox_tracks.append({
                            'track_id': len(bbox_tracks),
                            'start_frame': frame_idx,
                            'end_frame': frame_idx,
                            'frames': [frame_idx],
                            'bboxes': [region]
                        })
            
            # Filter bbox tracks that have sufficient temporal consistency
            for track in bbox_tracks:
                duration_frames = track['end_frame'] - track['start_frame'] + 1
                duration_seconds = duration_frames / motion_seq['start_timestamp'] if motion_seq['start_timestamp'] > 0 else duration_frames / 30.0  # fallback fps
                
                if duration_seconds >= self.min_track_duration and len(track['bboxes']) >= 3:
                    consistent_bbox_sequences.append({
                        'motion_sequence_idx': seq_idx,
                        'bbox_track': track,
                        'duration_seconds': duration_seconds,
                        'consistency_score': len(track['bboxes']) / duration_frames  # How many frames had this bbox
                    })
                    analysis_logger.info(f"Consistent bbox track found: {len(track['bboxes'])} detections over {duration_seconds:.2f}s")
        
        return consistent_bbox_sequences
    
    def run_ml_on_bbox_crops(self, video_path: Path, consistent_bboxes: List[Dict]) -> List[Dict]:
        """STEP 3: Run ENHANCED ML pipeline on crops with TTA, multi-scale, and DeepSORT tracking."""
        cap = self._open_video_stream(video_path)
        if not cap:
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        validated_sequences = []
        
        # Initialize DeepSORT tracker for this video
        try:
            from deepsort_tracker import EnhancedDeepSortTracker
            deepsort_tracker = EnhancedDeepSortTracker(
                max_age=30,        # Wildlife animals may move slowly
                n_init=2,          # Confirm tracks faster for camera traps
                max_iou_distance=0.8,  # More permissive for animal movement
                max_cosine_distance=0.5  # Appearance can vary with lighting
            )
            deepsort_available = True
            analysis_logger.info("🎯 DeepSORT tracker initialized for enhanced accuracy")
        except ImportError:
            analysis_logger.warning("⚠️ DeepSORT not available, using simple tracking")
            deepsort_tracker = None
            deepsort_available = False
        
        for bbox_seq in consistent_bboxes:
            track = bbox_seq['bbox_track']
            analysis_logger.info(f"🚀 ENHANCED ML PROCESSING: bbox track {track['track_id']} ({len(track['bboxes'])} crops)")
            
            all_detections = []
            enhancement_stats = {
                'total_crops_processed': 0,
                'enhanced_detections_found': 0,
                'deepsort_tracks_created': 0,
                'high_confidence_detections': 0
            }
            
            # Run enhanced ML on each bbox crop in this track
            for frame_idx, bbox in zip(track['frames'], track['bboxes']):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                timestamp = frame_idx / fps
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                crop = frame[y1:y2, x1:x2]
                
                enhancement_stats['total_crops_processed'] += 1
                
                # Run ENHANCED ML ensemble on crop with TTA, multi-scale, and advanced NMS
                analysis_logger.info(f"🎯 Running enhanced ensemble on crop {frame_idx} (size: {crop.shape})")
                crop_detections = self.ml_ensemble.run_ensemble_detection(crop, timestamp, frame_idx)
                enhancement_stats['enhanced_detections_found'] += len(crop_detections)
                
                for det in crop_detections:
                    # Adjust bbox coordinates back to full frame
                    crop_x1, crop_y1, crop_x2, crop_y2 = det['bbox']
                    full_bbox = [x1 + crop_x1, y1 + crop_y1, x1 + crop_x2, y1 + crop_y2]
                    
                    detection = {
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'bbox': full_bbox,
                        'confidence': det['confidence'],
                        'source': f"enhanced_{det['source']}",
                        'augmentation': det.get('augmentation', 'original'),
                        'scale': det.get('scale', 1.0),
                        'enhancement_stats': det.get('enhancement_stats', {}),
                        'model_threshold': det.get('model_threshold', self.confidence_threshold)
                    }
                    all_detections.append(detection)
                    
                    # Count high confidence detections (using lower threshold due to enhanced detection)
                    if det['confidence'] >= self.confidence_threshold * 1.5:  # Lower threshold for enhanced detection
                        enhancement_stats['high_confidence_detections'] += 1
            
            analysis_logger.info(f"📊 Enhancement statistics for track {track['track_id']}:")
            analysis_logger.info(f"  🎯 Crops processed: {enhancement_stats['total_crops_processed']}")
            analysis_logger.info(f"  🔍 Enhanced detections: {enhancement_stats['enhanced_detections_found']}")
            analysis_logger.info(f"  ⭐ High confidence: {enhancement_stats['high_confidence_detections']}")
            
            # Apply DeepSORT tracking if available
            if deepsort_available and deepsort_tracker and all_detections:
                analysis_logger.info(f"🎯 Applying DeepSORT tracking to {len(all_detections)} detections")
                
                # Group detections by frame for DeepSORT processing
                frame_detections = {}
                for det in all_detections:
                    frame_idx = det['frame_idx']
                    if frame_idx not in frame_detections:
                        frame_detections[frame_idx] = []
                    frame_detections[frame_idx].append(det)
                
                tracked_detections = []
                for frame_idx in sorted(frame_detections.keys()):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_dets = frame_detections[frame_idx]
                        tracked_frame_dets = deepsort_tracker.update_tracks(frame, frame_dets, frame_idx)
                        tracked_detections.extend(tracked_frame_dets)
                
                # Get valid DeepSORT tracks
                valid_deepsort_tracks = deepsort_tracker.get_valid_tracks(
                    min_duration_frames=5,  # Lower for crops
                    min_consistency_score=0.5  # Lower threshold for enhanced accuracy
                )
                
                enhancement_stats['deepsort_tracks_created'] = len(valid_deepsort_tracks)
                analysis_logger.info(f"🎯 DeepSORT created {len(valid_deepsort_tracks)} valid tracks")
                
                # Use DeepSORT results if available, otherwise fall back to original detections
                if valid_deepsort_tracks:
                    best_track = max(valid_deepsort_tracks, key=lambda t: t['consistency_score'])
                    best_detection = best_track['best_detection']
                    
                    validated_sequences.append({
                        'bbox_sequence': bbox_seq,
                        'detections': tracked_detections,
                        'deepsort_tracks': valid_deepsort_tracks,
                        'best_track': best_track,
                        'high_confidence_count': enhancement_stats['high_confidence_detections'],
                        'consistency_ratio': best_track['consistency_score'],
                        'best_detection': {
                            'frame_idx': best_detection['frame_idx'],
                            'timestamp': best_detection['timestamp'],
                            'bbox': best_detection['bbox'],
                            'confidence': best_detection['confidence'],
                            'source': f"deepsort_enhanced_{best_detection.get('source', 'unknown')}"
                        },
                        'confidence_score': best_detection['confidence'] * best_track['consistency_score'],
                        'duration_seconds': best_track['duration_seconds'],
                        'enhancement_stats': enhancement_stats,
                        'tracking_method': 'deepsort'
                    })
                    analysis_logger.info(f"✅ Track {track['track_id']} VALIDATED via DeepSORT: consistency={best_track['consistency_score']:.3f}")
                    continue
            
            # Fallback to original validation logic if DeepSORT not available or no valid tracks
            if all_detections:
                high_confidence_count = enhancement_stats['high_confidence_detections']
                consistency_ratio = high_confidence_count / len(track['frames']) if track['frames'] else 0
                
                # Lower thresholds due to enhanced detection providing better quality
                if consistency_ratio >= 0.05 and high_confidence_count >= 1:  # Very permissive for small animals like Video 7
                    best_detection = max(all_detections, key=lambda d: d['confidence'])
                    
                    validated_sequences.append({
                        'bbox_sequence': bbox_seq,
                        'detections': all_detections,
                        'high_confidence_count': high_confidence_count,
                        'consistency_ratio': consistency_ratio,
                        'best_detection': best_detection,
                        'confidence_score': best_detection['confidence'] * consistency_ratio,
                        'duration_seconds': bbox_seq['duration_seconds'],
                        'enhancement_stats': enhancement_stats,
                        'tracking_method': 'enhanced_fallback'
                    })
                    analysis_logger.info(f"✅ Track {track['track_id']} VALIDATED via enhanced fallback: {high_confidence_count}/{len(track['frames'])} high conf ({consistency_ratio:.2f})")
                else:
                    analysis_logger.info(f"❌ Track {track['track_id']} REJECTED: {high_confidence_count}/{len(track['frames'])} high conf ({consistency_ratio:.2f})")
            else:
                analysis_logger.info(f"❌ Track {track['track_id']} REJECTED: no enhanced detections found")
        
        cap.release()
        
        # Log overall enhancement impact
        total_enhancement_stats = {
            'total_tracks_processed': len(consistent_bboxes),
            'validated_sequences': len(validated_sequences),
            'deepsort_validations': len([s for s in validated_sequences if s.get('tracking_method') == 'deepsort']),
            'fallback_validations': len([s for s in validated_sequences if s.get('tracking_method') == 'enhanced_fallback'])
        }
        
        analysis_logger.info(f"🚀 ENHANCED ML PROCESSING COMPLETE:")
        analysis_logger.info(f"  📊 Tracks processed: {total_enhancement_stats['total_tracks_processed']}")
        analysis_logger.info(f"  ✅ Validated sequences: {total_enhancement_stats['validated_sequences']}")
        analysis_logger.info(f"  🎯 DeepSORT validations: {total_enhancement_stats['deepsort_validations']}")
        analysis_logger.info(f"  🔄 Enhanced fallback: {total_enhancement_stats['fallback_validations']}")
        
        return validated_sequences
    
    def run_full_frame_validation_on_sequences(self, video_path: Path, validated_sequences: List[Dict]) -> List[Dict]:
        """STEP 4: Full-frame validation on N frames per validated sequence."""
        cap = self._open_video_stream(video_path)
        if not cap:
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        final_validated = []
        
        for seq in validated_sequences:
            track = seq['bbox_sequence']['bbox_track']
            analysis_logger.info(f"Full-frame validation for sequence {track['track_id']}")
            
            # Select N frames evenly distributed across the sequence for full-frame validation
            validation_frames = []
            start_frame = track['start_frame']
            end_frame = track['end_frame']
            frame_step = max(1, (end_frame - start_frame) // self.full_frame_validation_frames)
            
            for i in range(self.full_frame_validation_frames):
                frame_idx = start_frame + (i * frame_step)
                if frame_idx <= end_frame:
                    validation_frames.append(frame_idx)
            
            # Run full-frame ML on these frames
            full_frame_confirmations = 0
            
            for frame_idx in validation_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                timestamp = frame_idx / fps
                
                # Run full-frame ML ensemble
                full_frame_detections = self.ml_ensemble.run_ensemble_detection(frame, timestamp)
                
                if any(det['confidence'] >= self.confidence_threshold for det in full_frame_detections):
                    full_frame_confirmations += 1
            
            # Require majority of validation frames to confirm animal presence
            confirmation_ratio = full_frame_confirmations / len(validation_frames) if validation_frames else 0
            
            if confirmation_ratio >= 0.6:  # At least 60% of frames must confirm
                seq['full_frame_confirmations'] = full_frame_confirmations
                seq['confirmation_ratio'] = confirmation_ratio
                seq['best_frame'] = seq['best_detection']['frame_idx']
                seq['best_timestamp'] = seq['best_detection']['timestamp']
                seq['best_confidence'] = seq['best_detection']['confidence']
                seq['best_bbox'] = seq['best_detection']['bbox']
                seq['best_source'] = seq['best_detection']['source']
                
                final_validated.append(seq)
                analysis_logger.info(f"Sequence {track['track_id']} FINAL VALIDATION: {full_frame_confirmations}/{len(validation_frames)} frames confirmed ({confirmation_ratio:.2f})")
            else:
                analysis_logger.info(f"Sequence {track['track_id']} FINAL REJECTION: {full_frame_confirmations}/{len(validation_frames)} frames confirmed ({confirmation_ratio:.2f})")
        
        cap.release()
        return final_validated
    
    def process_video_with_features(self, video_path: Path) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Process video with correct pipeline: motion tracking -> bbox extraction -> ML on crops -> full-frame validation."""
        analysis_logger.info(f"=== NEXT-GEN VIDEO PROCESSING START: {video_path.name} ===")
        
        cap = self._open_video_stream(video_path)
        if not cap:
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create debug directory
        video_debug_dir = self.debug_dir / video_path.stem
        video_debug_dir.mkdir(exist_ok=True)
        
        # STEP 1: Find sequences with consistent motion/tracking
        analysis_logger.info(f"=== STEP 1: MOTION TRACKING PHASE START ===")
        consistent_motion_sequences = self.find_consistent_motion_sequences(video_path, fps, total_frames)
        analysis_logger.info(f"Found {len(consistent_motion_sequences)} consistent motion sequences")
        
        if not consistent_motion_sequences:
            analysis_logger.info("No consistent motion sequences found")
            cap.release()
            return None, None
        
        # STEP 2: Extract consistent bboxes from motion sequences
        analysis_logger.info(f"=== STEP 2: BBOX EXTRACTION PHASE START ===")
        consistent_bboxes = self.extract_consistent_bboxes(video_path, consistent_motion_sequences)
        analysis_logger.info(f"Extracted {len(consistent_bboxes)} consistent bbox sequences")
        
        if not consistent_bboxes:
            analysis_logger.info("No consistent bboxes extracted")
            cap.release()
            return None, None
        
        # STEP 3: Run ML pipeline on crops, filter to high confidence animals
        analysis_logger.info(f"=== STEP 3: ML ON CROPS PHASE START ===")
        validated_sequences = self.run_ml_on_bbox_crops(video_path, consistent_bboxes)
        analysis_logger.info(f"Validated {len(validated_sequences)} sequences with high confidence animals")
        
        if not validated_sequences:
            analysis_logger.info("No sequences passed ML validation on crops")
            cap.release()
            return None, None
        
        # STEP 4: Full-frame validation on N frames per validated sequence
        analysis_logger.info(f"=== STEP 4: FULL-FRAME VALIDATION PHASE START ===")
        final_validated_sequences = self.run_full_frame_validation_on_sequences(video_path, validated_sequences)
        analysis_logger.info(f"Final validation: {len(final_validated_sequences)} sequences passed full-frame validation")
        
        cap.release()
        
        if not final_validated_sequences:
            analysis_logger.info("No sequences passed full-frame validation")
            return None, None
        
        # Create final analysis from best validated sequence
        best_sequence = max(final_validated_sequences, key=lambda s: s['confidence_score'])
        
        analysis = {
            'video_path': str(video_path),
            'animals_detected': ['animal'],
            'detection_count': sum(len(seq['detections']) for seq in final_validated_sequences),
            'frames_processed': total_frames,
            'consistent_motion_sequences': len(consistent_motion_sequences),
            'validated_sequences': len(final_validated_sequences),
            'temporal_consistency_duration': best_sequence['duration_seconds'],
            'best_detection_frame': best_sequence['best_frame'],
            'best_detection_timestamp': best_sequence['best_timestamp'],
            'detection': {
                'confidence': best_sequence['best_confidence'],
                'bbox': best_sequence['best_bbox'],
                'area_ratio': self.calculate_area_ratio(best_sequence['best_bbox'], cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'source': best_sequence['best_source']
            },
            'processing_mode': 'next_generation_efficient_pipeline'
        }
        
        # Extract features from best detection
        features = self.extract_features_from_best_sequence(video_path, best_sequence)
        
        analysis_logger.info(f"=== NEXT-GEN VIDEO PROCESSING END: SUCCESS ===")
        return analysis, features
    
    def calculate_area_ratio(self, bbox: Tuple[int, int, int, int], frame_w: float, frame_h: float) -> float:
        """Calculate detection area ratio relative to frame."""
        x1, y1, x2, y2 = bbox
        detection_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_w * frame_h
        return detection_area / frame_area if frame_area > 0 else 0.0
    
    def extract_features_from_best_sequence(self, video_path: Path, best_sequence: Dict) -> Optional[np.ndarray]:
        """Extract ResNet features from best detection in sequence."""
        cap = self._open_video_stream(video_path)
        if not cap:
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_sequence['best_frame'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Crop detection region
        x1, y1, x2, y2 = best_sequence['best_bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop = frame[y1:y2, x1:x2]
        
        # Extract features using ResNet
        return self.ml_ensemble.extract_features(frame, best_sequence['best_bbox'])
    
    def process_all_videos(self, video_filter=None):
        """Process all videos using next-generation approach."""
        analysis_logger.info("###############################################")
        analysis_logger.info("NEXT-GENERATION BATCH PROCESSING SESSION START")
        analysis_logger.info("###############################################")
        
        # Get videos to process
        videos_to_process = self.get_filtered_videos(video_filter)
        
        if not videos_to_process:
            if video_filter:
                analysis_logger.info(f"BATCH RESULT: No videos found matching filter: {video_filter}")
                logger.info(f"⚠️ No videos found matching filter: {video_filter}")
            else:
                analysis_logger.info("BATCH RESULT: No unprocessed videos found")
                logger.info("✅ No unprocessed videos found")
            return
        
        analysis_logger.info(f"Videos to process: {[v.name for v in videos_to_process]}")
        logger.info(f"🎬 Found {len(videos_to_process)} videos to process")
        
        # Clear previous session data
        self.all_features = []
        self.video_metadata = []
        all_analyses = []
        
        # Process each video
        for i, video_path in enumerate(videos_to_process):
            analysis_logger.info(f"Processing video {i+1}/{len(videos_to_process)}: {video_path.name}")
            try:
                analysis, features = self.process_video_with_features(video_path)
                if analysis:
                    analysis_logger.info(f"VIDEO SUCCESS: {video_path.name} - Animal detected with temporal consistency")
                    all_analyses.append(analysis)
                    self.save_analysis(analysis, video_path)
                    
                    if features is not None:
                        self.all_features.append(features)
                        self.video_metadata.append(analysis)
                        analysis_logger.info(f"Features extracted: {len(features)} dimensions")
                else:
                    analysis_logger.info(f"VIDEO SKIPPED: {video_path.name} - No consistent animal movement detected")
                    logger.info(f"⚪ {video_path.name}: No consistent animal movement")
                    
            except Exception as e:
                analysis_logger.error(f"VIDEO ERROR: {video_path.name} - {str(e)}")
                logger.error(f"❌ {video_path.name}: Processing failed - {str(e)}")
        
        # Generate final summary
        if all_analyses:
            logger.info(f"✅ Successfully processed {len(all_analyses)} videos with temporal consistency")
            logger.info("📊 Analysis files already saved for each video")
            
            # Run clustering if we have features
            if len(self.all_features) > 1:
                logger.info("🔗 Running similarity clustering...")
                self.cluster_similar_videos()
        else:
            logger.info("⚪ No videos contained consistent animal movement")
        
        analysis_logger.info("###############################################")
        analysis_logger.info("NEXT-GENERATION BATCH PROCESSING SESSION END")
        analysis_logger.info("###############################################")

def main():
    """Main entry point for next-generation processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Next Generation Wildlife Video Processor with Temporal Consistency')
    parser.add_argument('--videos', '-v', nargs='+', help='Optional list of video indices (e.g. 7 8 9) or names to process')
    
    # Add common arguments from base class
    VideoProcessorBase.setup_common_arguments(parser)
    
    # Add motion detection arguments
    VideoProcessorBase.setup_motion_detection_arguments(parser)
    
    # Add temporal consistency arguments
    parser.add_argument('--min-track-duration', type=float, default=2.0,
                       help='Minimum track duration in seconds (default: 2.0)')
    parser.add_argument('--max-skip-frames', type=int, default=3,
                       help='Maximum frames to skip in tracking (default: 3)')
    parser.add_argument('--tracking-distance-threshold', type=float, default=100.0,
                       help='Maximum distance for tracking association in pixels (default: 100.0)')
    parser.add_argument('--full-frame-validation-frames', type=int, default=5,
                       help='Consecutive frames needed for full-frame validation (default: 5)')
    
    args = parser.parse_args()
    
    # Convert video arguments
    video_filter = None
    if args.videos:
        video_filter = []
        for video in args.videos:
            try:
                video_filter.append(int(video))
            except ValueError:
                video_filter.append(video)
    
    # Set environment variables from arguments
    VideoProcessorBase.set_environment_from_args(args, include_motion=True)
    
    # Set temporal consistency parameters
    os.environ['MIN_TRACK_DURATION'] = str(args.min_track_duration) 
    os.environ['MAX_SKIP_FRAMES'] = str(args.max_skip_frames)
    os.environ['TRACKING_DISTANCE_THRESHOLD'] = str(args.tracking_distance_threshold)
    os.environ['FULL_FRAME_VALIDATION_FRAMES'] = str(args.full_frame_validation_frames)
    
    try:
        processor = NextGenVideoProcessor()
        print(f"🎬 Starting Next Generation wildlife video processing...")
        print(f"📊 Mode: Motion detection + temporal consistency + full-frame validation")
        print(f"🕒 Temporal parameters: {args.min_track_duration}s duration, {args.max_skip_frames} skip frames")
        
        logger.info(f"🎯 Processing strategy: Next Generation Temporal Consistency")
        logger.info(f"🕒 Min track duration: {args.min_track_duration}s")
        logger.info(f"✅ Full-frame validation frames: {args.full_frame_validation_frames}")
        
        processor.process_all_videos(video_filter=video_filter)
        
    except KeyboardInterrupt:
        print("🛑 Processing interrupted by user")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()