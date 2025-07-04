#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "opencv-python>=4.8.0",
#   "ultralytics>=8.0.0",
#   "scikit-learn>=1.3.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
#   "tqdm>=4.66.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0"
# ]
# ///
"""
Base video processor with common functionality shared between process.py and process2.py.
Contains video I/O, analysis management, clustering, and reporting functions.
"""

import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Import shared ML detection module
from ml import MLDetectionEnsemble

# Import new video I/O modules
from video_io import FrameExtractor, AnalysisWriter, ProcessedVideoTracker

# Import configuration
from config import ProcessingConfig


# Get loggers
logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger('analysis')

class VideoProcessorBase:
    """Base class for wildlife video processing with common functionality."""
    
    def __init__(self, config: ProcessingConfig):
        
        # Directory setup
        self.video_dir = Path(config.video_dir)
        self.output_dir = self.video_dir / 'analysis'
        self.debug_dir = self.output_dir / 'debug_frames'
        self.logs_dir = Path('./logs')
        self.models_cache_dir = Path('./models_cache')
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.models_cache_dir.mkdir(exist_ok=True)
        
        # Setup logging for this session
        self.setup_logging()
        
        # Clean up debug directory from previous runs
        if self.debug_dir.exists():
            import shutil
            shutil.rmtree(self.debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        
        
        # Store all extracted features for clustering
        self.all_features = []
        self.video_metadata = []
        
        # Initialize I/O components
        self.frame_extractor = FrameExtractor(
            max_frames=config.max_frames_per_video,
            sampling_strategy='uniform',
            debug_dir=self.debug_dir
        )
        self.analysis_writer = AnalysisWriter(self.output_dir)
        self.processed_tracker = ProcessedVideoTracker(self.video_dir)
        
        # Initialize ML detection ensemble after all parameters are set
        self.ml_ensemble = MLDetectionEnsemble(
            confidence_threshold=config.confidence_threshold,
            ensemble_models=config.ensemble_models,
            cache_dir=self.models_cache_dir
        )
        
        # Log model configuration after everything is initialized
        self._log_model_configuration(config)
        
        logger.info(f"🎬 Video processor base initialized")
        logger.info(f"📁 Video directory: {self.video_dir}")
        logger.info(f"📊 Analysis output: {self.output_dir}")
        logger.info(f"📋 Logs directory: {self.logs_dir}")
    
    def setup_logging(self):
        """Setup single logger to file and console."""
        # Create timestamp for this session
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup single logger
        global logger, analysis_logger
        logger = logging.getLogger('wildcams')
        analysis_logger = logger  # Use same logger
        
        # Remove any existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create log file handler
        log_file = self.logs_dir / f'wildcams_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.setLevel(logging.DEBUG)
        
        # Log the setup
        logger.info(f"📋 Logging initialized - session {timestamp}")
        logger.info(f"📋 Log file: {log_file}")
    
    def _log_model_configuration(self, config: ProcessingConfig):
        """Log the model configuration."""
        logger.info("=" * 80)
        logger.info("🎯 MODEL CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Ensemble Models: {', '.join(config.ensemble_models)}")
        logger.info(f"Confidence Threshold: {config.confidence_threshold}")
        logger.info("=" * 80)
    
    def get_unprocessed_videos(self) -> List[Path]:
        """Get list of videos that haven't been processed yet."""
        return self.processed_tracker.get_unprocessed_videos()
    
    def get_filtered_videos(self, video_filter: List) -> List[Path]:
        """Get filtered videos, ignoring .processed status when filter is provided."""
        if video_filter:
            analysis_logger.info("OVERRIDE: Video filter provided - ignoring .processed files and forcing reprocessing")
            logger.info("🔄 Filter provided - forcing reprocessing of specified videos")
            
            # Convert to string list for ProcessedVideoTracker
            filter_strings = [str(item) for item in video_filter]
            return self.processed_tracker.get_filtered_videos(filter_strings)
        else:
            # No filter provided - use normal unprocessed video logic
            return self.get_unprocessed_videos()
    
    def extract_frames(self, video_path: Path) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames evenly distributed throughout the video for analysis."""
        return self.frame_extractor.extract_frames_from_path(video_path)
    
    def detect_camera_handling(self, all_detections: List[Dict], frames_processed: int, config: ProcessingConfig) -> bool:
        """
        Detect if this appears to be camera handling/equipment rather than animals.
        Returns True if likely camera handling (should be rejected).
        """
        if not all_detections:
            return False
        
        total_detections = len(all_detections)
        detection_density = total_detections / frames_processed
        
        # Count low confidence detections
        low_conf_detections = [d for d in all_detections if d['confidence'] < config.low_confidence_cutoff]
        low_conf_ratio = len(low_conf_detections) / total_detections
        
        analysis_logger.info(f"CAMERA HANDLING CHECK:")
        analysis_logger.info(f"  Detection density: {detection_density:.2f} (threshold: {config.detection_density_threshold})")
        analysis_logger.info(f"  Low confidence ratio: {low_conf_ratio:.3f} (threshold: {config.low_confidence_ratio_threshold})")
        analysis_logger.info(f"  Total detections: {total_detections}, Frames: {frames_processed}")
        
        # High detection density + lots of low confidence = likely camera handling
        if (detection_density > config.detection_density_threshold and 
            low_conf_ratio > config.low_confidence_ratio_threshold):
            analysis_logger.info("CAMERA HANDLING DETECTED: High density + low confidence mass")
            return True
        
        # Very high detection density alone is suspicious 
        if detection_density > config.detection_density_threshold * 2:
            analysis_logger.info("CAMERA HANDLING DETECTED: Extremely high detection density")
            return True
        
        return False
    
    def ensemble_validation(self, all_detections: List[Dict], config: ProcessingConfig) -> bool:
        """
        Validate detections using ensemble logic across multiple models.
        Returns True if evidence supports real animal presence.
        """
        if not all_detections:
            analysis_logger.info("ENSEMBLE VALIDATION: No detections - rejected")
            return False
        
        # Count detections by source
        source_counts = {}
        high_conf_detections = []
        
        for detection in all_detections:
            source = detection.get('source', 'unknown')
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
            
            # Track high confidence detections
            conf = detection['confidence']
            if source.startswith('megadetector') and conf >= config.megadetector_high_confidence:
                high_conf_detections.append(detection)
            elif source.startswith(('primary', 'backup')) and conf >= config.yolo_high_confidence:
                high_conf_detections.append(detection)
            elif source.startswith(('deepfaune', 'wildlife')) and conf >= config.wildlife_model_confidence:
                high_conf_detections.append(detection)
        
        analysis_logger.info(f"ENSEMBLE VALIDATION: Source counts: {source_counts}")
        analysis_logger.info(f"ENSEMBLE VALIDATION: High confidence detections: {len(high_conf_detections)}")
        
        # Strong evidence: Multiple models agree OR high confidence detections
        strong_evidence = len(source_counts) >= 2 or len(high_conf_detections) >= 2
        
        # Moderate evidence: YOLO models with sufficient detections
        yolo_detections = sum(source_counts.get(s, 0) for s in source_counts if s.startswith(('primary', 'backup')))
        moderate_evidence = yolo_detections >= config.min_yolo_detections
        
        # Weak evidence: Any wildlife-specific model detection
        wildlife_evidence = any(s.startswith(('megadetector', 'deepfaune', 'wildlife')) for s in source_counts)
        
        if strong_evidence:
            analysis_logger.info("ENSEMBLE VALIDATION: STRONG evidence - multiple models agree")
            return True
        elif moderate_evidence:
            analysis_logger.info("ENSEMBLE VALIDATION: MODERATE evidence - sufficient YOLO detections")
            return True
        elif wildlife_evidence:
            # Check if weak evidence meets threshold
            avg_confidence = np.mean([d['confidence'] for d in all_detections])
            if avg_confidence >= config.weak_evidence_threshold:
                analysis_logger.info(f"ENSEMBLE VALIDATION: WEAK evidence accepted - avg conf {avg_confidence:.3f} >= {config.weak_evidence_threshold}")
                return True
            else:
                analysis_logger.info(f"ENSEMBLE VALIDATION: WEAK evidence rejected - avg conf {avg_confidence:.3f} < {config.weak_evidence_threshold}")
                return False
        else:
            analysis_logger.info("ENSEMBLE VALIDATION: Insufficient evidence - rejected")
            return False
    
    def score_and_find_best_detection(self, all_detections: List[Dict]) -> Tuple[Optional[Dict], int]:
        """Score all detections and find the best one for feature extraction."""
        if not all_detections:
            return None, 0
        
        best_detection = None
        best_score = 0.0
        total_scored = 0
        
        analysis_logger.info("=== DETECTION SCORING START ===")
        
        for detection in all_detections:
            # Calculate bounding box area and size ratio
            bbox = detection['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = 1600 * 900  # Typical frame size
            size_ratio = area / frame_area
            
            # Score based on confidence, size, and source reliability
            base_score = detection['confidence'] * size_ratio
            
            # Apply source-specific multipliers
            source = detection.get('source', 'unknown')
            if source.startswith('megadetector'):
                multiplier = 1.5  # Wildlife-specific model bonus
            elif source.startswith('deepfaune'):
                multiplier = 1.4  # Specialized wildlife model
            elif source.startswith('primary'):
                multiplier = 1.2  # Primary model bonus
            elif source.startswith('backup'):
                multiplier = 1.1  # Backup model
            else:
                multiplier = 1.0  # Default
            
            final_score = base_score * multiplier
            
            # Enhanced detection gets additional boost
            if 'enhanced' in source:
                final_score *= 1.2
            
            analysis_logger.info(f"SCORING: frame_{detection.get('frame_idx', '?')}, source={source}, conf={detection['confidence']:.4f}, area={area}, size_ratio={size_ratio:.4f}, multiplier={multiplier}, score={final_score:.6f}")
            
            if final_score > best_score:
                analysis_logger.info(f"NEW BEST: previous_score={best_score:.6f}, new_score={final_score:.6f}, frame={detection.get('frame_idx', '?')}, source={source}")
                best_score = final_score
                best_detection = detection
            
            total_scored += 1
        
        analysis_logger.info(f"=== DETECTION SCORING END: {total_scored} detections scored, best_score={best_score:.6f} ===")
        return best_detection, total_scored
    
    def extract_features_from_detection(self, frame: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """Extract ResNet18 features from the best detection."""
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                analysis_logger.warning(f"Invalid detection bbox after clipping: {bbox}")
                return None
            
            # Extract region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                analysis_logger.warning("Empty crop region")
                return None
            
            # Use ML ensemble feature extraction
            features = self.ml_ensemble.extract_features(frame, bbox)
            
            if features is not None:
                analysis_logger.info(f"FEATURE EXTRACTION: Successfully extracted {len(features)} features from detection")
                return features
            else:
                analysis_logger.warning("FEATURE EXTRACTION: Failed to extract features")
                return None
                
        except Exception as e:
            analysis_logger.error(f"FEATURE EXTRACTION: Error extracting features: {e}")
            return None
    
    def cluster_animal_videos(self, video_metadata: List[Dict], config: ProcessingConfig) -> Dict:
        """Cluster videos with similar animal features."""
        if len(self.all_features) < 2:
            analysis_logger.info("CLUSTERING: Not enough features for clustering")
            return {}
        
        analysis_logger.info(f"CLUSTERING: Starting with {len(self.all_features)} feature vectors")
        
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(self.all_features)
            
            # Apply PCA for dimensionality reduction if needed
            if features_scaled.shape[1] > 50:
                pca = PCA(n_components=50)
                features_scaled = pca.fit_transform(features_scaled)
                analysis_logger.info(f"CLUSTERING: Applied PCA, reduced to {features_scaled.shape[1]} dimensions")
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=config.clustering_eps, min_samples=config.min_samples)
            cluster_labels = clustering.fit_predict(features_scaled)
            
            analysis_logger.info(f"CLUSTERING: Found {len(set(cluster_labels))} clusters (including noise)")
            
            # Group videos by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'video_metadata': video_metadata[i],
                    'features': self.all_features[i]
                })
            
            # Format results
            result = {}
            for cluster_id, cluster_videos in clusters.items():
                if cluster_id == -1:
                    continue  # Skip noise
                
                result[f'cluster_{cluster_id}'] = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_videos),
                    'videos': [v['video_metadata'] for v in cluster_videos],
                    'avg_confidence': np.mean([v['video_metadata']['detection']['confidence'] for v in cluster_videos]),
                    'avg_animal_size': np.mean([v['video_metadata']['detection']['area_ratio'] for v in cluster_videos])
                }
            
            if -1 in clusters:
                result['unclustered'] = {
                    'cluster_id': -1,
                    'size': len(clusters[-1]),
                    'videos': [v['video_metadata'] for v in clusters[-1]],
                    'description': 'Videos that did not fit into any cluster'
                }
            
            return result
            
        except Exception as e:
            analysis_logger.error(f"CLUSTERING: Failed with error: {e}")
            return {}
    
    def save_clustering_results(self, clusters: Dict):
        """Save clustering results and feature data."""
        # Convert clusters dict to list format expected by AnalysisWriter
        clusters_list = [clusters] if clusters else []
        self.analysis_writer.save_clustering_results(clusters_list, self.all_features, self.video_metadata)
    
    def _debug_numpy_objects(self, obj, path=""):
        """Debug helper to find numpy objects in nested data."""
        if hasattr(obj, 'dtype'):  # numpy array/scalar
            logger.error(f"Found numpy object at {path}: type={type(obj)}, dtype={obj.dtype}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self._debug_numpy_objects(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                self._debug_numpy_objects(item, f"{path}[{i}]")

    def _convert_for_json(self, obj):
        """Convert numpy types and other non-JSON types for serialization."""
        import numpy as np
        
        # Check if object has numpy module in its type string - catch all numpy types
        obj_type_str = str(type(obj))
        if 'numpy' in obj_type_str:
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                # Last resort for numpy objects
                return float(obj)
        
        # Handle specific numpy types
        if isinstance(obj, (np.integer, np.floating, np.bool_, np.complexfloating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):  # Other numpy-like objects
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
        elif hasattr(obj, 'item') and hasattr(obj, '__float__'):  # numpy scalars without dtype
            try:
                return obj.item()
            except:
                return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj

    def save_analysis(self, analysis: Dict, video_path: Path):
        """Save analysis results to JSON file."""
        self.analysis_writer.save_analysis(video_path, analysis)
    
    def mark_as_processed(self, video_path: Path):
        """Create .processed marker file."""
        self.processed_tracker.mark_as_processed(video_path)
    
    def generate_summary_report(self, all_analyses: List[Dict], clusters: Dict):
        """Generate a summary report of all processed videos."""
        self.analysis_writer.generate_summary_report(all_analyses)