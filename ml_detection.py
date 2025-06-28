#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "python-dotenv>=1.0.0",
#   "opencv-python>=4.8.0",
#   "ultralytics>=8.0.0",
#   "scikit-learn>=1.3.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "pytorchwildlife>=1.0.0"
# ]
# ///
"""
Shared ML detection module for wildlife video processing.
Contains the ensemble detection system used by both process.py and process2.py.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# PyTorch-Wildlife imports
try:
    from PytorchWildlife.models import detection as pw_detection
    PYTORCH_WILDLIFE_AVAILABLE = True
except ImportError:
    PYTORCH_WILDLIFE_AVAILABLE = False

# Get loggers
logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger('analysis')

class MLDetectionEnsemble:
    """Ensemble ML detection system for wildlife video processing."""
    
    def __init__(self, confidence_threshold: float = 0.1):
        self.confidence_threshold = confidence_threshold
        
        # Model initialization flags
        self.detector = None
        self.detector_backup = None
        self.megadetector = None
        self.megadetector_v6 = None
        self.deepfaune_detector = None
        self.feature_extractor = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models in the ensemble."""
        try:
            logger.info("🤖 Initializing ML detection ensemble...")
            
            # Primary YOLOv8x detector
            try:
                logger.info("Loading YOLOv8x primary detector...")
                self.detector = YOLO('yolov8x.pt')
                logger.info("✅ YOLOv8x primary detector loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load YOLOv8x primary detector: {e}")
            
            # Backup YOLOv8m detector
            try:
                logger.info("Loading YOLOv8m backup detector...")
                self.detector_backup = YOLO('yolov8m.pt')
                logger.info("✅ YOLOv8m backup detector loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load YOLOv8m backup detector: {e}")
            
            # MegaDetector fallback (YOLOv8n)
            try:
                logger.info("Loading YOLOv8n as MegaDetector fallback...")
                self.megadetector = YOLO('yolov8n.pt')
                logger.info("✅ YOLOv8n MegaDetector fallback loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load YOLOv8n MegaDetector fallback: {e}")
            
            # PyTorch-Wildlife models
            if PYTORCH_WILDLIFE_AVAILABLE:
                try:
                    logger.info("🦎 Loading MegaDetector v6 (PyTorch-Wildlife)...")
                    self.megadetector_v6 = pw_detection.MegaDetectorV6()
                    logger.info("✅ MegaDetector v6 loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load MegaDetector v6: {e}")
                    self.megadetector_v6 = None
                
                try:
                    logger.info("🦎 Loading DeepFaune detector (PyTorch-Wildlife)...")
                    self.deepfaune_detector = pw_detection.DeepFaune()
                    logger.info("✅ DeepFaune detector loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load DeepFaune detector: {e}")
                    self.deepfaune_detector = None
            else:
                logger.warning("⚠️ PyTorch-Wildlife not available - MegaDetector v6 and DeepFaune disabled")
            
            # Feature extractor for clustering
            try:
                logger.info("Loading ResNet18 feature extractor...")
                self.feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.feature_extractor.eval()
                if torch.cuda.is_available():
                    self.feature_extractor = self.feature_extractor.cuda()
                logger.info("✅ ResNet18 feature extractor loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load ResNet18 feature extractor: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML ensemble: {e}")
    
    def run_ensemble_detection(
        self, 
        frame: np.ndarray, 
        timestamp_seconds: float = 0.0,
        frame_idx: int = 0
    ) -> List[Dict]:
        """
        Run the full 5-model ensemble detection on a frame.
        
        Args:
            frame: OpenCV frame (BGR format)
            timestamp_seconds: Video timestamp in seconds
            frame_idx: Frame index for logging
            
        Returns:
            List of detection dictionaries with keys: confidence, bbox, source, class
        """
        detections = []
        
        # 1. Primary YOLOv8x detector
        if self.detector is not None:
            try:
                analysis_logger.info(f"ENSEMBLE_STEP_1: Running YOLOv8x primary model with conf={self.confidence_threshold}")
                results = self.detector(frame, conf=self.confidence_threshold, verbose=False)
                primary_detections = 0
                for result in results:
                    for box in result.boxes:
                        detection = {
                            'confidence': float(box.conf),
                            'bbox': box.xyxy.tolist()[0],
                            'source': 'primary_original'
                        }
                        detections.append(detection)
                        primary_detections += 1
                        analysis_logger.info(f"PRIMARY: timestamp={timestamp_seconds:.2f}s, conf={detection['confidence']:.4f}, bbox={detection['bbox']}")
                analysis_logger.info(f"Primary model found {primary_detections} detections")
            except Exception as e:
                analysis_logger.error(f"Primary model failed: {e}")
        
        # 2. Backup YOLOv8m detector
        if self.detector_backup is not None:
            try:
                analysis_logger.info(f"ENSEMBLE_STEP_2: Running YOLOv8m backup model with conf={self.confidence_threshold}")
                results = self.detector_backup(frame, conf=self.confidence_threshold, verbose=False)
                backup_detections = 0
                for result in results:
                    for box in result.boxes:
                        detection = {
                            'confidence': float(box.conf),
                            'bbox': box.xyxy.tolist()[0],
                            'source': 'backup_original'
                        }
                        detections.append(detection)
                        backup_detections += 1
                        analysis_logger.info(f"BACKUP: timestamp={timestamp_seconds:.2f}s, conf={detection['confidence']:.4f}, bbox={detection['bbox']}")
                analysis_logger.info(f"Backup model found {backup_detections} detections")
            except Exception as e:
                analysis_logger.error(f"Backup model failed: {e}")
        
        # 3. MegaDetector fallback (YOLOv8n)
        if self.megadetector is not None:
            try:
                analysis_logger.info(f"ENSEMBLE_STEP_3: Running MegaDetector fallback (YOLOv8n) with conf=0.1")
                results = self.megadetector(frame, conf=0.1, verbose=False)
                mega_detections = 0
                for result in results:
                    for box in result.boxes:
                        detection = {
                            'confidence': float(box.conf),
                            'bbox': box.xyxy.tolist()[0],
                            'source': 'megadetector_fallback',
                            'class': 'detection'
                        }
                        detections.append(detection)
                        mega_detections += 1
                        analysis_logger.info(f"MEGADETECTOR_FALLBACK: timestamp={timestamp_seconds:.2f}s, conf={detection['confidence']:.4f}, bbox={detection['bbox']}")
                analysis_logger.info(f"MegaDetector fallback found {mega_detections} total detections")
            except Exception as e:
                analysis_logger.error(f"MegaDetector fallback failed: {e}")
        
        # 4. MegaDetector v6 (PyTorch-Wildlife)
        if self.megadetector_v6 is not None:
            try:
                analysis_logger.info("ENSEMBLE_STEP_4: Running MegaDetector v6 (PyTorch-Wildlife)")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                analysis_logger.info(f"MegaDetector v6 input: numpy array {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
                
                results = self.megadetector_v6.single_image_detection(
                    rgb_frame,
                    det_conf_thres=0.05
                )
                
                md_v6_detections = 0
                analysis_logger.info(f"MegaDetector v6 result type: {type(results)}")
                
                if results is not None and isinstance(results, dict):
                    detections_obj = results.get('detections', None)
                    
                    if detections_obj is not None and hasattr(detections_obj, 'xyxy'):
                        analysis_logger.info("Processing MegaDetector v6 supervision.Detections format")
                        try:
                            boxes = detections_obj.xyxy
                            confidences = detections_obj.confidence
                            class_ids = getattr(detections_obj, 'class_id', None)
                            
                            # Convert to numpy if needed
                            if hasattr(boxes, 'cpu'):
                                boxes = boxes.cpu().numpy()
                            if hasattr(confidences, 'cpu'):
                                confidences = confidences.cpu().numpy()
                            if class_ids is not None and hasattr(class_ids, 'cpu'):
                                class_ids = class_ids.cpu().numpy()
                            
                            for i in range(len(boxes)):
                                box = boxes[i]
                                conf = confidences[i]
                                
                                bbox = [float(box[j]) for j in range(4)]
                                confidence = float(conf)
                                
                                if confidence >= 0.05:
                                    det = {
                                        'confidence': confidence,
                                        'bbox': bbox,
                                        'source': 'megadetector_v6',
                                        'class': int(class_ids[i]) if class_ids is not None else 1
                                    }
                                    detections.append(det)
                                    md_v6_detections += 1
                                    analysis_logger.info(f"MEGADETECTOR_V6: timestamp={timestamp_seconds:.2f}s, conf={det['confidence']:.4f}, bbox={det['bbox']}")
                        except Exception as e:
                            analysis_logger.error(f"Error parsing MegaDetector v6 supervision format: {e}")
                    else:
                        analysis_logger.info(f"MegaDetector v6 no detections found in result dict")
                
                analysis_logger.info(f"MegaDetector v6 found {md_v6_detections} detections")
            except Exception as e:
                analysis_logger.error(f"MegaDetector v6 failed: {e}")
        
        # 5. DeepFaune detector (PyTorch-Wildlife)
        if self.deepfaune_detector is not None:
            try:
                analysis_logger.info("ENSEMBLE_STEP_5: Running DeepFaune detector (PyTorch-Wildlife)")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                analysis_logger.info(f"DeepFaune input: numpy array {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
                
                results = self.deepfaune_detector.single_image_detection(
                    rgb_frame,
                    det_conf_thres=0.05
                )
                
                deepfaune_detections = 0
                analysis_logger.info(f"DeepFaune result type: {type(results)}")
                
                if results is not None and isinstance(results, dict):
                    detections_list = results.get('detections', [])
                    analysis_logger.info(f"DeepFaune detections list type: {type(detections_list)}, length: {len(detections_list)}")
                    
                    if hasattr(detections_list, 'xyxy') and hasattr(detections_list, 'confidence'):
                        analysis_logger.info("Handling supervision.detection.core.Detections format")
                        try:
                            boxes = detections_list.xyxy
                            confidences = detections_list.confidence
                            class_ids = getattr(detections_list, 'class_id', None)
                            
                            # Convert to numpy if needed
                            if hasattr(boxes, 'cpu'):
                                boxes = boxes.cpu().numpy()
                            if hasattr(confidences, 'cpu'):
                                confidences = confidences.cpu().numpy()
                            if class_ids is not None and hasattr(class_ids, 'cpu'):
                                class_ids = class_ids.cpu().numpy()
                            
                            if len(boxes) == 0:
                                analysis_logger.info("DeepFaune supervision.Detections contains 0 detections")
                            else:
                                for i in range(len(boxes)):
                                    box = boxes[i]
                                    conf = confidences[i]
                                    
                                    bbox = [float(box[j]) for j in range(4)]
                                    confidence = float(conf)
                                    
                                    if confidence >= 0.05:
                                        det = {
                                            'confidence': confidence,
                                            'bbox': bbox,
                                            'source': 'deepfaune_detector',
                                            'class': int(class_ids[i]) if class_ids is not None else 0
                                        }
                                        detections.append(det)
                                        deepfaune_detections += 1
                                        analysis_logger.info(f"DEEPFAUNE_DETECTOR: timestamp={timestamp_seconds:.2f}s, conf={det['confidence']:.4f}, bbox={det['bbox']}")
                        except Exception as e:
                            analysis_logger.error(f"Error parsing supervision Detections: {e}")
                
                analysis_logger.info(f"DeepFaune detector found {deepfaune_detections} detections")
            except Exception as e:
                analysis_logger.error(f"DeepFaune detector failed: {e}")
        
        return detections
    
    def run_enhanced_preprocessing(
        self, 
        frame: np.ndarray, 
        timestamp_seconds: float = 0.0
    ) -> List[Dict]:
        """
        Run enhanced preprocessing techniques to improve detection.
        
        Args:
            frame: OpenCV frame (BGR format)
            timestamp_seconds: Video timestamp in seconds
            
        Returns:
            List of detection dictionaries from enhanced preprocessing
        """
        detections = []
        
        if self.detector is None:
            return detections
        
        try:
            analysis_logger.info("ENHANCEMENT: Running enhanced preprocessing (histogram equalization)")
            
            # Histogram equalization for better contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Run primary detector on enhanced frame
            results = self.detector(enhanced_frame, conf=self.confidence_threshold, verbose=False)
            enhanced_detections = 0
            for result in results:
                for box in result.boxes:
                    detection = {
                        'confidence': float(box.conf),
                        'bbox': box.xyxy.tolist()[0],
                        'source': 'primary_enhanced'
                    }
                    detections.append(detection)
                    enhanced_detections += 1
                    analysis_logger.info(f"ENHANCED: timestamp={timestamp_seconds:.2f}s, conf={detection['confidence']:.4f}, bbox={detection['bbox']}")
            
            analysis_logger.info(f"Enhanced preprocessing found {enhanced_detections} detections")
            
        except Exception as e:
            analysis_logger.error(f"Enhanced preprocessing failed: {e}")
        
        return detections
    
    def run_multiscale_analysis(
        self, 
        frame: np.ndarray, 
        timestamp_seconds: float = 0.0,
        scales: List[float] = [0.8, 1.2]
    ) -> List[Dict]:
        """
        Run multi-scale analysis for better detection coverage.
        
        Args:
            frame: OpenCV frame (BGR format)
            timestamp_seconds: Video timestamp in seconds
            scales: List of scale factors to try
            
        Returns:
            List of detection dictionaries from multi-scale analysis
        """
        detections = []
        
        if self.detector is None:
            return detections
        
        for scale in scales:
            try:
                analysis_logger.info(f"MULTISCALE: Running multi-scale analysis at {scale}x")
                
                # Scale frame
                height, width = frame.shape[:2]
                new_height, new_width = int(height * scale), int(width * scale)
                scaled_frame = cv2.resize(frame, (new_width, new_height))
                analysis_logger.info(f"Scaled frame from {height}x{width} to {new_height}x{new_width}")
                
                # Run detection on scaled frame
                results = self.detector(scaled_frame, conf=self.confidence_threshold, verbose=False)
                scale_detections = 0
                for result in results:
                    for box in result.boxes:
                        # Scale coordinates back to original frame size
                        bbox = box.xyxy.tolist()[0]
                        original_bbox = [
                            bbox[0] / scale,  # x1
                            bbox[1] / scale,  # y1
                            bbox[2] / scale,  # x2
                            bbox[3] / scale   # y2
                        ]
                        
                        detection = {
                            'confidence': float(box.conf),
                            'bbox': original_bbox,
                            'source': f'primary_scale_{scale}'
                        }
                        detections.append(detection)
                        scale_detections += 1
                        analysis_logger.info(f"SCALE_{scale}: timestamp={timestamp_seconds:.2f}s, conf={detection['confidence']:.4f}, original_bbox={bbox}, scaled_bbox={original_bbox}")
                
                analysis_logger.info(f"Scale {scale}x found {scale_detections} detections")
                
            except Exception as e:
                analysis_logger.error(f"Multi-scale analysis at {scale}x failed: {e}")
        
        return detections
    
    def extract_features(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Extract ResNet18 features from a bounding box region.
        
        Args:
            frame: OpenCV frame (BGR format)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector as numpy array, or None if extraction failed
        """
        if self.feature_extractor is None:
            return None
        
        try:
            # Crop region
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            
            # Prepare for ResNet18
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224))
            
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(crop_resized).unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            return None