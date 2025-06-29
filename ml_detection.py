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
logger = logging.getLogger('wildcams')
analysis_logger = logger

class MLDetectionEnsemble:
    """Enhanced ensemble ML detection system with accuracy improvements."""
    
    def __init__(self, confidence_threshold: float = 0.1, megadetector_version: str = 'MDV6-rtdetr-c', ensemble_models: List[str] = None, cache_dir: Optional['Path'] = None):
        self.confidence_threshold = confidence_threshold
        self.megadetector_version = megadetector_version
        self.ensemble_models = ensemble_models or ['yolov8x', 'yolov8m', 'yolov8n', 'megadetector_v6']
        self.cache_dir = cache_dir
        
        # Accuracy Enhancement 2: Model-specific confidence thresholds (optimized for recall)
        self.model_thresholds = {
            'yolov8x': 0.05,        # Lower for primary model
            'yolov8m': 0.08,        # Slightly higher for backup
            'yolov8n': 0.12,        # Higher for fallback
            'megadetector_v6': 0.1, # Wildlife-specific threshold
            'deepfaune': 0.15       # Conservative for classification model
        }
        
        # Accuracy Enhancement 3: Multi-scale detection settings
        self.detection_scales = [0.8, 1.0, 1.2, 1.5]
        
        # Accuracy Enhancement 1: Test-Time Augmentation settings
        self.enable_tta = True
        self.tta_transforms = [
            'original',
            'horizontal_flip', 
            'brightness_adjust',
            'contrast_adjust',
            'gaussian_blur'
        ]
        
        # Model initialization flags
        self.detector = None
        self.detector_backup = None
        self.megadetector = None
        self.megadetector_v6 = None
        self.deepfaune_detector = None
        self.feature_extractor = None
        
        # Initialize models based on configuration
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models based on ensemble configuration."""
        try:
            logger.info(f"🤖 Initializing ML detection ensemble with models: {self.ensemble_models}")
            logger.info(f"🎯 MegaDetector version: {self.megadetector_version}")
            
            # Set up model caching for PyTorch-Wildlife
            if self.cache_dir:
                import os
                os.environ['TORCH_HOME'] = str(self.cache_dir / 'torch')
                os.environ['PYTORCH_WILDLIFE_CACHE'] = str(self.cache_dir / 'pytorch_wildlife')
                logger.info(f"📦 Model cache directory: {self.cache_dir}")
                
                # Create cache subdirectories
                (self.cache_dir / 'torch').mkdir(exist_ok=True)
                (self.cache_dir / 'pytorch_wildlife').mkdir(exist_ok=True)
            
            # Primary YOLOv8x detector
            if 'yolov8x' in self.ensemble_models:
                try:
                    logger.info("Loading YOLOv8x primary detector...")
                    self.detector = YOLO('yolov8x.pt')
                    logger.info("✅ YOLOv8x primary detector loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load YOLOv8x primary detector: {e}")
            else:
                logger.info("⏭️ Skipping YOLOv8x (not in ensemble configuration)")
            
            # Backup YOLOv8m detector
            if 'yolov8m' in self.ensemble_models:
                try:
                    logger.info("Loading YOLOv8m backup detector...")
                    self.detector_backup = YOLO('yolov8m.pt')
                    logger.info("✅ YOLOv8m backup detector loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load YOLOv8m backup detector: {e}")
            else:
                logger.info("⏭️ Skipping YOLOv8m (not in ensemble configuration)")
            
            # MegaDetector fallback (YOLOv8n)
            if 'yolov8n' in self.ensemble_models:
                try:
                    logger.info("Loading YOLOv8n as MegaDetector fallback...")
                    self.megadetector = YOLO('yolov8n.pt')
                    logger.info("✅ YOLOv8n MegaDetector fallback loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load YOLOv8n MegaDetector fallback: {e}")
            else:
                logger.info("⏭️ Skipping YOLOv8n (not in ensemble configuration)")
            
            # PyTorch-Wildlife MegaDetector v6
            if 'megadetector_v6' in self.ensemble_models and PYTORCH_WILDLIFE_AVAILABLE:
                try:
                    logger.info(f"🦎 Loading MegaDetector v6 ({self.megadetector_version})...")
                    
                    # Check if model is already cached
                    if self.cache_dir:
                        model_cache_path = self.cache_dir / 'pytorch_wildlife' / f'{self.megadetector_version}.pt'
                        if model_cache_path.exists():
                            logger.info(f"📦 Using cached model: {model_cache_path}")
                    
                    # Load model with caching enabled and verbose disabled
                    self.megadetector_v6 = pw_detection.MegaDetectorV6(
                        version=self.megadetector_version, 
                        pretrained=True,
                        device='auto'  # Let PyTorch-Wildlife choose best device
                    )
                    
                    # Disable verbose to prevent KeyError with unknown class IDs
                    if hasattr(self.megadetector_v6, 'predictor') and hasattr(self.megadetector_v6.predictor, 'args'):
                        self.megadetector_v6.predictor.args.verbose = False
                    logger.info(f"✅ MegaDetector v6 ({self.megadetector_version}) loaded successfully")
                    
                    # Log cache status
                    if self.cache_dir:
                        logger.info(f"📦 Model cached to: {self.cache_dir / 'pytorch_wildlife'}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load MegaDetector v6 ({self.megadetector_version}): {e}")
                    self.megadetector_v6 = None
            elif 'megadetector_v6' in self.ensemble_models:
                logger.warning("⚠️ PyTorch-Wildlife not available - MegaDetector v6 disabled")
            else:
                logger.info("⏭️ Skipping MegaDetector v6 (not in ensemble configuration)")
            
            # DeepFaune is a classification model, not detection - always disable for detection ensemble
            logger.info("🦎 DeepFaune is a classification model, not detection - skipping in detection ensemble")
            self.deepfaune_detector = None
            
            # Feature extractor for clustering (always load)
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
    
    def apply_tta_transforms(self, frame: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Accuracy Enhancement 1: Apply Test-Time Augmentation transforms.
        
        Returns:
            List of (transformed_frame, transform_name) tuples
        """
        transforms = []
        
        # Original frame
        transforms.append((frame.copy(), 'original'))
        
        if not self.enable_tta:
            return transforms
        
        # Horizontal flip
        if 'horizontal_flip' in self.tta_transforms:
            flipped = cv2.flip(frame, 1)
            transforms.append((flipped, 'horizontal_flip'))
        
        # Brightness adjustment (+20%)
        if 'brightness_adjust' in self.tta_transforms:
            bright = cv2.convertScaleAbs(frame, alpha=1.0, beta=20)
            transforms.append((bright, 'brightness_adjust'))
        
        # Contrast adjustment (1.2x)
        if 'contrast_adjust' in self.tta_transforms:
            contrast = cv2.convertScaleAbs(frame, alpha=1.2, beta=0)
            transforms.append((contrast, 'contrast_adjust'))
        
        # Gaussian blur (slight)
        if 'gaussian_blur' in self.tta_transforms:
            blurred = cv2.GaussianBlur(frame, (3, 3), 0.5)
            transforms.append((blurred, 'gaussian_blur'))
        
        return transforms
    
    def apply_multiscale_detection(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float, str]]:
        """
        Accuracy Enhancement 3: Apply multi-scale detection.
        
        Returns:
            List of (scaled_frame, scale_factor, scale_name) tuples
        """
        scaled_frames = []
        
        for scale in self.detection_scales:
            scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            scale_name = f"scale_{scale:.1f}x"
            scaled_frames.append((scaled_frame, scale, scale_name))
        
        return scaled_frames
    
    def apply_advanced_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Accuracy Enhancement 5: Apply advanced Non-Maximum Suppression across ensemble.
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered detections after NMS
        """
        if len(detections) <= 1:
            return detections
        
        # Convert to format needed for NMS
        import torchvision.ops as ops
        
        boxes = []
        scores = []
        sources = []
        
        for det in detections:
            boxes.append(det['bbox'])
            scores.append(det['confidence'])
            sources.append(det['source'])
        
        if not boxes:
            return []
        
        # Convert to tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # Apply NMS
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        
        # Return filtered detections
        filtered_detections = []
        for idx in keep_indices:
            filtered_detections.append(detections[idx.item()])
        
        analysis_logger.info(f"🧹 ADVANCED NMS: {len(detections)} → {len(filtered_detections)} detections (removed {len(detections) - len(filtered_detections)} duplicates)")
        
        return filtered_detections
    
    def run_single_model_detection(self, model, model_name: str, frame: np.ndarray, 
                                 timestamp_seconds: float = 0.0) -> List[Dict]:
        """
        Run detection on a single model with model-specific threshold.
        
        Returns:
            List of detections from this model
        """
        if model is None:
            return []
        
        # Use model-specific threshold
        threshold = self.model_thresholds.get(model_name, self.confidence_threshold)
        
        try:
            results = model(frame, conf=threshold, verbose=False)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    detection = {
                        'confidence': float(box.conf),
                        'bbox': box.xyxy.tolist()[0],
                        'source': f'{model_name}',
                        'model_threshold': threshold
                    }
                    detections.append(detection)
            
            analysis_logger.info(f"🔍 {model_name.upper()}: {len(detections)} detections (threshold: {threshold})")
            return detections
            
        except Exception as e:
            analysis_logger.error(f"❌ {model_name} detection failed: {e}")
            return []
    
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
        
        step_counter = 1
        
        # 1. Primary YOLOv8x detector
        if self.detector is not None:
            try:
                logger.info(f"🔍 Running YOLOv8x detection...")
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
                        logger.debug(f"YOLOv8x: conf={detection['confidence']:.4f}, bbox={detection['bbox']}")
                logger.info(f"✅ YOLOv8x: {primary_detections} detections found")
            except Exception as e:
                analysis_logger.error(f"Primary model failed: {e}")
                logger.error(f"❌ YOLOv8x failed: {e}")
        
        # 2. Backup YOLOv8m detector
        if self.detector_backup is not None:
            try:
                analysis_logger.info(f"ENSEMBLE_STEP_{step_counter}: Running YOLOv8m backup model with conf={self.confidence_threshold}")
                logger.info(f"🔍 Running YOLOv8m detection...")
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
                logger.info(f"✅ YOLOv8m: {backup_detections} detections found")
                analysis_logger.info(f"Backup model found {backup_detections} detections")
                step_counter += 1
            except Exception as e:
                analysis_logger.error(f"Backup model failed: {e}")
                logger.error(f"❌ YOLOv8m failed: {e}")
        
        # 3. MegaDetector fallback (YOLOv8n)
        if self.megadetector is not None:
            try:
                analysis_logger.info(f"ENSEMBLE_STEP_{step_counter}: Running MegaDetector fallback (YOLOv8n) with conf=0.1")
                logger.info(f"🔍 Running YOLOv8n (MegaDetector fallback)...")
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
                logger.info(f"✅ YOLOv8n: {mega_detections} detections found")
                analysis_logger.info(f"MegaDetector fallback found {mega_detections} total detections")
                step_counter += 1
            except Exception as e:
                analysis_logger.error(f"MegaDetector fallback failed: {e}")
                logger.error(f"❌ YOLOv8n failed: {e}")
        
        # 4. MegaDetector v6 (PyTorch-Wildlife)
        if self.megadetector_v6 is not None:
            try:
                analysis_logger.info(f"ENSEMBLE_STEP_{step_counter}: Running MegaDetector v6 (PyTorch-Wildlife)")
                logger.info(f"🔍 Running MegaDetector v6 ({self.megadetector_version})...")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                analysis_logger.info(f"MegaDetector v6 input: numpy array {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
                
                try:
                    results = self.megadetector_v6.single_image_detection(
                        rgb_frame,
                        det_conf_thres=0.05
                    )
                except KeyError as ke:
                    # Handle PyTorch-Wildlife bug with unknown class IDs
                    # The model detected classes outside the standard MegaDetector mapping (0:animal, 1:person, 2:vehicle)
                    logger.info(f"🔍 MegaDetector v6 detected unknown class ID {ke} - likely animal detection from extended class set")
                    logger.debug(f"Standard MegaDetector classes: 0=animal, 1=person, 2=vehicle. Detected class {ke} suggests extended model.")
                    
                    # Try to get raw detection results directly from the predictor
                    try:
                        if hasattr(self.megadetector_v6, 'predictor') and hasattr(self.megadetector_v6.predictor, 'results'):
                            raw_results = self.megadetector_v6.predictor.results[-1] if self.megadetector_v6.predictor.results else None
                            if raw_results and hasattr(raw_results, 'boxes'):
                                # Create custom results with raw class IDs
                                results = {'detections': raw_results, 'raw_class_ids': True}
                            else:
                                results = {'detections': None}
                        else:
                            results = {'detections': None}
                    except Exception as e:
                        logger.debug(f"Could not extract raw results: {e}")
                        results = {'detections': None}
                
                md_v6_detections = 0
                analysis_logger.info(f"MegaDetector v6 result type: {type(results)}")
                
                if results is not None and isinstance(results, dict):
                    detections_obj = results.get('detections', None)
                    is_raw_class_ids = results.get('raw_class_ids', False)
                    
                    if detections_obj is not None:
                        if is_raw_class_ids and hasattr(detections_obj, 'boxes'):
                            # Handle raw Ultralytics results with potentially unknown class IDs
                            logger.info("Processing MegaDetector v6 raw results with extended class IDs")
                            try:
                                boxes = detections_obj.boxes
                                if hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
                                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                                    cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                                    
                                    for i in range(len(xyxy)):
                                        # Fix coordinate handling - xyxy should be [x1, y1, x2, y2]
                                        x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
                                        bbox = [x1, y1, x2, y2]
                                        confidence = float(confs[i])
                                        class_id = int(cls_ids[i])
                                        
                                        # Skip detections with invalid coordinates
                                        if confidence >= 0.05 and x2 > x1 and y2 > y1:
                                            det = {
                                                'confidence': confidence,
                                                'bbox': bbox,
                                                'source': 'megadetector_v6',
                                                'class': class_id,
                                                'raw_class_id': True
                                            }
                                            detections.append(det)
                                            md_v6_detections += 1
                                            logger.info(f"✅ MegaDetector v6 raw: class_id={class_id}, conf={confidence:.4f}")
                            except Exception as e:
                                logger.error(f"Error parsing MegaDetector v6 raw results: {e}")
                        elif hasattr(detections_obj, 'xyxy'):
                            # Standard supervision.Detections format
                            logger.debug("Processing MegaDetector v6 supervision.Detections format")
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
                                        logger.debug(f"MegaDetector v6: conf={det['confidence']:.4f}, bbox={det['bbox']}")
                            except Exception as e:
                                logger.error(f"Error parsing MegaDetector v6 supervision format: {e}")
                        else:
                            logger.debug(f"MegaDetector v6 no detections found in result dict")
                
                logger.info(f"✅ MegaDetector v6: {md_v6_detections} detections found")
                analysis_logger.info(f"MegaDetector v6 found {md_v6_detections} detections")
            except Exception as e:
                import traceback
                logger.error(f"❌ MegaDetector v6 failed: {e}")
                logger.debug(f"MegaDetector v6 traceback: {traceback.format_exc()}")
        
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