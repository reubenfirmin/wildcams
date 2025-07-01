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

# Set torch cache directory BEFORE importing torch
cache_dir = os.path.abspath('./models_cache/torch')
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir

import torch
# Set hub directory immediately after torch import
torch.hub.set_dir(cache_dir)

from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights


# PyTorch-Wildlife imports
try:
    from PytorchWildlife.models import detection as pw_detection
    PYTORCH_WILDLIFE_AVAILABLE = True
except ImportError:
    PYTORCH_WILDLIFE_AVAILABLE = False

def calculate_bbox_overlap(bbox1, bbox2):
    """Calculate overlap percentage of bbox2 within bbox1.
    
    Args:
        bbox1: [x1, y1, x2, y2] - reference bbox (crop region)
        bbox2: [x1, y1, x2, y2] - detection bbox (RT-DETR detection)
    
    Returns:
        float: Overlap percentage (0.0 to 1.0) of bbox2 area that overlaps with bbox1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0  # No overlap
    
    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    if bbox2_area <= 0:
        return 0.0
    
    return intersection_area / bbox2_area

# Get loggers
logger = logging.getLogger('wildcams')
analysis_logger = logger

class MLDetectionEnsemble:
    """Enhanced ensemble ML detection system with accuracy improvements."""
    
    def __init__(self, confidence_threshold: float = 0.1, ensemble_models: List[str] = None, cache_dir: Optional['Path'] = None):
        self.confidence_threshold = confidence_threshold
        self.ensemble_models = ensemble_models  # No default here - comes from base processor
        self.cache_dir = cache_dir
        
        # Unified model registry - all YOLO-compatible models
        self.yolo_detectors = {}  # Will store all YOLO models (v8, v10, v12, etc.)
        
        # Model storage for multiple MegaDetector variants
        self.megadetector_variants = {}  # Will store multiple MD models
        
        # Accuracy Enhancement 2: Model-specific confidence thresholds (optimized for recall)
        self.model_thresholds = {
            'yolov8x': 0.05,           # Primary YOLO model
            'yolov8m': 0.08,           # Medium YOLO model
            'MDV6-rtdetr-c': 0.1,      # RT-DETR variant (highest accuracy)
            'MDV6-yolov9-e': 0.1,      # YOLOv9 variant (balanced)
            'MDV6-yolov9-c': 0.1,      # YOLOv9 compact
            'MDV6-yolov10-e': 0.1,     # YOLOv10 variants
            'MDV6-yolov10-c': 0.1,
            'deepfaune': 0.15          # Conservative for classification model
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
        self.deepfaune_detector = None
        self.feature_extractor = None
        
        # Initialize models based on configuration
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models based on ensemble configuration."""
        try:
            logger.info(f"🤖 Initializing ML detection ensemble with models: {self.ensemble_models}")
            
            # Cache directory already set at module import time
            if self.cache_dir:
                logger.info(f"📦 Model cache directory: {self.cache_dir}")
                logger.info(f"🔧 TORCH_HOME: {os.environ.get('TORCH_HOME', 'not set')}")
                logger.info(f"🔧 torch.hub.get_dir(): {torch.hub.get_dir()}")
            
            # Load all YOLO models in unified registry
            all_yolo_variants = [
                'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                'yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x',
                'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'
            ]
            
            for variant in all_yolo_variants:
                if variant in self.ensemble_models:
                    try:
                        logger.info(f"Loading {variant.upper()} detector...")
                        self.yolo_detectors[variant] = YOLO(f'{variant}.pt')
                        logger.info(f"✅ {variant.upper()} detector loaded successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to load {variant.upper()} detector: {e}")
                        self.yolo_detectors[variant] = None
                else:
                    logger.info(f"⏭️ Skipping {variant.upper()} (not in ensemble configuration)")
            
            # Load multiple MegaDetector variants
            megadetector_variants_in_ensemble = [model for model in self.ensemble_models if model.startswith('MDV6-')]
            
            if megadetector_variants_in_ensemble and PYTORCH_WILDLIFE_AVAILABLE:
                for variant in megadetector_variants_in_ensemble:
                    try:
                        logger.info(f"🦎 Loading MegaDetector v6 variant: {variant}")
                        
                        # Load model - PyTorch-Wildlife should use environment cache settings
                        md_model = pw_detection.MegaDetectorV6(
                            version=variant, 
                            pretrained=True,
                            device='auto'  # Let PyTorch-Wildlife choose best device
                        )
                        
                        # Disable verbose to prevent KeyError with unknown class IDs
                        if hasattr(md_model, 'predictor') and hasattr(md_model.predictor, 'args'):
                            md_model.predictor.args.verbose = False
                        
                        self.megadetector_variants[variant] = md_model
                        logger.info(f"✅ MegaDetector v6 ({variant}) loaded successfully")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load MegaDetector v6 variant {variant}: {e}")
                        self.megadetector_variants[variant] = None
                
                # Log cache status
                if self.cache_dir and self.megadetector_variants:
                    logger.info(f"📦 Models cached to: {self.cache_dir / 'pytorch_wildlife'}")
                        
            elif megadetector_variants_in_ensemble:
                logger.warning("⚠️ PyTorch-Wildlife not available - MegaDetector variants disabled")
            else:
                logger.info("⏭️ Skipping MegaDetector variants (none in ensemble configuration)")
            
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
        frame_idx: int = 0,
        full_frame: np.ndarray = None,
        crop_region: Tuple[int, int, int, int] = None,
        crop_regions: List[Tuple[int, int, int, int]] = None,
        accepted_rtdetr_overlap: float = 0.5
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
        
        # Run all YOLO models from unified registry
        yolo_models_in_ensemble = [model for model in self.ensemble_models if not model.startswith('MDV6-')]
        
        for model_name in yolo_models_in_ensemble:
            detector = self.yolo_detectors.get(model_name)
            if detector is not None:
                try:
                    analysis_logger.info(f"ENSEMBLE_STEP_{step_counter}: Running {model_name.upper()} model with conf={self.confidence_threshold}")
                    logger.info(f"🔍 Running {model_name.upper()} detection...")
                    results = detector(frame, conf=self.confidence_threshold, verbose=False)
                    model_detections = 0
                    for result in results:
                        for box in result.boxes:
                            confidence = float(box.conf)
                            bbox = box.xyxy.tolist()[0]
                            
                            # Log high-confidence detections with spatial analysis
                            if confidence >= 0.3 and crop_regions is not None and len(crop_regions) > 0:
                                max_overlap = 0.0
                                for crop_bbox in crop_regions:
                                    overlap = calculate_bbox_overlap(crop_bbox, bbox)
                                    max_overlap = max(max_overlap, overlap)
                                
                                logger.info(f"🔍 {model_name.upper()} high-conf: conf={confidence:.3f}, bbox={bbox}, max_overlap={max_overlap:.3f} at {timestamp_seconds:.1f}s")
                                if len(crop_regions) <= 3:  # Show details for first few crop regions
                                    for crop_bbox in crop_regions:
                                        overlap = calculate_bbox_overlap(crop_bbox, bbox)
                                        logger.info(f"  📏 Crop {crop_bbox} vs {model_name.upper()} {bbox} = overlap {overlap:.3f}")
                            
                            detection = {
                                'confidence': confidence,
                                'bbox': bbox,
                                'source': model_name
                            }
                            detections.append(detection)
                            model_detections += 1
                            logger.debug(f"{model_name.upper()}: conf={confidence:.4f}, bbox={bbox}")
                    
                    if model_detections > 0:
                        analysis_logger.info(f"{model_name.upper()}: {model_detections} detections")
                    logger.info(f"✅ {model_name.upper()}: {model_detections} detections found")
                    step_counter += 1
                except Exception as e:
                    analysis_logger.error(f"{model_name.upper()} model failed: {e}")
                    logger.error(f"❌ {model_name.upper()} failed: {e}")
            else:
                logger.warning(f"⚠️ {model_name.upper()} detector not loaded")
        
        # 3. MegaDetector v6 variants (PyTorch-Wildlife)
        for variant_name, md_model in self.megadetector_variants.items():
            if md_model is not None:
                try:
                    analysis_logger.info(f"ENSEMBLE_STEP_{step_counter}: Running MegaDetector v6 variant {variant_name}")
                    logger.info(f"🔍 Running MegaDetector v6 ({variant_name})...")
                    
                    # RT-DETR needs full frame context, skip if only crop available
                    if 'rtdetr' in variant_name.lower():
                        if full_frame is not None:
                            input_frame = full_frame
                            logger.info(f"🖼️ Using full frame for RT-DETR: {input_frame.shape}")
                        else:
                            logger.info(f"⏭️ Skipping RT-DETR {variant_name} - no full frame available (crop-only context)")
                            continue
                    else:
                        input_frame = frame
                        logger.info(f"🔲 Using crop for {variant_name}: {input_frame.shape}")
                    
                    rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                    analysis_logger.info(f"MegaDetector v6 input: numpy array {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
                    
                    # Debug RT-DETR input frame
                    if 'rtdetr' in variant_name.lower():
                        logger.info(f"🔍 RT-DETR input frame shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
                        logger.info(f"🔍 RT-DETR input frame range: min={rgb_frame.min()}, max={rgb_frame.max()}")
                        logger.info(f"🔍 RT-DETR model threshold: {self.model_thresholds.get(variant_name, 0.1)}")
                    
                    try:
                        results = md_model.single_image_detection(
                            rgb_frame,
                            det_conf_thres=self.model_thresholds.get(variant_name, 0.1)
                        )
                    except KeyError as ke:
                        # Handle PyTorch-Wildlife bug with unknown class IDs
                        # The model detected classes outside the standard MegaDetector mapping (0:animal, 1:person, 2=vehicle)
                        logger.info(f"🔍 MegaDetector v6 ({variant_name}) detected unknown class ID {ke} - class not in standard mapping")
                        logger.debug(f"Standard MegaDetector classes: 0=animal, 1=person, 2=vehicle. Class {ke} is from extended model.")
                        
                        # Try to get raw detection results directly from the predictor
                        try:
                            if hasattr(md_model, 'predictor') and hasattr(md_model.predictor, 'results'):
                                raw_results = md_model.predictor.results[-1] if md_model.predictor.results else None
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
                    except Exception as e:
                        analysis_logger.error(f"{variant_name} failed: {e}")
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
                                    
                                    # Debug: examine the raw box structure for RT-DETR
                                    if 'rtdetr' in variant_name.lower():
                                        logger.info(f"🔍 RT-DETR boxes object: {type(boxes)}")
                                        if hasattr(boxes, 'data'):
                                            raw_data = boxes.data.cpu().numpy() if hasattr(boxes.data, 'cpu') else boxes.data
                                            logger.info(f"🔍 RT-DETR boxes.data shape: {raw_data.shape if hasattr(raw_data, 'shape') else 'no shape'}")
                                            if hasattr(raw_data, 'shape') and len(raw_data.shape) >= 2 and raw_data.shape[0] > 0:
                                                logger.info(f"🔍 RT-DETR raw data tensor (first detection): {raw_data[0]}")
                                                # Check if this is actually CXCYWH format that needs conversion
                                                if raw_data.shape[1] >= 4:
                                                    cx, cy, w, h = raw_data[0][:4]
                                                    x1_calc = cx - w/2
                                                    y1_calc = cy - h/2  
                                                    x2_calc = cx + w/2
                                                    y2_calc = cy + h/2
                                                    logger.info(f"🔍 RT-DETR CXCYWH interpretation: [{cx:.6f},{cy:.6f},{w:.6f},{h:.6f}] → [{x1_calc:.6f},{y1_calc:.6f},{x2_calc:.6f},{y2_calc:.6f}]")
                                        if hasattr(boxes, 'xyxy'):
                                            xyxy_data = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                                            logger.info(f"🔍 RT-DETR boxes.xyxy shape: {xyxy_data.shape if hasattr(xyxy_data, 'shape') else 'no shape'}")
                                            if hasattr(xyxy_data, 'shape') and len(xyxy_data.shape) >= 2 and xyxy_data.shape[0] > 0:
                                                logger.info(f"🔍 RT-DETR boxes.xyxy (first detection): {xyxy_data[0]}")
                                        if hasattr(boxes, 'xywh'):
                                            xywh_data = boxes.xywh.cpu().numpy() if hasattr(boxes.xywh, 'cpu') else boxes.xywh  
                                            logger.info(f"🔍 RT-DETR boxes.xywh shape: {xywh_data.shape if hasattr(xywh_data, 'shape') else 'no shape'}")
                                            if hasattr(xywh_data, 'shape') and len(xywh_data.shape) >= 2 and xywh_data.shape[0] > 0:
                                                logger.info(f"🔍 RT-DETR boxes.xywh (first detection): {xywh_data[0]}")
                                    
                                    # Try multiple coordinate formats for RT-DETR compatibility
                                    coords_extracted = False
                                    
                                    if hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
                                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                                        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                                        coords_extracted = True
                                        
                                        if 'rtdetr' in variant_name.lower():
                                            logger.info(f"🔍 RT-DETR extracted xyxy coords: {xyxy[:3] if len(xyxy) > 0 else 'empty'}")
                                            logger.info(f"🔍 RT-DETR extracted confs: {confs[:3] if len(confs) > 0 else 'empty'}")
                                            logger.info(f"🔍 RT-DETR extracted cls_ids: {cls_ids[:3] if len(cls_ids) > 0 else 'empty'}")
                                            if len(xyxy) > 0:
                                                for i, coord in enumerate(xyxy[:3]):
                                                    logger.info(f"🔍 RT-DETR detection {i}: [{coord[0]:.6f},{coord[1]:.6f},{coord[2]:.6f},{coord[3]:.6f}] conf={confs[i]:.3f} cls={cls_ids[i]}")
                                    
                                    elif hasattr(boxes, 'xywh') and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
                                        # RT-DETR might output in xywh format - convert to xyxy
                                        xywh = boxes.xywh.cpu().numpy() if hasattr(boxes.xywh, 'cpu') else boxes.xywh
                                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                                        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                                        
                                        # Convert xywh to xyxy format
                                        xyxy = []
                                        for i in range(len(xywh)):
                                            cx, cy, w, h = xywh[i]
                                            x1 = cx - w/2
                                            y1 = cy - h/2
                                            x2 = cx + w/2
                                            y2 = cy + h/2
                                            xyxy.append([x1, y1, x2, y2])
                                        xyxy = np.array(xyxy)
                                        coords_extracted = True
                                        
                                        logger.info(f"🔄 RT-DETR converted xywh→xyxy: {xyxy[:3] if len(xyxy) > 0 else 'empty'}")
                                    
                                    elif 'rtdetr' in variant_name.lower() and hasattr(boxes, 'data'):
                                        # RT-DETR coordinate fix: handle CXCYWH format from raw tensor
                                        # RT-DETR often outputs in [cx, cy, w, h, conf, cls] format
                                        logger.info("🔄 RT-DETR coordinate fix: checking CXCYWH format in raw data tensor")
                                        raw_data = boxes.data.cpu().numpy() if hasattr(boxes.data, 'cpu') else boxes.data
                                        
                                        if len(raw_data.shape) >= 2 and raw_data.shape[1] >= 6:
                                            # Extract raw values
                                            cx_values = raw_data[:, 0]  # Center X
                                            cy_values = raw_data[:, 1]  # Center Y  
                                            w_values = raw_data[:, 2]   # Width
                                            h_values = raw_data[:, 3]   # Height
                                            confs = raw_data[:, 4]      # Confidence
                                            cls_ids = raw_data[:, 5]    # Class ID
                                            
                                            # Check if this looks like CXCYWH format (non-zero cy values)
                                            if len(cy_values) > 0 and any(cy > 0.01 for cy in cy_values):
                                                # Convert CXCYWH to XYXY manually
                                                xyxy = []
                                                for i in range(len(raw_data)):
                                                    cx, cy, w, h = cx_values[i], cy_values[i], w_values[i], h_values[i]
                                                    x1 = cx - w/2
                                                    y1 = cy - h/2
                                                    x2 = cx + w/2
                                                    y2 = cy + h/2
                                                    xyxy.append([x1, y1, x2, y2])
                                                xyxy = np.array(xyxy)
                                                coords_extracted = True
                                                
                                                logger.info(f"🔧 RT-DETR CXCYWH→XYXY conversion successful: {len(xyxy)} detections")
                                                if len(xyxy) > 0:
                                                    logger.info(f"🔍 RT-DETR converted coordinates (first): [{xyxy[0][0]:.6f},{xyxy[0][1]:.6f},{xyxy[0][2]:.6f},{xyxy[0][3]:.6f}] conf={confs[0]:.3f}")
                                            else:
                                                # Fallback to raw XYXY extraction
                                                xyxy = raw_data[:, :4]  # First 4 columns: x1, y1, x2, y2
                                                coords_extracted = True
                                                logger.info(f"🔄 RT-DETR using raw XYXY format: {xyxy[:3] if len(xyxy) > 0 else 'empty'}")
                                    
                                    if coords_extracted:
                                        
                                        # Collect all detections for summary logging
                                        animals, persons, unknowns = [], [], []
                                        
                                        for i in range(len(xyxy)):
                                            # Fix coordinate handling - xyxy should be [x1, y1, x2, y2]
                                            x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
                                            confidence = float(confs[i])
                                            class_id = int(cls_ids[i])
                                            
                                            # Debug: log raw coordinates to understand coordinate system
                                            if confidence >= 0.8 and 'rtdetr' in variant_name.lower():
                                                logger.info(f"🔍 RT-DETR raw coords: [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}], frame_shape={frame.shape}, conf={confidence:.3f}")
                                            
                                            # RT-DETR coordinate repair: fix zero Y-coordinates
                                            if 'rtdetr' in variant_name.lower() and y1 == 0.0 and y2 == 0.0:
                                                # Estimate reasonable Y-coordinates based on frame dimensions and X-coordinates
                                                h, w = frame.shape[:2] if len(frame.shape) > 2 else (frame.shape[0], frame.shape[1])
                                                x_width = abs(x2 - x1)
                                                
                                                # Assume detection is roughly square/rectangular with similar aspect ratio
                                                estimated_height = min(x_width * 1.2, 0.3)  # Max 30% of frame height
                                                y_center = 0.5  # Assume center of frame
                                                y1 = max(0.0, y_center - estimated_height/2)
                                                y2 = min(1.0, y_center + estimated_height/2)
                                                
                                                logger.info(f"🔧 RT-DETR coordinate repair: Y [{0.0:.3f},{0.0:.3f}] → [{y1:.3f},{y2:.3f}] (width={x_width:.3f}, est_height={estimated_height:.3f})")
                                            
                                            # Use MegaDetector with confidence threshold - RT-DETR with full frame has reliable coordinates
                                            coord_valid = x2 > x1 and y2 > y1
                                            if confidence >= 0.05 and (coord_valid or 'rtdetr' in variant_name.lower()):
                                                # RT-DETR coordinates: convert to pixel coordinates if needed
                                                h, w = frame.shape[:2] if len(frame.shape) > 2 else (frame.shape[0], frame.shape[1])
                                                
                                                # RT-DETR coordinate conversion: always convert normalized coordinates to pixels
                                                if 'rtdetr' in variant_name.lower() or (x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0 and x1 >= 0.0 and y1 >= 0.0):
                                                    # Convert normalized coordinates to pixel coordinates (RT-DETR can have x>1.0)
                                                    x1_px, y1_px = x1 * w, y1 * h
                                                    x2_px, y2_px = x2 * w, y2 * h
                                                    bbox = [x1_px, y1_px, x2_px, y2_px]
                                                    if 'rtdetr' in variant_name.lower() and confidence >= 0.8:
                                                        logger.info(f"🔄 RT-DETR normalized→pixel: [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] → [{x1_px:.1f},{y1_px:.1f},{x2_px:.1f},{y2_px:.1f}]")
                                                elif x2 > x1 and y2 > y1:
                                                    # Already pixel coordinates - use as-is
                                                    bbox = [x1, y1, x2, y2]
                                                else:
                                                    # Invalid coordinates - use full frame as fallback
                                                    bbox = [0, 0, w, h]
                                                    logger.warning(f"⚠️ RT-DETR invalid coordinates [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] - using full frame fallback")
                                                
                                                # Categorize detections for summary
                                                accepted_detection = False
                                                
                                                if class_id == 0:  # 0=animal in standard MegaDetector
                                                    det = {
                                                        'confidence': confidence,
                                                        'bbox': bbox,
                                                        'source': f'megadetector_{variant_name}',
                                                        'class': class_id,
                                                        'raw_class_id': True
                                                    }
                                                    detections.append(det)
                                                    md_v6_detections += 1
                                                    animals.append((class_id, confidence))
                                                    accepted_detection = True
                                                elif class_id == 1:
                                                    persons.append((class_id, confidence))
                                                elif (confidence >= 0.8 and 'rtdetr' in variant_name.lower() and 
                                                      crop_regions is not None):
                                                    # Check spatial overlap with crop regions for high-confidence RT-DETR detections
                                                    logger.info(f"🔍 RT-DETR checking {len(crop_regions)} crop regions for class_id={class_id}, conf={confidence:.3f}, bbox={bbox} at timestamp={timestamp_seconds:.1f}s")
                                                    max_overlap = 0.0
                                                    for crop_bbox in crop_regions[:3]:  # Show first 3 for debugging
                                                        overlap = calculate_bbox_overlap(crop_bbox, bbox)
                                                        max_overlap = max(max_overlap, overlap)
                                                        logger.info(f"  📏 Crop {crop_bbox} vs RT-DETR {bbox} = overlap {overlap:.3f}")
                                                    # Calculate max over all regions
                                                    for crop_bbox in crop_regions[3:]:
                                                        overlap = calculate_bbox_overlap(crop_bbox, bbox)
                                                        max_overlap = max(max_overlap, overlap)
                                                    
                                                    if max_overlap >= accepted_rtdetr_overlap:
                                                        # Accept high-confidence RT-DETR detection with sufficient spatial overlap
                                                        det = {
                                                            'confidence': confidence,
                                                            'bbox': bbox,
                                                            'source': f'megadetector_{variant_name}',
                                                            'class': class_id,
                                                            'raw_class_id': True,
                                                            'extended_class': True,
                                                            'spatial_overlap': max_overlap
                                                        }
                                                        detections.append(det)
                                                        md_v6_detections += 1
                                                        animals.append((class_id, confidence))
                                                        accepted_detection = True
                                                        logger.info(f"🔥 RT-DETR SPATIAL MATCH: class_id={class_id}, conf={confidence:.3f}, overlap={max_overlap:.3f}, bbox={bbox} at {timestamp_seconds:.1f}s")
                                                    else:
                                                        unknowns.append((class_id, confidence))
                                                        logger.info(f"🔍 RT-DETR HIGH-CONF REJECTED: class_id={class_id}, conf={confidence:.3f}, max_overlap={max_overlap:.3f} < {accepted_rtdetr_overlap} at {timestamp_seconds:.1f}s")
                                                else:
                                                    unknowns.append((class_id, confidence))
                                        
                                        # Summary logging (one line per category)
                                        if animals:
                                            logger.info(f"✅ MegaDetector v6 ANIMALS: {len(animals)} detections added {[(c, f'{conf:.3f}') for c, conf in animals]}")
                                        if persons:
                                            logger.info(f"⚪ MegaDetector v6 PERSONS: {len(persons)} detections {[(c, f'{conf:.3f}') for c, conf in persons]}")
                                        if unknowns:
                                            logger.info(f"📊 MegaDetector v6 UNKNOWNS: {len(unknowns)} detections {[(c, f'{conf:.3f}') for c, conf in unknowns]}")
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
                                    
                                    # Debug bbox format to catch tuple error
                                    logger.debug(f"Box type: {type(box)}, Box value: {box}")
                                    try:
                                        if hasattr(box, 'tolist'):
                                            bbox = box.tolist()[:4]
                                        elif isinstance(box, (list, tuple)):
                                            bbox = [float(box[j]) for j in range(min(4, len(box)))]
                                        else:
                                            logger.warning(f"Unexpected box format: {type(box)} - {box}")
                                            continue
                                    except Exception as e:
                                        logger.error(f"Error processing box {box}: {e}")
                                        continue
                                    confidence = float(conf)
                                    
                                    # Log high-confidence detections with spatial analysis  
                                    if confidence >= 0.3 and crop_regions is not None and len(crop_regions) > 0:
                                        max_overlap = 0.0
                                        for crop_bbox in crop_regions:
                                            overlap = calculate_bbox_overlap(crop_bbox, bbox)
                                            max_overlap = max(max_overlap, overlap)
                                        
                                        logger.info(f"🔍 {variant_name.upper()} high-conf: conf={confidence:.3f}, bbox={bbox}, max_overlap={max_overlap:.3f} at {timestamp_seconds:.1f}s")
                                        if len(crop_regions) <= 3:  # Show details for first few crop regions
                                            for crop_bbox in crop_regions:
                                                overlap = calculate_bbox_overlap(crop_bbox, bbox)
                                                logger.info(f"  📏 Crop {crop_bbox} vs {variant_name.upper()} {bbox} = overlap {overlap:.3f}")
                                    
                                    if confidence >= 0.05:
                                        det = {
                                            'confidence': confidence,
                                            'bbox': bbox,
                                            'source': f'megadetector_{variant_name}',
                                            'class': int(class_ids[i]) if class_ids is not None else 1
                                        }
                                        detections.append(det)
                                        md_v6_detections += 1
                                        logger.debug(f"MegaDetector v6: conf={confidence:.4f}, bbox={bbox}")
                            except Exception as e:
                                logger.error(f"Error parsing MegaDetector v6 supervision format: {e}")
                        else:
                            logger.debug(f"MegaDetector v6 no detections found in result dict")
                
                    logger.info(f"✅ MegaDetector v6 ({variant_name}): {md_v6_detections} detections found")
                    analysis_logger.info(f"MegaDetector v6 ({variant_name}) found {md_v6_detections} detections")
                    step_counter += 1
                except Exception as e:
                    import traceback
                    analysis_logger.error(f"{variant_name} failed: {e}")
                    logger.debug(f"MegaDetector v6 ({variant_name}) traceback: {traceback.format_exc()}")
        
        # 4. DeepFaune detector (PyTorch-Wildlife)
        if self.deepfaune_detector is not None:
            try:
                analysis_logger.info(f"ENSEMBLE_STEP_{step_counter}: Running DeepFaune detector (PyTorch-Wildlife)")
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
                                    
                                    # Debug bbox format to catch tuple error
                                    logger.debug(f"Box type: {type(box)}, Box value: {box}")
                                    try:
                                        if hasattr(box, 'tolist'):
                                            bbox = box.tolist()[:4]
                                        elif isinstance(box, (list, tuple)):
                                            bbox = [float(box[j]) for j in range(min(4, len(box)))]
                                        else:
                                            logger.warning(f"Unexpected box format: {type(box)} - {box}")
                                            continue
                                    except Exception as e:
                                        logger.error(f"Error processing box {box}: {e}")
                                        continue
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
        
        # Check for correlations between MegaDetector extended classes and YOLO detections
        self._check_extended_class_correlations(detections, timestamp_seconds)
        
        return detections
    
    def _check_extended_class_correlations(self, detections: List[Dict], timestamp: float) -> None:
        """Check for correlations between MegaDetector extended classes and YOLO detections."""
        megadetector_extended = [d for d in detections if d.get('source') == 'megadetector_v6' and d.get('raw_class_id')]
        yolo_detections = [d for d in detections if 'yolo' in d.get('source', '').lower()]
        
        if not megadetector_extended or not yolo_detections:
            return
            
        # Check for bbox overlaps (IoU > 0.3 indicates correlation)
        for md_det in megadetector_extended:
            md_bbox = md_det['bbox']
            md_class = md_det.get('class', 'unknown')
            md_conf = md_det['confidence']
            
            for yolo_det in yolo_detections:
                yolo_bbox = yolo_det['bbox']
                yolo_source = yolo_det['source']
                yolo_conf = yolo_det['confidence']
                
                iou = self._calculate_iou(md_bbox, yolo_bbox)
                if iou > 0.3:  # Significant overlap
                    logger.warning(f"🚨🚨🚨 CORRELATION DETECTED 🚨🚨🚨")
                    logger.warning(f"⚡ MegaDetector extended class {md_class} (conf={md_conf:.3f}) correlates with {yolo_source} (conf={yolo_conf:.3f})")
                    logger.warning(f"📍 IoU={iou:.3f}, timestamp={timestamp:.2f}s")
                    logger.warning(f"📊 MegaDetector bbox: {[round(x,1) for x in md_bbox]}")
                    logger.warning(f"📊 YOLO bbox: {[round(x,1) for x in yolo_bbox]}")
                    logger.warning(f"🔥 CONSIDER ADDING CLASS {md_class} TO WHITELIST!")
                    logger.warning(f"🚨🚨🚨 END CORRELATION 🚨🚨🚨")
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
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