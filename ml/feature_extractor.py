"""
Feature Extractor for ML Detection Ensemble.
Handles ResNet18-based feature extraction for clustering.
"""

import cv2
import torch
import numpy as np
import logging
from typing import List, Optional
import torchvision.transforms as transforms

logger = logging.getLogger('wildcams')

class FeatureExtractor:
    """Extracts ResNet18 features from detection crops for clustering."""
    
    def __init__(self, feature_extractor_model):
        """
        Initialize with a pre-loaded ResNet18 model.
        
        Args:
            feature_extractor_model: Pre-loaded ResNet18 model from ModelManager
        """
        self.feature_extractor = feature_extractor_model
        
        # Preprocessing transform for ResNet18
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
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
            input_tensor = self.transform(crop_resized).unsqueeze(0)
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
    
    def extract_batch_features(self, frame: np.ndarray, bboxes: List[List[float]]) -> List[Optional[np.ndarray]]:
        """
        Extract features from multiple bounding boxes in batch.
        
        Args:
            frame: OpenCV frame (BGR format)
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            List of feature vectors (or None for failed extractions)
        """
        if self.feature_extractor is None:
            return [None] * len(bboxes)
        
        features_list = []
        
        try:
            # Prepare all crops
            crops = []
            valid_indices = []
            
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    features_list.append(None)
                    continue
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    features_list.append(None)
                    continue
                
                # Prepare for ResNet18
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_resized = cv2.resize(crop_rgb, (224, 224))
                
                crops.append(crop_resized)
                valid_indices.append(i)
            
            if not crops:
                return [None] * len(bboxes)
            
            # Batch process valid crops
            batch_tensor = torch.stack([self.transform(crop) for crop in crops])
            if torch.cuda.is_available():
                batch_tensor = batch_tensor.cuda()
            
            # Extract features in batch
            with torch.no_grad():
                batch_features = self.feature_extractor(batch_tensor)
                batch_features = batch_features.cpu().numpy()
            
            # Insert results at correct positions
            feature_results = [None] * len(bboxes)
            for i, valid_idx in enumerate(valid_indices):
                feature_results[valid_idx] = batch_features[i].flatten()
            
            return feature_results
            
        except Exception as e:
            logger.error(f"Batch feature extraction failed: {e}")
            return [None] * len(bboxes)
    
    def calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Cosine similarity between 0 and 1
        """
        try:
            # Normalize features
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.debug(f"Feature similarity calculation failed: {e}")
            return 0.0
    
    def is_available(self) -> bool:
        """Check if feature extractor is available."""
        return self.feature_extractor is not None