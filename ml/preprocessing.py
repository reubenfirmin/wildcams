"""
Preprocessing Pipeline for ML Detection Ensemble.
Handles TTA, multi-scale detection, and frame preprocessing.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger("wildcams")


class PreprocessingPipeline:
    """Handles all preprocessing operations for ML detection."""

    def __init__(self, enable_tta: bool = True, enable_multiscale: bool = True):
        self.enable_tta = enable_tta
        self.enable_multiscale = enable_multiscale

        # Test-Time Augmentation settings
        self.tta_transforms = ["original", "horizontal_flip", "brightness_adjust", "contrast_adjust", "gaussian_blur"]

        # Multi-scale detection settings
        self.detection_scales = [0.8, 1.0, 1.2, 1.5]

    def apply_tta_transforms(self, frame: np.ndarray) -> list[tuple[np.ndarray, str]]:
        """
        Apply Test-Time Augmentation transforms.

        Args:
            frame: Input frame to transform

        Returns:
            List of (transformed_frame, transform_name) tuples
        """
        transforms = []

        # Original frame (always included)
        transforms.append((frame.copy(), "original"))

        if not self.enable_tta:
            return transforms

        # Horizontal flip
        if "horizontal_flip" in self.tta_transforms:
            flipped = cv2.flip(frame, 1)
            transforms.append((flipped, "horizontal_flip"))

        # Brightness adjustment (+20%)
        if "brightness_adjust" in self.tta_transforms:
            bright = cv2.convertScaleAbs(frame, alpha=1.0, beta=20)
            transforms.append((bright, "brightness_adjust"))

        # Contrast adjustment (1.2x)
        if "contrast_adjust" in self.tta_transforms:
            contrast = cv2.convertScaleAbs(frame, alpha=1.2, beta=0)
            transforms.append((contrast, "contrast_adjust"))

        # Gaussian blur (slight)
        if "gaussian_blur" in self.tta_transforms:
            blurred = cv2.GaussianBlur(frame, (3, 3), 0.5)
            transforms.append((blurred, "gaussian_blur"))

        return transforms

    def apply_multiscale_detection(self, frame: np.ndarray) -> list[tuple[np.ndarray, float, str]]:
        """
        Apply multi-scale detection.

        Args:
            frame: Input frame to scale

        Returns:
            List of (scaled_frame, scale_factor, scale_name) tuples
        """
        if not self.enable_multiscale:
            return [(frame, 1.0, "scale_1.0x")]

        scaled_frames = []

        for scale in self.detection_scales:
            scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            scale_name = f"scale_{scale:.1f}x"
            scaled_frames.append((scaled_frame, scale, scale_name))

        return scaled_frames

    def apply_histogram_equalization(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast.

        Args:
            frame: Input frame

        Returns:
            Enhanced frame
        """
        # Convert to YUV color space
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # Apply histogram equalization to Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        return enhanced

    def apply_clahe(
        self, frame: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Args:
            frame: Input frame
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of the neighborhood area

        Returns:
            Enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame for better model performance.

        Args:
            frame: Input frame

        Returns:
            Normalized frame
        """
        # Convert to float32 and normalize to [0, 1]
        normalized = frame.astype(np.float32) / 255.0

        # Apply ImageNet normalization (commonly used by pre-trained models)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        normalized = (normalized - mean) / std

        return normalized

    def denormalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Denormalize frame back to [0, 255] range.

        Args:
            frame: Normalized frame

        Returns:
            Denormalized frame
        """
        # Reverse ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        denormalized = (frame * std) + mean

        # Convert back to [0, 255] and uint8
        denormalized = np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)

        return denormalized

    def preprocess_for_model(self, frame: np.ndarray, model_type: str) -> np.ndarray:
        """
        Apply model-specific preprocessing.

        Args:
            frame: Input frame
            model_type: Type of model ('yolo', 'rtdetr', 'megadetector')

        Returns:
            Preprocessed frame
        """
        if model_type.lower() in ["yolo", "rtdetr"]:
            # YOLO and RT-DETR models typically expect BGR input
            return frame
        elif model_type.lower() == "megadetector":
            # MegaDetector may have specific preprocessing requirements
            return frame
        else:
            # Default preprocessing
            return frame
