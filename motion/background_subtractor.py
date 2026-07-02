"""Background subtractor factory for motion detection."""

import logging

import cv2

from config.processing_config import ProcessingConfig

logger = logging.getLogger("wildcams")


class BackgroundSubtractorFactory:
    """Factory for creating OpenCV background subtractors."""

    @staticmethod
    def create_subtractor(method: str, config: ProcessingConfig):
        """
        Create a background subtractor based on method and configuration.

        Args:
            method: Method name ('MOG2' or 'KNN')
            config: ProcessingConfig object

        Returns:
            OpenCV background subtractor object
        """
        if method == "MOG2":
            return BackgroundSubtractorFactory._create_mog2(config)
        elif method == "KNN":
            return BackgroundSubtractorFactory._create_knn(config)
        else:
            raise ValueError(f"Unknown background subtraction method: {method}")

    @staticmethod
    def _create_mog2(config: ProcessingConfig):
        """Create MOG2 background subtractor."""
        subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=config.motion_var_threshold, history=config.motion_history
        )

        logger.info("🔍 Created MOG2 background subtractor:")
        logger.info(f"  Variance threshold: {config.motion_var_threshold}")
        logger.info(f"  History: {config.motion_history} frames")
        logger.info("  Shadow detection: True")

        return subtractor

    @staticmethod
    def _create_knn(config: ProcessingConfig):
        """Create KNN background subtractor."""
        subtractor = cv2.createBackgroundSubtractorKNN(
            detectShadows=True, dist2Threshold=400, history=config.motion_history
        )

        logger.info("🔍 Created KNN background subtractor:")
        logger.info("  Distance threshold: 400")
        logger.info(f"  History: {config.motion_history} frames")
        logger.info("  Shadow detection: True")

        return subtractor
