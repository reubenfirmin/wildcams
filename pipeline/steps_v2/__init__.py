"""
Pipeline steps using typed objects.

Clean implementations with proper data structures.
"""

from .motion_detection_step import MotionDetectionStepImpl
from .camera_handling_step import CameraHandlingStepImpl
from .fullframe_validation_step import FullFrameValidationStepImpl

__all__ = [
    'MotionDetectionStepImpl',
    'CameraHandlingStepImpl', 
    'FullFrameValidationStepImpl'
]