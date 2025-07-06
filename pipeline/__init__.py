"""Pipeline package for wildlife video processing."""

from .step_interface_v2 import (
    PipelineOrchestrator,
    MotionDetectionStep,
    CameraHandlingStep,
    FullFrameValidationStep
)

__all__ = [
    'PipelineOrchestrator',
    'MotionDetectionStep',
    'CameraHandlingStep',
    'FullFrameValidationStep'
]