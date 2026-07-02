"""Pipeline package for wildlife video processing."""

from .step_interface_v2 import CameraHandlingStep, FullFrameValidationStep, MotionDetectionStep, PipelineOrchestrator

__all__ = ["PipelineOrchestrator", "MotionDetectionStep", "CameraHandlingStep", "FullFrameValidationStep"]
