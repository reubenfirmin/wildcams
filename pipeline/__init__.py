"""Pipeline package for wildlife video processing."""

from .step_interface import PipelineStep, StepInput, StepOutput
from .pipeline_orchestrator import PipelineOrchestrator

__all__ = [
    'PipelineStep',
    'StepInput', 
    'StepOutput',
    'PipelineOrchestrator'
]