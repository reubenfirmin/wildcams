"""Pipeline step interface and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np


@dataclass
class StepInput:
    """Input data structure for pipeline steps."""
    video_path: Path
    frames: Optional[List[np.ndarray]] = None
    timestamps: Optional[List[float]] = None
    motion_tracks: Optional[List[Dict]] = None
    detections: Optional[List[Dict]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class StepOutput:
    """Output data structure for pipeline steps."""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    early_exit: bool = False
    early_exit_reason: Optional[str] = None
    
    def __post_init__(self):
        """Initialize data and metadata if not provided."""
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    def __init__(self):
        """Initialize step."""
        self.step_name = self.__class__.__name__
    
    @abstractmethod
    def process(self, step_input: StepInput, config) -> StepOutput:
        """
        Process the input data and return results.
        
        Args:
            step_input: Input data for this step
            
        Returns:
            StepOutput containing results and metadata
        """
        pass
    
    def validate_input(self, step_input: StepInput) -> bool:
        """
        Validate that the input contains required data for this step.
        
        Args:
            step_input: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        return True
    
    def get_step_name(self) -> str:
        """Get the name of this pipeline step."""
        return self.step_name