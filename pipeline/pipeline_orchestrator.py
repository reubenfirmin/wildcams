"""Pipeline orchestrator for coordinating processing steps."""

import logging
from typing import List, Dict, Any
from pathlib import Path

from .step_interface import PipelineStep, StepInput, StepOutput

logger = logging.getLogger('wildcams')


class PipelineOrchestrator:
    """Orchestrates execution of processing pipeline steps."""
    
    def __init__(self, steps: List[PipelineStep]):
        """
        Initialize orchestrator with pipeline steps.
        
        Args:
            steps: List of pipeline steps to execute in order
        """
        self.steps = steps
        self.pipeline_name = "WildlifeCameraPipeline"
        
        logger.info(f"🔗 Pipeline initialized with {len(self.steps)} steps:")
        for i, step in enumerate(self.steps, 1):
            logger.info(f"  Step {i}: {step.get_step_name()}")
    
    def process(self, video_path: Path, config, initial_data: Dict[str, Any] = None) -> StepOutput:
        """
        Execute the complete pipeline on a video.
        
        Args:
            video_path: Path to video file to process
            initial_data: Optional initial data to pass to first step
            
        Returns:
            Final StepOutput from the last executed step
        """
        logger.info(f"🎬 Starting pipeline processing for: {video_path.name}")
        
        # Initialize input for first step
        step_input = StepInput(
            video_path=video_path,
            metadata=initial_data or {}
        )
        
        current_output = None
        
        for step_num, step in enumerate(self.steps, 1):
            step_name = step.get_step_name()
            logger.info(f"▶️ Step {step_num}/{len(self.steps)}: {step_name}")
            
            # Validate input for this step
            if not step.validate_input(step_input):
                logger.error(f"❌ Step {step_num} input validation failed")
                return StepOutput(
                    success=False,
                    data={},
                    metadata={"error": f"Input validation failed for {step_name}"},
                    early_exit=True,
                    early_exit_reason=f"Input validation failed for {step_name}"
                )
            
            # Execute step
            current_output = step.process(step_input, config)
            
            if not current_output.success:
                logger.error(f"❌ Step {step_num} ({step_name}) failed")
                return current_output
            
            # Check for early exit
            if current_output.early_exit:
                logger.info(f"⏹️ Early exit at step {step_num} ({step_name}): {current_output.early_exit_reason}")
                return current_output
            
            logger.info(f"✅ Step {step_num} ({step_name}) completed successfully")
            
            # Prepare input for next step
            if step_num < len(self.steps):
                step_input = self._prepare_next_input(step_input, current_output)
        
        logger.info(f"🎉 Pipeline processing completed successfully for: {video_path.name}")
        return current_output
    
    def _prepare_next_input(self, current_input: StepInput, current_output: StepOutput) -> StepInput:
        """
        Prepare input for the next pipeline step.
        
        Args:
            current_input: Input that was used for current step
            current_output: Output from current step
            
        Returns:
            StepInput for next step
        """
        # Carry forward video path and merge data
        next_input = StepInput(
            video_path=current_input.video_path,
            metadata=current_input.metadata.copy()
        )
        
        # Update with output data
        next_input.metadata.update(current_output.metadata)
        
        # Extract specific data types from output
        if 'frames' in current_output.data:
            next_input.frames = current_output.data['frames']
        elif current_input.frames is not None:
            next_input.frames = current_input.frames
            
        if 'timestamps' in current_output.data:
            next_input.timestamps = current_output.data['timestamps']
        elif current_input.timestamps is not None:
            next_input.timestamps = current_input.timestamps
            
        if 'motion_tracks' in current_output.data:
            next_input.motion_tracks = current_output.data['motion_tracks']
        elif current_input.motion_tracks is not None:
            next_input.motion_tracks = current_input.motion_tracks
            
        if 'detections' in current_output.data:
            next_input.detections = current_output.data['detections']
        elif current_input.detections is not None:
            next_input.detections = current_input.detections
        
        return next_input
    
    def get_step_count(self) -> int:
        """Get the number of steps in this pipeline."""
        return len(self.steps)
    
    def get_step_names(self) -> List[str]:
        """Get names of all steps in this pipeline."""
        return [step.get_step_name() for step in self.steps]