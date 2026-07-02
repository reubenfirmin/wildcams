"""
Pipeline step interface using typed objects.

Clean typed system with proper data structures.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

from config.processing_config import ProcessingConfig
from core.data_types import (
    AnimalClassificationResult,
    CameraHandlingResult,
    FullFrameValidationResult,
    MotionDetectionResult,
    ValidationResult,
    VideoAnalysis,
)

logger = logging.getLogger("wildcams")


class PipelineStep(ABC):
    """Abstract base class for pipeline steps.

    Each concrete step subclass declares its own `process` signature (steps take
    different inputs), so the base intentionally does not fix one here.
    """

    @abstractmethod
    def get_step_name(self) -> str:
        """Get the name of this step."""
        pass


# Union type for all possible step results
StepResult = MotionDetectionResult | CameraHandlingResult | FullFrameValidationResult | AnimalClassificationResult


class MotionDetectionStep(PipelineStep):
    """Motion detection step."""

    def get_step_name(self) -> str:
        return "motion_detection"

    @abstractmethod
    def process(self, video_path: Path, config: ProcessingConfig) -> MotionDetectionResult:
        """Process motion detection."""
        pass


class CameraHandlingStep(PipelineStep):
    """Camera handling step."""

    def get_step_name(self) -> str:
        return "camera_handling"

    @abstractmethod
    def process(
        self, video_path: Path, config: ProcessingConfig, motion_result: MotionDetectionResult
    ) -> CameraHandlingResult:
        """Process camera handling filtering."""
        pass


class FullFrameValidationStep(PipelineStep):
    """Full frame validation step."""

    def get_step_name(self) -> str:
        return "fullframe_validation"

    @abstractmethod
    def process(
        self,
        video_path: Path,
        config: ProcessingConfig,
        motion_result: MotionDetectionResult,
        camera_result: CameraHandlingResult,
    ) -> FullFrameValidationResult:
        """Process full frame validation."""
        pass


class AnimalClassificationStep(PipelineStep):
    """Animal classification step."""

    def get_step_name(self) -> str:
        return "animal_classification"

    @abstractmethod
    def process(
        self, video_path: Path, validation_result: FullFrameValidationResult, config: ProcessingConfig
    ) -> AnimalClassificationResult:
        """Process animal classification."""
        pass


class PipelineOrchestrator:
    """Pipeline orchestrator using typed objects."""

    def __init__(
        self,
        motion_step: MotionDetectionStep,
        camera_step: CameraHandlingStep,
        validation_step: FullFrameValidationStep,
        classification_step: AnimalClassificationStep | None = None,
    ):
        self.motion_step = motion_step
        self.camera_step = camera_step
        self.validation_step = validation_step
        self.classification_step = classification_step

    def process(self, video_path: Path, config: ProcessingConfig) -> VideoAnalysis:
        """Process video through entire typed pipeline."""
        pipeline_start = time.time()
        logger.info(f"⏱️  Pipeline START for {video_path.name}")

        # Step 1: Motion Detection
        step1_start = time.time()
        logger.info("⏱️  Step 1 (Motion Detection) START")
        motion_result = self.motion_step.process(video_path, config)
        step1_duration = time.time() - step1_start
        logger.info(f"⏱️  Step 1 (Motion Detection) COMPLETE - {step1_duration:.2f}s")

        if not motion_result.success:
            pipeline_duration = time.time() - pipeline_start
            logger.info(f"⏱️  Pipeline FAILED at Step 1 - Total: {pipeline_duration:.2f}s")
            return self._create_failed_analysis(video_path, motion_result, None, None)

        if motion_result.early_exit:
            pipeline_duration = time.time() - pipeline_start
            logger.info(f"⏱️  Pipeline EARLY EXIT at Step 1 - Total: {pipeline_duration:.2f}s")
            return self._create_early_exit_analysis(video_path, motion_result, None, None)

        # Step 2: Camera Handling
        step2_start = time.time()
        logger.info("⏱️  Step 2 (Camera Handling) START")
        camera_result = self.camera_step.process(video_path, config, motion_result)
        step2_duration = time.time() - step2_start
        logger.info(f"⏱️  Step 2 (Camera Handling) COMPLETE - {step2_duration:.2f}s")

        if not camera_result.success:
            pipeline_duration = time.time() - pipeline_start
            logger.info(f"⏱️  Pipeline FAILED at Step 2 - Total: {pipeline_duration:.2f}s")
            return self._create_failed_analysis(video_path, motion_result, camera_result, None)

        if camera_result.early_exit:
            pipeline_duration = time.time() - pipeline_start
            logger.info(f"⏱️  Pipeline EARLY EXIT at Step 2 - Total: {pipeline_duration:.2f}s")
            return self._create_early_exit_analysis(video_path, motion_result, camera_result, None)

        # Step 3: Full Frame Validation
        step3_start = time.time()
        logger.info("⏱️  Step 3 (Full Frame Validation) START")
        validation_result = self.validation_step.process(video_path, config, motion_result, camera_result)
        step3_duration = time.time() - step3_start
        logger.info(f"⏱️  Step 3 (Full Frame Validation) COMPLETE - {step3_duration:.2f}s")

        if not validation_result.success:
            pipeline_duration = time.time() - pipeline_start
            logger.info(f"⏱️  Pipeline FAILED at Step 3 - Total: {pipeline_duration:.2f}s")
            return self._create_failed_analysis(video_path, motion_result, camera_result, validation_result)

        # Step 4: Animal Classification (optional)
        classification_result = None
        if self.classification_step and config.enable_animal_classification:
            step4_start = time.time()
            logger.info("⏱️  Step 4 (Animal Classification) START")
            # Step 4 reads validated_sequences directly off the Step-3 result.
            classification_result = self.classification_step.process(video_path, validation_result, config)
            step4_duration = time.time() - step4_start
            logger.info(f"⏱️  Step 4 (Animal Classification) COMPLETE - {step4_duration:.2f}s")

            # If classification filtered out all sequences, create failed analysis but KEEP classification_result
            if len(classification_result.animal_sequences) == 0:
                pipeline_duration = time.time() - pipeline_start
                logger.info(f"⏱️  Pipeline COMPLETE (filtered by classification) - Total: {pipeline_duration:.2f}s")
                return self._create_classification_filtered_analysis(
                    video_path, motion_result, camera_result, validation_result, classification_result
                )

        # Create complete analysis
        pipeline_duration = time.time() - pipeline_start
        logger.info(f"⏱️  Pipeline COMPLETE - Total: {pipeline_duration:.2f}s")
        return self._create_successful_analysis(
            video_path, motion_result, camera_result, validation_result, classification_result
        )

    def _create_successful_analysis(
        self,
        video_path: Path,
        motion_result: MotionDetectionResult,
        camera_result: CameraHandlingResult,
        validation_result: FullFrameValidationResult,
        classification_result: AnimalClassificationResult | None = None,
    ) -> VideoAnalysis:
        """Create successful video analysis."""

        # Use Step 4 results if available, otherwise use Step 3 results
        if classification_result and classification_result.classification_enabled:
            # Step 4 ran - use its results
            if not classification_result.animal_sequences:
                # Step 4 filtered out all sequences - no animals
                return self._create_classification_filtered_analysis(
                    video_path, motion_result, camera_result, validation_result, classification_result
                )

            # Use Step 4 confirmed sequences
            sequences_to_use = [cs.sequence for cs in classification_result.animal_sequences]
            best_sequence = max(sequences_to_use, key=lambda s: s.ensemble_score)

            # Add species information from Step 4
            species_info = classification_result.species_counts
            species_summary = (
                ", ".join(f"{species}({count})" for species, count in species_info.items()) if species_info else None
            )

        else:
            # No Step 4 or Step 4 disabled - use Step 3 results
            if not validation_result.validated_sequences:
                return self._create_failed_analysis(video_path, motion_result, camera_result, validation_result)

            sequences_to_use = validation_result.validated_sequences
            best_sequence = max(sequences_to_use, key=lambda s: s.ensemble_score)
            species_summary = None

        # Create validation result
        validation = ValidationResult(
            animals_detected=True,
            confidence=best_sequence.best_detection.confidence,
            ensemble_score=best_sequence.ensemble_score,
            composite_score=best_sequence.composite_score,
            validated_sequences_count=len(sequences_to_use),
            best_detection=best_sequence.best_detection,
        )

        total_processing_time = (
            motion_result.metadata.timing.duration
            + camera_result.metadata.timing.duration
            + validation_result.metadata.timing.duration
        )

        # Add Step 4 processing time if available
        if classification_result:
            total_processing_time += classification_result.processing_time

        return VideoAnalysis(
            video_path=video_path,
            video_name=video_path.name,
            processing_time=total_processing_time,
            validation_result=validation,
            motion_result=motion_result,
            camera_result=camera_result,
            fullframe_result=validation_result,
            species_summary=species_summary,
            classification_result=classification_result,
        )

    def _create_failed_analysis(
        self,
        video_path: Path,
        motion_result: MotionDetectionResult,
        camera_result: CameraHandlingResult | None,
        validation_result: FullFrameValidationResult | None,
    ) -> VideoAnalysis:
        """Create failed video analysis."""
        from core.data_types import (
            BoundingBox,
            Detection,
            create_empty_camera_result,
            create_empty_validation_result,
        )

        # Create empty detection for failed cases
        empty_detection = Detection(confidence=0.0, bbox=BoundingBox(0, 0, 0, 0), source="none", class_name="none")

        validation = ValidationResult(
            animals_detected=False,
            confidence=0.0,
            ensemble_score=0.0,
            composite_score=0.0,
            validated_sequences_count=0,
            best_detection=empty_detection,
            reason="processing_failed",
        )

        return VideoAnalysis(
            video_path=video_path,
            video_name=video_path.name,
            processing_time=motion_result.metadata.timing.duration,
            validation_result=validation,
            motion_result=motion_result,
            camera_result=camera_result or create_empty_camera_result("not_processed"),
            fullframe_result=validation_result or create_empty_validation_result("not_processed"),
        )

    def _create_early_exit_analysis(
        self,
        video_path: Path,
        motion_result: MotionDetectionResult,
        camera_result: CameraHandlingResult | None,
        validation_result: FullFrameValidationResult | None,
    ) -> VideoAnalysis:
        """Create early exit video analysis."""
        from core.data_types import (
            BoundingBox,
            Detection,
            create_empty_camera_result,
            create_empty_validation_result,
        )

        # Create empty detection for early exit cases
        empty_detection = Detection(confidence=0.0, bbox=BoundingBox(0, 0, 0, 0), source="none", class_name="none")

        # Determine reason for early exit
        if motion_result.early_exit:
            reason = motion_result.early_exit_reason
        elif camera_result and camera_result.early_exit:
            reason = camera_result.early_exit_reason
        else:
            reason = "early_exit"

        validation = ValidationResult(
            animals_detected=False,
            confidence=0.0,
            ensemble_score=0.0,
            composite_score=0.0,
            validated_sequences_count=0,
            best_detection=empty_detection,
            reason=reason,
        )

        return VideoAnalysis(
            video_path=video_path,
            video_name=video_path.name,
            processing_time=(
                motion_result.metadata.timing.duration
                + (camera_result.metadata.timing.duration if camera_result else 0.0)
            ),
            validation_result=validation,
            motion_result=motion_result,
            camera_result=camera_result or create_empty_camera_result("early_exit"),
            fullframe_result=validation_result or create_empty_validation_result("early_exit"),
        )

    def _create_classification_filtered_analysis(
        self,
        video_path: Path,
        motion_result: MotionDetectionResult,
        camera_result: CameraHandlingResult,
        validation_result: FullFrameValidationResult,
        classification_result: AnimalClassificationResult,
    ) -> VideoAnalysis:
        """Create analysis when Step 4 filtered out all sequences as non-animals."""
        from core.data_types import BoundingBox, Detection

        # Create empty detection for filtered cases
        empty_detection = Detection(confidence=0.0, bbox=BoundingBox(0, 0, 0, 0), source="none", class_name="none")

        validation = ValidationResult(
            animals_detected=False,
            confidence=0.0,
            ensemble_score=0.0,
            composite_score=0.0,
            validated_sequences_count=0,
            best_detection=empty_detection,
            reason="filtered_by_animal_classification",
        )

        total_processing_time = (
            motion_result.metadata.timing.duration
            + camera_result.metadata.timing.duration
            + validation_result.metadata.timing.duration
            + classification_result.processing_time
        )

        return VideoAnalysis(
            video_path=video_path,
            video_name=video_path.name,
            processing_time=total_processing_time,
            validation_result=validation,
            motion_result=motion_result,
            camera_result=camera_result,
            fullframe_result=validation_result,
            classification_result=classification_result,  # KEEP the classification result!
        )
