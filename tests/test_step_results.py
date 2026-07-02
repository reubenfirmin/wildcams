"""Tests for the tightened pipeline-step result types."""

from core.data_types import (
    MotionDetectionResult, MotionDetectionMetadata, StepTiming, InferenceResult,
)


def test_motion_result_exposes_tracks_directly():
    r = MotionDetectionResult(
        success=True,
        motion_tracks=[],
        metadata=MotionDetectionMetadata(
            step_name="motion", timing=StepTiming(0.0, 1.0, 1.0),
            tracks_found=0, total_motion_area=0.0, motion_method="MOG2",
        ),
    )
    assert r.motion_tracks == []
    assert not hasattr(r, "data")


def test_inference_result_carries_predictions():
    r = InferenceResult(
        model_name="BioCLIP", is_animal=True, animal_confidence=0.4,
        species="coati", species_confidence=0.4, can_identify_species=True,
        processing_time=0.1, all_predictions={"coati": 0.4, "agouti": 0.2},
    )
    assert r.all_predictions["coati"] == 0.4


def test_inference_result_predictions_default_none():
    r = InferenceResult(
        model_name="DeepFaune", is_animal=False, animal_confidence=0.1,
        species=None, species_confidence=0.0, can_identify_species=False,
        processing_time=0.1,
    )
    assert r.all_predictions is None
