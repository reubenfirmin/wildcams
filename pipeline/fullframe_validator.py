"""Full-frame validation for wildlife video processing."""

import dataclasses
import logging
from pathlib import Path
from typing import NamedTuple

import cv2

from core.data_types import (
    BoundingBox,
    CompositeScore,
    Detection,
    ExtendedTrack,
    MotionTrack,
    ScoredDetection,
    Track,
    ValidationSequence,
)
from pipeline.scoring import (
    calculate_bbox_overlap,
    calculate_composite_score,
    check_temporal_continuity,
)

logger = logging.getLogger("wildcams")


class _Candidate(NamedTuple):
    """A model detection that spatially overlaps a track's motion region."""

    detection: Detection
    overlap: float
    bbox_str: str


class FullFrameValidator:
    """Validates motion tracks using full-frame ML ensemble analysis with spatial overlap."""

    def __init__(self, ml_ensemble):
        """
        Initialize full-frame validator.

        Args:
            ml_ensemble: ML ensemble for running detections
        """
        self.ml_ensemble = ml_ensemble

        # Statistics tracking
        self._failed_tracks_data = []
        self._model_contributions = {}

        logger.info("🎯 Full-Frame Validator initialized")

    def validate_motion_tracks(
        self, video_path: Path, motion_tracks: list[MotionTrack], config
    ) -> list[ValidationSequence]:
        """
        Run full-frame analysis on motion tracks with spatial overlap validation.

        Args:
            video_path: Path to video file
            motion_tracks: List of motion tracks from previous steps

        Returns:
            List of validated sequences with spatial overlap confirmation
        """
        if not motion_tracks:
            return []

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or config.default_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"[STEP3] Running full-frame analysis on {len(motion_tracks)} motion tracks")

        # Initialize model contributions tracking for this video
        self._model_contributions = {}
        for model_name in config.ensemble_models:
            self._model_contributions[model_name] = {
                "total_detections": 0,
                "max_confidence": 0.0,
                "contributing_tracks": 0,
                "spatial_valid_count": 0,
                "total_score": 0.0,
            }

        # Convert motion tracks to extended tracks for validation
        extended_tracks = self._build_extended_tracks(motion_tracks, fps, total_frames)

        # Sample frames for analysis
        all_sample_frames, track_sample_frames = self._sample_track_frames(extended_tracks, config)

        logger.info(f"[STEP3] Processing {len(all_sample_frames)} unique frames across {len(extended_tracks)} tracks")

        # Initialize tracking
        track_detections: dict[int, list] = {track.track_id: [] for track in extended_tracks}
        frame_results = []

        # Process each sampled frame
        for frame_idx in sorted(all_sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = frame_idx / fps
            frame_result = self._process_frame(
                frame, frame_idx, timestamp, extended_tracks, track_detections, config, video_path
            )
            frame_results.append(frame_result)

        cap.release()

        # Evaluate tracks
        validated_results = self._evaluate_tracks(
            extended_tracks, track_detections, frame_results, track_sample_frames, fps, config, video_path
        )

        return validated_results

    def _build_extended_tracks(
        self, motion_tracks: list[MotionTrack], fps: float, total_frames: int
    ) -> list[ExtendedTrack]:
        """Convert basic motion tracks to extended bbox tracks for spatial validation."""
        extended_tracks = []

        for motion_track in motion_tracks:
            track_id = motion_track.track_id
            motion_frames = [region.frame_idx for region in motion_track.regions]
            motion_regions = motion_track.regions

            if not motion_frames:
                continue

            # Get representative bbox from motion regions
            if motion_regions and len(motion_regions) > 0:
                first_region = motion_regions[0]
                last_region = motion_regions[-1]
                first_known_position = [
                    first_region.bbox.x1,
                    first_region.bbox.y1,
                    first_region.bbox.x2,
                    first_region.bbox.y2,
                ]
                last_known_position = [
                    last_region.bbox.x1,
                    last_region.bbox.y1,
                    last_region.bbox.x2,
                    last_region.bbox.y2,
                ]
            else:
                first_known_position = [100, 100, 200, 200]
                last_known_position = first_known_position[:]

            motion_start_frame = min(motion_frames)
            motion_end_frame = max(motion_frames)

            # Bbox for each actual motion frame (first region wins on a duplicate frame_idx,
            # matching the previous first-match-then-break behavior). Coverage for backfill /
            # forward-fill frames is computed on demand in _get_track_bbox_for_frame rather
            # than materializing a TrackFrameBBox for every frame of the video.
            motion_bboxes: dict[int, list[float]] = {}
            for region in motion_regions:
                motion_bboxes.setdefault(
                    region.frame_idx,
                    [region.bbox.x1, region.bbox.y1, region.bbox.x2, region.bbox.y2],
                )

            extended_track = ExtendedTrack(
                track_id=track_id,
                motion_start_frame=motion_start_frame,
                motion_end_frame=motion_end_frame,
                first_known_position=first_known_position,
                last_known_position=last_known_position,
                motion_bboxes=motion_bboxes,
                duration_seconds=motion_track.duration_seconds,
                motion_frames=len(motion_frames),
                original_motion_track=motion_track,
            )

            extended_tracks.append(extended_track)

        return extended_tracks

    def _sample_track_frames(self, extended_tracks: list[ExtendedTrack], config) -> tuple[set, dict]:
        """Sample frames for analysis from all tracks."""
        all_sample_frames = set()
        track_sample_frames = {}

        for track in extended_tracks:
            track_id = track.track_id
            motion_track = track.original_motion_track
            motion_frames = [region.frame_idx for region in motion_track.regions]

            # Sample frames from motion track
            if len(motion_frames) <= config.max_validation_frames:
                sample_frames = motion_frames
            else:
                # Sample evenly distributed frames
                step = len(motion_frames) / config.max_validation_frames
                sample_frames = [motion_frames[int(i * step)] for i in range(config.max_validation_frames)]
                # Always include the last frame
                if motion_frames[-1] not in sample_frames:
                    sample_frames[-1] = motion_frames[-1]

            track_sample_frames[track_id] = sample_frames
            all_sample_frames.update(sample_frames)

        return all_sample_frames, track_sample_frames

    def _process_frame(
        self,
        frame,
        frame_idx: int,
        timestamp: float,
        extended_tracks: list[ExtendedTrack],
        track_detections: dict,
        config,
        video_path: Path,
    ) -> dict:
        """Process a single frame with ensemble models and validate against tracks."""
        frame_valid = False
        track_ensemble_results = []

        logger.info(f"EVAL | {video_path.stem} | {timestamp:.2f}s | {frame_idx}")

        # Run each model against the frame (original nested loop structure)
        for model_name in config.ensemble_models:
            # Full-frame mode: detect once on the whole frame and share across all tracks.
            # Crop mode (POC): detection runs per-track on the motion-region crop, below.
            full_frame_detections = None
            if not config.enable_crop_detection:
                full_frame_detections = self.ml_ensemble.run_single_model_detection(
                    model_name,
                    frame,
                    config,
                    timestamp_seconds=timestamp,
                    frame_idx=frame_idx,
                    full_frame=frame,
                    accepted_rtdetr_overlap=config.spatial_overlap_threshold,
                )
                self._record_model_contribution(model_name, full_frame_detections)

            # For each track, determine if there was an overlap above the threshold
            for track in extended_tracks:
                track_id = track.track_id
                track_bbox = self._get_track_bbox_for_frame(track, frame_idx)
                # "motion" = this is the track's own active frame (explicit); backfill/
                # forward_fill = a frame the track is not active in (implicit).
                fill_type = self._get_track_fill_type(track, frame_idx) if track_bbox is not None else "none"

                # Detection source for this (model, track): the shared full-frame set, or a
                # per-track crop of the motion region (POC). A crop detection is inherently
                # inside the motion region, so it cannot produce a no_overlap miss.
                model_detections: list[Detection] | None
                if config.enable_crop_detection:
                    # Only crop-detect where the track is actually active. Implicit frames
                    # would crop a stale static position on a frame the track isn't in
                    # (~90% of per-track evaluations, ~no signal), so skip the inference.
                    if track_bbox is not None and fill_type == "motion":
                        model_detections = self._detect_on_crop(
                            model_name, frame, track_bbox, config, timestamp, frame_idx
                        )
                        self._record_model_contribution(model_name, model_detections)
                    else:
                        model_detections = None
                else:
                    model_detections = full_frame_detections

                if track_bbox is not None:
                    overlap_type = "explicit" if fill_type == "motion" else f"implicit_{fill_type}"
                    motion_bbox_str = (
                        f"motn:{track_bbox[0]:.0f},{track_bbox[1]:.0f},{track_bbox[2]:.0f},{track_bbox[3]:.0f}"
                    )

                    overlapping_count = 0
                    valid_detections: list[_Candidate] = []

                    if model_detections:
                        for det in model_detections:
                            det_bbox = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                            overlap = calculate_bbox_overlap(track_bbox, det_bbox)

                            if overlap >= config.spatial_overlap_threshold:
                                valid_detections.append(
                                    _Candidate(
                                        detection=det,
                                        overlap=overlap,
                                        bbox_str=f"bbox:{det_bbox[0]:.0f},{det_bbox[1]:.0f},{det_bbox[2]:.0f},{det_bbox[3]:.0f}",
                                    )
                                )
                                overlapping_count += 1
                            elif overlap > 0.0:
                                overlapping_count += 1

                        # Process valid detections - create ONE synthetic detection per model per track
                        if valid_detections:
                            # Find best detection by score (confidence * overlap) after consensus boosting
                            consensus_boost = 1.0 + config.consensus_boost_per_detection * (len(valid_detections) - 1)

                            best_detection = None
                            best_score = 0.0

                            for valid_det in valid_detections:
                                det = valid_det.detection
                                overlap = valid_det.overlap
                                boosted_conf = min(1.0, det.confidence * consensus_boost)
                                score = boosted_conf * overlap

                                if score > best_score:
                                    best_score = score
                                    best_detection = {
                                        "detection": det,
                                        "overlap": overlap,
                                        "bbox_str": valid_det.bbox_str,
                                        "boosted_conf": boosted_conf,
                                        "score": score,
                                    }

                            # Log ONE synthetic detection representing the best from this model for this track
                            assert best_detection is not None  # set above when valid_detections is non-empty
                            det = best_detection["detection"]
                            overlap = best_detection["overlap"]
                            bbox_str = best_detection["bbox_str"]
                            boosted_conf = best_detection["boosted_conf"]
                            overall_score = best_detection["score"]

                            consensus_note = (
                                f"consensus:{len(valid_detections)}" if len(valid_detections) > 1 else "single"
                            )
                            logger.info(
                                f"✅ | {model_name} | {bbox_str} | conf:{det.confidence:.3f}→{boosted_conf:.3f} | ovlp:{overlap:.3f} | {motion_bbox_str} | scor:{overall_score:.3f} | spatial_valid | {overlap_type} | track_{track_id} | {consensus_note}"
                            )

                            # Track spatial valid contributions
                            self._model_contributions[model_name]["spatial_valid_count"] += 1
                            self._model_contributions[model_name]["total_score"] += overall_score

                            # Store ONE synthetic ScoredDetection for track evaluation. The wrapped
                            # Detection already carries source=model_name, frame_idx, timestamp.
                            track_detections[track_id].append(
                                ScoredDetection(
                                    detection=det,
                                    boosted_confidence=boosted_conf,
                                    motion_overlap=overlap,
                                    overlap_type=overlap_type,
                                    consensus_boost=consensus_boost,
                                    consensus_count=len(valid_detections),
                                )
                            )

                    # If no detections overlapped with this track, log summary
                    if overlapping_count == 0:
                        if model_detections:
                            non_overlapping = len(model_detections)
                            logger.info(
                                f"❌ | {model_name} | {non_overlapping}_detections | conf:0.000 | ovlp:0.000 | {motion_bbox_str} | scor:0.000 | no_overlap | {overlap_type} | track_{track_id}"
                            )
                        else:
                            logger.info(
                                f"❌ | {model_name} | none | conf:0.000 | ovlp:0.000 | {motion_bbox_str} | scor:0.000 | no_detection | {overlap_type} | track_{track_id}"
                            )
                    elif len(valid_detections) == 0 and overlapping_count > 0:
                        # All detections had some overlap but failed threshold
                        logger.info(
                            f"⚠️ | {model_name} | {overlapping_count}_detections | conf:0.000 | ovlp:0.000 | {motion_bbox_str} | scor:0.000 | threshold_failed | {overlap_type} | track_{track_id}"
                        )
                else:
                    # Track has no bbox for this frame
                    if model_detections:
                        logger.info(
                            f"❌ | {model_name} | {len(model_detections)}_detections | conf:0.000 | ovlp:0.000 | motn:none | scor:0.000 | no_track_bbox | track_{track_id}"
                        )
                    else:
                        logger.info(
                            f"❌ | {model_name} | none | conf:0.000 | ovlp:0.000 | motn:none | scor:0.000 | no_detection | track_{track_id}"
                        )

        # Calculate ensemble score per track for this frame (after all models processed)
        for track in extended_tracks:
            track_id = track.track_id
            track_frame_detections = [d for d in track_detections[track_id] if d.detection.frame_idx == frame_idx]

            # Always evaluate each track, even if no detections (use boosted confidence for ensemble scoring).
            # The per-frame gate is separately configurable; it defaults to confidence_threshold.
            track_ensemble_score = (
                sum(d.boosted_confidence for d in track_frame_detections) if track_frame_detections else 0.0
            )
            track_passed = track_ensemble_score >= config.frame_pass_confidence_threshold

            if track_passed:
                frame_valid = True

            track_ensemble_results.append(
                {
                    "track_id": track_id,
                    "ensemble_score": track_ensemble_score,
                    "detections": len(track_frame_detections),
                    "passed": track_passed,
                }
            )

        # Log overall frame result
        frame_icon = "✅" if frame_valid else "❌"
        passed_tracks = [r for r in track_ensemble_results if r["passed"]]
        frame_reason = f"{len(passed_tracks)}_tracks_passed" if frame_valid else "no_tracks_passed"

        logger.info("================================================================================")
        logger.info(
            f"{frame_icon} | FRAME_RESULT | {video_path.stem} | frame_{frame_idx} | {timestamp:.2f}s | tracks_evaluated={len(track_ensemble_results)} | tracks_passed={len(passed_tracks)} | {frame_reason}"
        )
        logger.info("================================================================================")

        return {"frame_idx": frame_idx, "track_results": track_ensemble_results, "frame_valid": frame_valid}

    def _record_model_contribution(self, model_name: str, detections: list[Detection] | None) -> None:
        """Fold a batch of detections into this video's per-model contribution stats."""
        if not detections:
            return
        contrib = self._model_contributions[model_name]
        contrib["total_detections"] += len(detections)
        contrib["max_confidence"] = max(contrib["max_confidence"], max(det.confidence for det in detections))

    def _detect_on_crop(
        self,
        model_name: str,
        frame,
        track_bbox: list[float],
        config,
        timestamp: float,
        frame_idx: int,
    ) -> list[Detection]:
        """POC crop-based detection: run one model on the padded motion-region crop and map
        detections back to full-frame pixel coordinates.

        Running on the crop makes a small animal large (higher confidence) and guarantees any
        detection lies inside the motion region, so it cannot be a no_overlap miss.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = track_bbox
        pad_x = (x2 - x1) * config.crop_detection_padding
        pad_y = (y2 - y1) * config.crop_detection_padding
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return []

        crop = frame[cy1:cy2, cx1:cx2]
        crop_detections = self.ml_ensemble.run_single_model_detection(
            model_name,
            crop,
            config,
            timestamp_seconds=timestamp,
            frame_idx=frame_idx,
            full_frame=crop,
            accepted_rtdetr_overlap=config.spatial_overlap_threshold,
        )

        # Map crop-local coordinates back to the full frame by adding the crop origin.
        mapped = []
        for det in crop_detections or []:
            mapped.append(
                dataclasses.replace(
                    det,
                    bbox=BoundingBox(
                        det.bbox.x1 + cx1,
                        det.bbox.y1 + cy1,
                        det.bbox.x2 + cx1,
                        det.bbox.y2 + cy1,
                    ),
                )
            )
        return mapped

    def _get_track_bbox_for_frame(self, track: ExtendedTrack, frame_idx: int) -> list[float] | None:
        """Bbox for a track at a frame (O(1), computed on demand).

        Before motion -> backfill position; after motion -> forward-fill position; during
        motion -> the region's bbox (None for an in-range frame with no motion region).
        """
        if frame_idx < track.motion_start_frame:
            return track.first_known_position
        if frame_idx > track.motion_end_frame:
            return track.last_known_position
        return track.motion_bboxes.get(frame_idx)

    def _get_track_fill_type(self, track: ExtendedTrack, frame_idx: int) -> str:
        """Fill type for a track at a frame. Only called when the bbox lookup is non-None,
        so an in-range frame here is always a motion frame."""
        if frame_idx < track.motion_start_frame:
            return "backfill"
        if frame_idx > track.motion_end_frame:
            return "forward_fill"
        return "motion"
        return "unknown"

    def _evaluate_tracks(
        self,
        extended_tracks: list[ExtendedTrack],
        track_detections: dict,
        frame_results: list[dict],
        track_sample_frames: dict,
        fps: float,
        config,
        video_path: Path,
    ) -> list[ValidationSequence]:
        """Evaluate tracks based on validation criteria."""
        validated_results = []

        for track in extended_tracks:
            track_id = track.track_id
            detections = track_detections[track_id]

            if not detections:
                continue

            # Calculate track statistics (use original confidence for reporting)
            avg_confidence = sum(d.detection.confidence for d in detections) / len(detections)
            max_confidence = max(d.detection.confidence for d in detections)
            summed_confidence = sum(d.detection.confidence for d in detections)

            # Boosted confidence drives the validation thresholding.
            summed_boosted_confidence = sum(d.boosted_confidence for d in detections)

            # Count passed frames for this track (distinct sampled frames in which it passed)
            track_frames = track_sample_frames[track_id]
            passed_frame_indices = []

            for frame_result in frame_results:
                if frame_result["frame_idx"] in track_frames:
                    track_result = next(
                        (tr for tr in frame_result["track_results"] if tr["track_id"] == track_id), None
                    )
                    if track_result and track_result["passed"]:
                        passed_frame_indices.append(frame_result["frame_idx"])
            passed_frames = len(passed_frame_indices)

            # Validation criteria (use boosted confidence for thresholding)
            confidence_passed = summed_boosted_confidence >= config.confidence_threshold
            # Count distinct passed frames, not synthetic detections (there is one detection
            # per model per frame, so len(detections) overcounts a single-frame track).
            frames_passed = passed_frames >= config.min_track_frames
            temporal_continuity_passed = check_temporal_continuity(passed_frame_indices, fps, config)

            validation_passed = confidence_passed and frames_passed and temporal_continuity_passed

            # Log track evaluation in original format
            logger.info(f"TRACK | {video_path.stem} | track_{track_id}")
            track_icon = "✅" if validation_passed else "❌"
            # Calculate composite score for logging (even for failed tracks)
            composite_score = (
                calculate_composite_score(track, detections, track_frames, fps, config)
                if detections
                else CompositeScore.empty()
            )

            logger.info(
                f"{track_icon} | duration={track.duration_seconds:.2f}s | frames_evaluated={len(track_frames)} | frames_passed={passed_frames} | detections={len(detections)} | summed_conf={summed_confidence:.3f} | avg_conf={avg_confidence:.3f} | max_conf={max_confidence:.3f} | ensemble={summed_boosted_confidence:.3f} | composite={composite_score.final_score:.3f} | models={composite_score.consensus_models} | motion_align={composite_score.motion_alignment:.3f} | temp_density={composite_score.temporal_density:.3f} | conf_pass={confidence_passed} | frames_pass={frames_passed} | temporal_pass={temporal_continuity_passed} | validated={validation_passed}"
            )

            if validation_passed:
                # Find best scored detection by original confidence
                best_scored = max(detections, key=lambda d: d.detection.confidence)
                best_raw = best_scored.detection

                # Calculate multi-dimensional confidence score
                composite_score = calculate_composite_score(track, detections, track_frames, fps, config)

                # Create typed best detection (tagged as a full-frame validation detection)
                best_detection = Detection(
                    confidence=best_raw.confidence,
                    bbox=best_raw.bbox,
                    source=f"fullframe_{best_raw.source}",
                    class_name="animal",
                    timestamp=best_raw.timestamp,
                    frame_idx=best_raw.frame_idx,
                )

                # The wrapped Detection objects are already typed
                typed_detections = [d.detection for d in detections]

                # Create track object
                track_obj = Track(
                    track_id=track_id,
                    detections=typed_detections,
                    start_frame=min(track_frames),
                    end_frame=max(track_frames),
                    duration_seconds=track.duration_seconds,
                    confidence_scores=[d.confidence for d in typed_detections],
                    bbox_sequence=[d.bbox for d in typed_detections],
                )

                # Create validation sequence
                validated_sequence = ValidationSequence(
                    sequence_id=track_id,
                    track=track_obj,
                    detections=typed_detections,
                    ensemble_score=summed_boosted_confidence,
                    composite_score=composite_score.final_score,
                    best_detection=best_detection,
                    frame_range=(min(track_frames), max(track_frames)),
                    duration_seconds=track.duration_seconds,
                )

                validated_results.append(validated_sequence)

        return validated_results

    def get_model_contributions(self) -> dict:
        """Get model contribution statistics for this video."""
        return self._model_contributions.copy()
