"""Frame extraction operations for wildlife video processing."""

import logging
from pathlib import Path

import cv2
import numpy as np

from .video_reader import VideoReader

logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger("analysis")


class FrameExtractor:
    """Handles frame extraction with various sampling strategies."""

    def __init__(self, max_frames: int = 20, sampling_strategy: str = "uniform", debug_dir: Path | None = None):
        """Initialize frame extractor."""
        self.max_frames = max_frames
        self.sampling_strategy = sampling_strategy
        self.debug_dir = debug_dir

    def extract_frames(self, video_reader: VideoReader) -> tuple[list[np.ndarray], list[float]]:
        """Extract frames from video using the configured strategy."""
        if not video_reader.is_opened():
            logger.error("❌ Video reader not opened")
            return [], []

        if video_reader.total_frames <= 0:
            logger.error(f"❌ Invalid frame count for {video_reader.video_path.name}")
            return [], []

        # Calculate frame indices to sample
        target_frame_indices = self._calculate_frame_indices(video_reader)

        logger.info(
            f"🎬 Processing {video_reader.video_path.name} ({video_reader.duration:.1f}s, {video_reader.total_frames} frames)"
        )
        logger.info(f"📍 Sampling {len(target_frame_indices)} frames using {self.sampling_strategy} strategy")

        # Extract frames at target indices
        frames = []
        timestamps = []

        for target_idx in target_frame_indices:
            success, frame = video_reader.get_frame_at_index(target_idx)

            if success and frame is not None:
                # Calculate timestamp for this frame
                timestamp_seconds = target_idx / video_reader.fps if video_reader.fps > 0 else 0
                frames.append(frame)
                timestamps.append(timestamp_seconds)
                analysis_logger.info(
                    f"Extracted frame {len(frames) - 1}: index={target_idx}, timestamp={timestamp_seconds:.2f}s"
                )
            else:
                logger.debug(f"⚠️ Failed to read frame {target_idx}")

        # Save frames to debug directory if specified
        if frames and self.debug_dir:
            self._save_debug_frames(frames, video_reader.video_path.stem)

        logger.info(f"📸 Successfully extracted {len(frames)} frames")
        return frames, timestamps

    def _calculate_frame_indices(self, video_reader: VideoReader) -> list[int]:
        """Calculate frame indices based on sampling strategy."""
        total_frames = video_reader.total_frames
        frames_to_extract = min(self.max_frames, total_frames)

        if self.sampling_strategy == "uniform":
            return self._uniform_sampling(total_frames, frames_to_extract)
        elif self.sampling_strategy == "temporal_clustering":
            # Future implementation for temporal clustering approach
            return self._uniform_sampling(total_frames, frames_to_extract)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _uniform_sampling(self, total_frames: int, frames_to_extract: int) -> list[int]:
        """Calculate evenly spaced frame indices."""
        if frames_to_extract == total_frames:
            # Extract every frame if video is short
            return list(range(total_frames))
        else:
            # Calculate evenly spaced frame indices
            step = total_frames / frames_to_extract
            return [int(i * step) for i in range(frames_to_extract)]

    def _save_debug_frames(self, frames: list[np.ndarray], video_name: str) -> None:
        """Save extracted frames to debug directory."""
        assert self.debug_dir is not None  # only called when debug_dir is set
        video_debug_dir = self.debug_dir / video_name
        video_debug_dir.mkdir(exist_ok=True)

        logger.info(f"💾 Saving {len(frames)} frames to debug directory...")
        for idx, frame in enumerate(frames):
            frame_path = video_debug_dir / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)

        logger.info(f"🛠️ Debug frames saved to: {video_debug_dir}")

    def extract_frames_from_path(self, video_path: Path) -> tuple[list[np.ndarray], list[float]]:
        """Convenience method to extract frames directly from video path."""
        with VideoReader(video_path) as reader:
            if not reader.is_opened():
                return [], []
            return self.extract_frames(reader)
