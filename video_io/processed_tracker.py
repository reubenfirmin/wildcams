"""Processed video tracking for wildlife video processing."""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessedVideoTracker:
    """Manages tracking of processed videos using .processed marker files."""

    def __init__(self, video_dir: Path):
        """Initialize processed video tracker."""
        self.video_dir = video_dir

    def get_unprocessed_videos(self, video_extensions: list[str] | None = None) -> list[Path]:
        """Get list of videos that haven't been processed yet."""
        if video_extensions is None:
            video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM"]

        processed_files: set[str] = set()
        video_files: list[Path] = []

        # Find all .processed files first
        for processed_file in self.video_dir.glob("*.processed"):
            # Extract the base name without .processed extension
            base_name = processed_file.stem
            processed_files.add(base_name)

        # Find all video files
        for ext in video_extensions:
            video_files.extend(self.video_dir.glob(f"*{ext}"))

        # Filter out already processed videos
        unprocessed_videos = []
        for video_file in video_files:
            if video_file.stem not in processed_files:
                unprocessed_videos.append(video_file)

        logger.info(f"🎬 Found {len(video_files)} total videos, {len(unprocessed_videos)} unprocessed")
        return sorted(unprocessed_videos)

    def get_filtered_videos(self, video_filter: list[str], video_extensions: list[str] | None = None) -> list[Path]:
        """Get videos based on filter criteria, ignoring .processed status."""
        if video_extensions is None:
            video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        # Get all videos in directory (try both lowercase and uppercase extensions)
        all_videos: list[Path] = []
        for ext in video_extensions:
            all_videos.extend(self.video_dir.glob(f"*{ext}"))
            all_videos.extend(self.video_dir.glob(f"*{ext.upper()}"))

        filtered_videos = []

        for video_filter_item in video_filter:
            if video_filter_item.isdigit():
                # Filter by video index - try IMG_xxxx.MP4 format first
                video_index = int(video_filter_item)
                video_name = f"IMG_{video_index:04d}.MP4"
                matching = [v for v in all_videos if v.name == video_name]

                if matching:
                    filtered_videos.extend(matching)
                elif 0 <= video_index < len(all_videos):
                    # Fallback to index-based selection
                    filtered_videos.append(all_videos[video_index])
                else:
                    logger.warning(
                        f"⚠️ Video index {video_index} not found (no IMG_{video_index:04d}.MP4 and index out of range)"
                    )
            else:
                # Filter by name pattern
                matching_videos = [v for v in all_videos if video_filter_item in v.name]
                filtered_videos.extend(matching_videos)

        # Remove duplicates while preserving order
        seen = set()
        unique_filtered = []
        for video in filtered_videos:
            if video not in seen:
                seen.add(video)
                unique_filtered.append(video)

        logger.info(f"🎯 Filter matched {len(unique_filtered)} videos")
        return unique_filtered

    def mark_as_processed(self, video_path: Path) -> None:
        """Mark a video as processed by creating a .processed file."""
        try:
            processed_path = video_path.with_suffix(video_path.suffix + ".processed")
            with open(processed_path, "w") as f:
                f.write(f"Processed at: {datetime.now().isoformat()}\n")
            logger.info(f"✅ Marked as processed: {video_path.name}")
        except Exception as e:
            logger.error(f"❌ Failed to mark {video_path.name} as processed: {e}")

    def is_processed(self, video_path: Path) -> bool:
        """Check if a video has been processed."""
        processed_path = video_path.with_suffix(video_path.suffix + ".processed")
        return processed_path.exists()

    def unmark_processed(self, video_path: Path) -> bool:
        """Remove processed marker for a video."""
        try:
            processed_path = video_path.with_suffix(video_path.suffix + ".processed")
            if processed_path.exists():
                processed_path.unlink()
                logger.info(f"🔄 Removed processed marker: {video_path.name}")
                return True
            else:
                logger.warning(f"⚠️ No processed marker found for: {video_path.name}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to remove processed marker for {video_path.name}: {e}")
            return False

    def get_all_processed_videos(self) -> list[Path]:
        """Get list of all processed videos."""
        processed_markers = list(self.video_dir.glob("*.processed"))
        processed_videos = []

        for marker in processed_markers:
            # Find corresponding video file
            base_name = marker.stem
            for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                video_path = self.video_dir / f"{base_name}{ext}"
                if video_path.exists():
                    processed_videos.append(video_path)
                    break

        return sorted(processed_videos)

    def get_processing_stats(self) -> dict:
        """Get statistics about processed vs unprocessed videos."""
        unprocessed = self.get_unprocessed_videos()
        processed = self.get_all_processed_videos()
        total = len(unprocessed) + len(processed)

        return {
            "total_videos": total,
            "processed_count": len(processed),
            "unprocessed_count": len(unprocessed),
            "processing_rate": len(processed) / total if total > 0 else 0.0,
        }
