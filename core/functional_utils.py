"""
Pure functional utilities for wildlife video processing.

Pure functions for video-file collection and filtering, extracted from the
core classes to keep them side-effect-free and unit-testable.
"""

from pathlib import Path

from core.constants import VIDEO_EXTENSIONS


def collect_video_files(video_dir: Path, extensions: set[str] | None = None) -> list[Path]:
    """
    Collect video files from a directory.

    Args:
        video_dir: Directory to search for videos
        extensions: Set of file extensions to match (defaults to VIDEO_EXTENSIONS)

    Returns:
        Sorted list of video file paths
    """
    if extensions is None:
        extensions = VIDEO_EXTENSIONS

    all_videos: list[Path] = []
    for ext in extensions:
        all_videos.extend(video_dir.glob(f"*{ext}"))

    return sorted(all_videos, key=lambda p: p.name)


def filter_videos_by_criteria(all_videos: list[Path], video_filter: list[int | str]) -> list[Path]:
    """
    Filter videos by 1-based indices or name substrings.

    Args:
        all_videos: Complete list of video files
        video_filter: List of indices (1-based) or name patterns

    Returns:
        Filtered list of video files
    """
    filtered_videos = []

    for item in video_filter:
        if isinstance(item, int):
            if 1 <= item <= len(all_videos):
                filtered_videos.append(all_videos[item - 1])
        else:
            matching_videos = [v for v in all_videos if item in v.name]
            filtered_videos.extend(matching_videos)

    return filtered_videos


def validate_video_filter(video_filter: list[int | str], available_videos: list[Path]) -> tuple[list[Path], list[str]]:
    """
    Apply a video filter and collect warnings for entries that matched nothing.

    Args:
        video_filter: List of indices or name patterns
        available_videos: List of available video files

    Returns:
        Tuple of (filtered_videos, warnings)
    """
    filtered_videos = []
    warnings = []

    for item in video_filter:
        if isinstance(item, int):
            if 1 <= item <= len(available_videos):
                filtered_videos.append(available_videos[item - 1])
            else:
                warnings.append(f"Video index {item} out of range (1-{len(available_videos)})")
        else:
            matching_videos = [v for v in available_videos if item in v.name]
            if matching_videos:
                filtered_videos.extend(matching_videos)
            else:
                warnings.append(f"No videos found matching '{item}'")

    return filtered_videos, warnings
