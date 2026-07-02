"""Analysis result writing operations for wildlife video processing."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AnalysisWriter:
    """Handles writing analysis results to various file formats."""

    def __init__(self, output_dir: Path):
        """Initialize analysis writer with output directory."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def save_analysis(self, video_path: Path, analysis_data: dict[str, Any]) -> None:
        """Save analysis results to JSON file."""
        try:
            # Convert data for JSON serialization
            json_data = self._convert_for_json(analysis_data)

            # Generate output filename
            output_filename = f"{video_path.stem}_analysis.json"
            output_path = self.output_dir / output_filename

            # Write to file
            with open(output_path, "w") as f:
                json.dump(json_data, f, indent=2)

            logger.info(f"💾 Analysis saved to: {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save analysis for {video_path.name}: {e}")

    def generate_summary_report(self, all_results: list[dict]) -> None:
        """Generate and save comprehensive summary report."""
        try:
            total_videos = len(all_results)
            animals_detected = sum(1 for r in all_results if r.get("has_animals", False))
            total_detections = sum(len(r.get("detections", [])) for r in all_results)

            # Calculate model performance statistics
            model_stats = {}
            for result in all_results:
                for detection in result.get("detections", []):
                    model = detection.get("model", "unknown")
                    if model not in model_stats:
                        model_stats[model] = {"count": 0, "avg_confidence": 0.0}
                    model_stats[model]["count"] += 1
                    model_stats[model]["avg_confidence"] += detection.get("confidence", 0.0)

            # Calculate average confidences
            for model in model_stats:
                if model_stats[model]["count"] > 0:
                    model_stats[model]["avg_confidence"] /= model_stats[model]["count"]

            # Create summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "processing_summary": {
                    "total_videos_processed": total_videos,
                    "videos_with_animals": animals_detected,
                    "detection_rate": animals_detected / total_videos if total_videos > 0 else 0.0,
                    "total_detections": total_detections,
                    "avg_detections_per_video": total_detections / total_videos if total_videos > 0 else 0.0,
                },
                "model_performance": model_stats,
                "video_results": all_results,
            }

            # Save summary
            summary_path = self.output_dir / "processing_summary.json"
            with open(summary_path, "w") as f:
                json.dump(self._convert_for_json(summary), f, indent=2)

            logger.info(f"📋 Processing summary saved to: {summary_path}")
            logger.info(
                f"📊 Processed {total_videos} videos, found animals in {animals_detected} ({animals_detected / total_videos * 100:.1f}%)"
            )

        except Exception as e:
            logger.error(f"❌ Failed to generate summary report: {e}")

    def _convert_for_json(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_for_json(item) for item in obj)
        else:
            return obj

    def _debug_numpy_objects(self, obj: Any, path: str = "root") -> None:
        """Debug helper to find numpy objects in nested data structures."""
        if isinstance(obj, (np.ndarray, np.floating, np.integer, np.bool_)):
            logger.debug(f"Found numpy object at {path}: {type(obj)}")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                self._debug_numpy_objects(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._debug_numpy_objects(item, f"{path}[{i}]")
