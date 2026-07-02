"""
Constants for ML Detection Ensemble.
"""

# Model detection threshold - minimal value to see ALL detections before ensemble filtering
# The ensemble applies the actual confidence threshold (config.confidence_threshold) later
# This allows the models to detect everything possible, then filtering happens at ensemble level
MODEL_DETECTION_THRESHOLD = 0.001
