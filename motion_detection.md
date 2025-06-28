# Motion Detection Pre-Processing for Wildlife Video Analysis

## Overview
Implement motion detection as a pre-processing step to identify regions of interest before running expensive ML models. This approach leverages the fact that wildlife camera videos are already motion-triggered, allowing us to focus computational resources on areas with actual movement.

## Core Concept
```
Current: Extract 20 frames → Run 5 ML models on full frames → Detect animals
Proposed: Extract frames → Motion detection → Crop motion regions → Run 5 ML models on crops → Detect animals
```

## Technical Implementation

### 1. Motion Detection Pipeline

#### Background Subtraction Methods
```python
# Option A: MOG2 (Mixture of Gaussians)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    detectShadows=True,
    varThreshold=16,  # Lower = more sensitive
    history=20        # Frames to build background model
)

# Option B: KNN (K-Nearest Neighbors) 
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    detectShadows=True,
    dist2Threshold=400,  # Distance threshold
    history=20
)

# Option C: Frame Differencing (lightweight)
def frame_difference(frame1, frame2, threshold=30):
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, motion_mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    return motion_mask
```

#### Motion Region Extraction
```python
def extract_motion_regions(motion_mask, min_area=500, max_area=100000):
    # Find contours in motion mask
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Expand box for context (20% padding)
            padding = 0.2
            x_pad = int(w * padding)
            y_pad = int(h * padding)
            
            motion_boxes.append({
                'bbox': [x-x_pad, y-y_pad, x+w+x_pad, y+h+y_pad],
                'area': area,
                'confidence': area / (w * h)  # Fill ratio
            })
    
    return motion_boxes
```

### 2. Intelligent Filtering

#### Size-Based Filtering
```python
# Costa Rica wildlife size categories
SIZE_FILTERS = {
    'insects': (50, 200),        # Too small - likely insects/debris
    'small_birds': (200, 1000),  # Small birds, rodents
    'medium_animals': (1000, 10000),  # Most Costa Rican mammals
    'large_animals': (10000, 50000),  # Jaguars, tapirs
    'camera_handling': (50000, float('inf'))  # Likely equipment manipulation
}
```

#### Temporal Consistency
```python
def track_motion_across_frames(motion_history, min_persistence=3):
    """Filter motion regions that appear consistently across frames"""
    consistent_regions = []
    
    for region in motion_history[-1]:  # Current frame regions
        persistence_count = 0
        
        # Check how many recent frames contained similar motion
        for past_frame in motion_history[-min_persistence:]:
            if has_overlapping_motion(region, past_frame):
                persistence_count += 1
                
        if persistence_count >= min_persistence:
            consistent_regions.append(region)
            
    return consistent_regions
```

#### Environmental Noise Filtering
```python
def filter_vegetation_motion(motion_regions, frame_shape):
    """Filter out likely vegetation movement"""
    filtered_regions = []
    
    for region in motion_regions:
        x, y, w, h = region['bbox']
        
        # Skip regions that are:
        # 1. Very wide and thin (likely vegetation sway)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            continue
            
        # 2. Located in typical vegetation zones (top 30% of frame)
        if y < frame_shape[0] * 0.3:
            continue
            
        # 3. Have very low density (sparse motion)
        if region['confidence'] < 0.3:
            continue
            
        filtered_regions.append(region)
        
    return filtered_regions
```

### 3. Integration with Existing Pipeline

#### Modified Frame Analysis
```python
def enhanced_motion_frame_analysis(self, frame, frame_idx, video_debug_dir, timestamp_seconds=None):
    """Enhanced frame analysis with motion pre-processing"""
    
    # Step 1: Detect motion regions
    motion_regions = self.detect_motion_regions(frame, frame_idx)
    
    if not motion_regions:
        # No significant motion - skip expensive ML processing
        analysis_logger.info(f"Frame {frame_idx}: No significant motion detected")
        return []
    
    detections = []
    
    # Step 2: Process each motion region with ML models
    for i, region in enumerate(motion_regions):
        analysis_logger.info(f"Frame {frame_idx}: Processing motion region {i+1}/{len(motion_regions)}")
        
        # Crop region from frame
        x1, y1, x2, y2 = region['bbox']
        cropped_frame = frame[y1:y2, x1:x2]
        
        if cropped_frame.size == 0:
            continue
            
        # Run ML ensemble on cropped region
        region_detections = self.run_ml_ensemble_on_crop(cropped_frame, region['bbox'])
        
        # Scale detections back to original frame coordinates
        scaled_detections = self.scale_detections_to_original(region_detections, region['bbox'])
        detections.extend(scaled_detections)
    
    return detections
```

## Configuration Parameters

### Motion Detection Settings
```python
MOTION_DETECTION_CONFIG = {
    # Background subtraction
    'method': 'MOG2',  # 'MOG2', 'KNN', or 'frame_diff'
    'var_threshold': 16,  # MOG2 sensitivity
    'history': 20,        # Frames for background model
    'detect_shadows': True,
    
    # Region filtering
    'min_motion_area': 500,      # Minimum pixels for motion region
    'max_motion_area': 100000,   # Maximum pixels (camera handling)
    'bbox_padding': 0.2,         # Expand regions by 20%
    'min_fill_ratio': 0.3,       # Motion density within bbox
    
    # Temporal filtering
    'min_persistence': 3,        # Frames motion must persist
    'motion_history_length': 5,  # Frames to track
    
    # Environmental filtering
    'max_aspect_ratio': 5.0,     # Filter vegetation sway
    'vegetation_zone_height': 0.3,  # Top 30% likely vegetation
    'min_region_confidence': 0.3,
}
```

## Expected Performance Improvements

### Processing Speed
- **Current**: 5 models × 20 frames × full resolution = 100 ML inferences per video
- **With Motion**: 5 models × ~3 motion regions × ~10 frames = ~15 ML inferences per video
- **Speedup**: ~6-7x faster processing

### Accuracy Improvements
- **Reduced False Positives**: Models won't detect static objects that resemble animals
- **Better Focus**: Tighter crops provide more detailed animal features for classification
- **Context Preservation**: Padding ensures animals aren't cut off at region edges

### Resource Optimization
- **Memory**: Smaller image regions reduce GPU memory usage
- **CPU**: Less data processing and transfer
- **I/O**: Fewer debug frames to save

## Implementation Strategy

### Phase 1: Basic Motion Detection
1. Implement MOG2 background subtraction
2. Extract motion regions with size filtering
3. Run existing ML pipeline on cropped regions
4. Compare results with current full-frame approach

### Phase 2: Intelligent Filtering
1. Add temporal consistency tracking
2. Implement vegetation motion filtering
3. Tune parameters based on Costa Rica footage
4. Add motion region quality scoring

### Phase 3: Optimization
1. Adaptive parameter tuning per video
2. Multi-scale motion detection
3. Region merging for nearby motion areas
4. Performance benchmarking and validation

## Validation Approach

### Test Cases
1. **Known Animal Videos** (7, 8, 9, 11, 12): Ensure motion detection captures all animals
2. **False Positive Videos** (1-6): Verify motion filtering reduces false detections
3. **Camera Handling Videos** (13-19): Confirm large motion regions are properly classified

### Success Metrics
- **Detection Rate**: % of known animals successfully detected via motion
- **False Positive Reduction**: Decrease in non-animal detections
- **Processing Speed**: Time reduction vs current approach
- **Resource Usage**: Memory and CPU utilization improvements

## Potential Challenges & Solutions

### Challenge: Small/Slow Animals
**Solution**: Multi-resolution motion detection with very low thresholds for small regions

### Challenge: Wind-Induced Vegetation Motion
**Solution**: Temporal filtering + aspect ratio filtering + zone masking

### Challenge: Lighting Changes
**Solution**: Shadow detection in background subtraction + adaptive thresholds

### Challenge: Camera Vibration
**Solution**: Global motion compensation before background subtraction

## Integration Points

### With Existing Pipeline
- Drop-in replacement for `enhanced_frame_analysis()`
- Maintains same output format for ensemble validation
- Preserves debug frame saving for motion regions
- Compatible with existing confidence thresholds

### With Future BioCLIP Integration
- Motion regions provide perfect ROI crops for species classification
- Reduced processing means BioCLIP can run on more candidates
- Better animal crops should improve species identification accuracy

This motion detection preprocessing should significantly improve both performance and accuracy of the wildlife detection pipeline while maintaining compatibility with existing validation and clustering systems.