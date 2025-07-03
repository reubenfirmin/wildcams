# Frame Sampling Robustness Analysis and Refactoring Plan

## Current Implementation Issues

The current frame sampling approach in `_sample_motion_track_frames()` has several robustness problems:

### 1. Motion-Only Sampling Limitation
- **Problem**: Only samples from `motion_detected=True` frames, ignoring backfill/forward-fill regions
- **Impact**: Misses potential detections in extended track regions where animals might be present
- **Code**: `motion_detections = [det for det in bbox_track['detections'] if det['motion_detected']]`

### 2. Limited Coverage
- **Problem**: Max 5 frames per track regardless of track duration or complexity
- **Impact**: Long tracks or tracks with sparse motion get inadequate sampling
- **Code**: `max_frames=config.max_validation_frames` (default 5)

### 3. Even Spacing Assumption
- **Problem**: `step = len(frames) / max_frames` assumes uniform motion distribution
- **Impact**: Misses animal behavior patterns (burst activity, resting periods)
- **Reality**: Animals often have clustered activity followed by quiet periods

### 4. No Temporal Clustering
- **Problem**: Doesn't account for temporal clustering of detections
- **Impact**: Might sample redundant nearby frames instead of spreading across track duration
- **Missing**: Minimum temporal gap between sampled frames

### 5. No Quality-Based Sampling
- **Problem**: All motion frames treated equally regardless of motion strength
- **Impact**: Might sample low-quality motion over high-quality detections
- **Missing**: Motion region size, confidence, or other quality metrics

## Proposed Robust Approach: Temporal Clustering with Strategic Fills

```python
def _sample_motion_track_frames(self, track: Dict, max_frames: int = 5) -> List[int]:
    """Sample representative frames with temporal spread and quality consideration."""
    bbox_track = track.get('bbox_track', track)
    
    if 'detections' not in bbox_track:
        return track.get('frames', [])
    
    # Get all track frames (motion + fills) for comprehensive coverage
    all_detections = bbox_track['detections']
    motion_detections = [det for det in all_detections if det['motion_detected']]
    
    # If very few motion frames, include some fills for better coverage
    if len(motion_detections) < max_frames:
        fill_detections = [det for det in all_detections if not det['motion_detected']]
        # Add strategic fills (e.g., start/middle/end of track)
        if fill_detections:
            track_duration = len(all_detections)
            strategic_fills = [
                fill_detections[0],  # Start
                fill_detections[track_duration // 2] if track_duration > 2 else None,  # Middle
                fill_detections[-1]  # End
            ]
            motion_detections.extend([f for f in strategic_fills if f])
    
    frames = [det['frame'] for det in motion_detections]
    frames = sorted(list(set(frames)))  # Remove duplicates and sort
    
    if len(frames) <= max_frames:
        return frames
    
    # Temporal clustering to ensure spread across track duration
    min_gap = config.temporal_spread_seconds * fps  # Convert to frames
    selected = [frames[0]]  # Always include first
    
    for frame in frames[1:]:
        if len(selected) >= max_frames:
            break
        # Only add if sufficiently separated from last selected
        if frame - selected[-1] >= min_gap:
            selected.append(frame)
    
    # If we still have space and didn't get enough spread, fill in gaps
    while len(selected) < max_frames and len(selected) < len(frames):
        # Find largest gap and add frame from it
        largest_gap_idx = 0
        largest_gap = 0
        for i in range(len(selected) - 1):
            gap = selected[i+1] - selected[i]
            if gap > largest_gap:
                largest_gap = gap
                largest_gap_idx = i
        
        # Add frame from middle of largest gap
        gap_start = selected[largest_gap_idx]
        gap_end = selected[largest_gap_idx + 1]
        middle_frame = (gap_start + gap_end) // 2
        
        # Find closest actual frame to middle
        closest_frame = min([f for f in frames if gap_start < f < gap_end], 
                          key=lambda x: abs(x - middle_frame))
        
        selected.insert(largest_gap_idx + 1, closest_frame)
        selected.sort()
    
    return selected[:max_frames]
```

## Alternative: Density-Based Sampling

```python
def _sample_adaptive_frames(self, track: Dict, max_frames: int = 5) -> List[int]:
    """Adaptive sampling based on motion density and temporal requirements."""
    motion_frames = [det['frame'] for det in track['bbox_track']['detections'] 
                    if det['motion_detected']]
    
    if len(motion_frames) <= max_frames:
        return motion_frames
    
    # Group frames into temporal clusters
    clusters = []
    current_cluster = [motion_frames[0]]
    
    for frame in motion_frames[1:]:
        if frame - current_cluster[-1] <= config.temporal_spread_seconds * fps:
            current_cluster.append(frame)
        else:
            clusters.append(current_cluster)
            current_cluster = [frame]
    clusters.append(current_cluster)
    
    # Sample from each cluster proportionally
    samples = []
    frames_per_cluster = max(1, max_frames // len(clusters))
    
    for cluster in clusters:
        if len(cluster) <= frames_per_cluster:
            samples.extend(cluster)
        else:
            # Sample evenly from cluster
            step = len(cluster) / frames_per_cluster
            samples.extend([cluster[int(i * step)] for i in range(frames_per_cluster)])
    
    return sorted(samples[:max_frames])
```

## Implementation Plan

### Phase 1: Temporal Clustering (Recommended)
1. **Replace current even-spacing algorithm** with temporal clustering approach
2. **Add strategic fill sampling** when motion frames are insufficient
3. **Implement minimum temporal gap** using `config.temporal_spread_seconds`
4. **Add gap-filling logic** to ensure good temporal coverage

### Phase 2: Quality-Based Enhancement
1. **Add motion quality scoring** based on region size and motion strength
2. **Prioritize high-quality frames** within temporal clusters
3. **Weight sampling** by motion confidence or region density

### Phase 3: Adaptive Parameters
1. **Dynamic max_frames** based on track duration and complexity
2. **Variable temporal gaps** based on animal behavior patterns
3. **Context-aware sampling** for different track types (short vs long, dense vs sparse)

## Expected Benefits

1. **Better Temporal Coverage**: Ensures samples spread across entire track duration
2. **Strategic Fill Usage**: Includes backfill/forward-fill when motion is sparse
3. **Reduced Redundancy**: Minimum temporal gaps prevent sampling clustered frames
4. **Improved Robustness**: Handles edge cases like very short or very long tracks
5. **Maintained Efficiency**: Still limits total frames processed while improving quality

## Configuration Parameters

```python
# Add to CLI parameters
--temporal-spread-seconds 2.0      # Minimum time gap between sampled frames
--max-validation-frames 5          # Maximum frames per track (existing)
--min-fill-coverage 0.3            # Minimum fill coverage when motion is sparse
--adaptive-sampling True           # Enable density-based clustering
```

## Testing Strategy

1. **Test on sparse motion tracks** (Videos 10-11) to verify fill inclusion
2. **Test on dense motion tracks** to verify temporal clustering
3. **Compare detection rates** between current and new sampling approaches
4. **Measure processing time impact** of improved sampling
5. **Validate temporal spread** ensures good coverage across track duration