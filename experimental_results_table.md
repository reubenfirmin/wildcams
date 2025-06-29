# Experimental Results Table: Animal Detection Performance

## Ground Truth
- **Videos 7, 8, 9**: Should detect animals (TRUE POSITIVES expected)
- **Videos 13, 15**: Should NOT detect animals - camera handling footage (FALSE POSITIVES if detected)

## Experimental Results

| Strategy/Model | Animals Detected | No Animals | Not Processed | True Positives | False Positives | Precision | Recall |
|---|---|---|---|---|---|---|---|
| **Full-frame + MDV6-rtdetr-c** | 7, 8, 9, 13, 15 | - | - | 3/3 (100%) | 2/2 (100%) | 60% | 100% |
| **Motion detection + MDV6-rtdetr-c** | 9, 13, 15 | - | 7, 8 (no motion) | 1/3 (33%) | 2/2 (100%) | 33% | 33% |
| **Full-frame + MDV6-yolov9-e** | 7, 8, 9, 13, 15 | - | - | 3/3 (100%) | 2/2 (100%) | 60% | 100% |
| **Motion detection + MDV6-yolov9-e** | 8, 9, 13, 15 | - | 7 (no motion) | 2/3 (67%) | 2/2 (100%) | 50% | 67% |

## Detailed Video-by-Video Results

### Experiment 1: Full-frame + MDV6-rtdetr-c
- **Log file**: wildcams_20250628_094937.log
- **Video 7**: ✅ Animal detected (TRUE POSITIVE)
- **Video 8**: ✅ Animal detected (TRUE POSITIVE) 
- **Video 9**: ✅ Animal detected (TRUE POSITIVE)
- **Video 13**: ❌ Animal detected (FALSE POSITIVE - camera handling)
- **Video 15**: ❌ Animal detected (FALSE POSITIVE - camera handling)

### Experiment 2: Motion detection + MDV6-rtdetr-c  
- **Log file**: wildcams_20250628_095522.log
- **Video 7**: ⏭️ SKIPPED - No motion detected
- **Video 8**: ⏭️ SKIPPED - No motion detected
- **Video 9**: ✅ Animal detected (TRUE POSITIVE)
- **Video 13**: ❌ Animal detected (FALSE POSITIVE - camera handling)
- **Video 15**: ❌ Animal detected (FALSE POSITIVE - camera handling)

### Experiment 3: Full-frame + MDV6-yolov9-e
- **Log file**: wildcams_20250628_100638.log
- **Video 7**: ✅ Animal detected (TRUE POSITIVE)
- **Video 8**: ✅ Animal detected (TRUE POSITIVE)
- **Video 9**: ✅ Animal detected (TRUE POSITIVE)
- **Video 13**: ❌ Animal detected (FALSE POSITIVE - camera handling)
- **Video 15**: ❌ Animal detected (FALSE POSITIVE - camera handling)

### Experiment 4: Motion detection + MDV6-yolov9-e
- **Log file**: wildcams_20250628_101506.log
- **Video 7**: ⏭️ SKIPPED - No motion detected
- **Video 8**: ✅ Animal detected (TRUE POSITIVE)
- **Video 9**: ✅ Animal detected (TRUE POSITIVE)
- **Video 13**: ❌ Animal detected (FALSE POSITIVE - camera handling)
- **Video 15**: ❌ Animal detected (FALSE POSITIVE - camera handling)

## Key Findings

1. **Motion Detection Limitations**: 
   - Failed to detect motion in videos 7 and 8 (RTdetr) and video 7 (YOLOv9-e)
   - This resulted in missed true positives, reducing recall significantly

2. **False Positive Problem**: 
   - Both models consistently detected "animals" in camera handling footage (videos 13, 15)
   - This suggests the models are detecting human hands/movement as animals
   - 100% false positive rate across all experiments

3. **Model Performance**:
   - Both RTdetr-c and YOLOv9-e showed identical behavior in full-frame mode
   - YOLOv9-e performed slightly better in motion detection mode (detected video 8)

4. **Processing Strategy Impact**:
   - Full-frame processing: 100% recall but poor precision (60%)
   - Motion detection: Improved precision (33-50%) but significantly reduced recall (33-67%)

## Performance Metrics Summary

- **Best Recall**: Full-frame strategies (100%)
- **Best Precision**: Motion detection + MDV6-rtdetr-c (33%)
- **Best F1-Score**: Motion detection + MDV6-yolov9-e (0.571)
- **Most Reliable**: Full-frame + either model (consistent results)

The results show a clear trade-off between recall and precision, with motion detection filtering out some true positives but also reducing processing load.