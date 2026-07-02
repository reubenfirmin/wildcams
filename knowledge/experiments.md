# Motion Detection Parameter Experiments

## SCIENTIFIC METHOD NOTE FOR CLAUDE
When documenting experiments:
1. **ALWAYS check prior results** before claiming regressions or improvements
2. **Compare to baseline** - what was the previous state for each video?
3. **Be precise** - distinguish between "new false positives" vs "existing false positives"
4. **Verify ground truth** - check what each video actually contains before labeling outcomes
5. **Document changes accurately** - only report actual differences from previous runs

## Full Run: 2025-07-01 19:09 (Log: wildcams_20250701_190900.log)

**Parameters:**
```
--min-motion-area 100
--motion-var-threshold 50  
--min-track-duration 0.1
--composite-motion-threshold 3000000
--ensemble yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l
--conf 0.25
```

**Results:**
- **Successful (6):** IMG_0002, IMG_0007, IMG_0008, IMG_0009, IMG_0012, IMG_0018
- **Failed insufficient motion (4):** IMG_0003, IMG_0005, IMG_0010, IMG_0011  
- **Failed camera handling (9):** IMG_0013-0017, IMG_0019
- **Failed no motion (6):** IMG_0001, IMG_0004, IMG_0006, IMG_0020

## Problem
Videos 10-11 failing motion filter (score=88 < 100) when they previously passed (scores 730, 4619) and contain obvious animals.

**Previous successful (17:35):** min-motion-area=300, var-threshold=32
**Current failed (19:09):** min-motion-area=100, var-threshold=50

## Test: 2025-07-01 20:14 (Log: wildcams_20250701_201442.log)

**Parameters:**
```
-v 10 11 -e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.25 --min-motion-area 300 --motion-var-threshold 32 --min-track-duration 0.1 --accepted-rtdetr-overlap 0.1
```

**Results:**
- **IMG_0010:** Motion passed (score=730), failed full-frame analysis
- **IMG_0011:** Motion passed (score=4619), animals detected (conf=0.369)

**Analysis:** Motion parameters fixed but IMG_0010 now fails at Step 4 full-frame validation instead of Step 2 motion filter.

## Test (Bug Fix): 2025-07-01 20:23 (Log: wildcams_20250701_202357.log)

**Parameters:** Full run (all 20 videos)
```
-e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.25 --min-motion-area 300 --motion-var-threshold 32 --min-track-duration 0.1 --accepted-rtdetr-overlap 0.1
```

**Results:**
- **Successful (9):** IMG_0002, IMG_0007, IMG_0008, IMG_0009, IMG_0011, IMG_0012, IMG_0018, plus 2 others
- **IMG_0010:** Motion passed (score=730), but failed Step 3 full-frame analysis 
- **IMG_0011:** Motion passed (score=4619), animals detected (conf=0.369) ✅
- **NEW FALSE POSITIVES:** IMG_0003, IMG_0004 now incorrectly detected (were previously filtered)

**Analysis:** Partial regression - Motion parameters fixed videos 10-11 but introduced 2 new false positives. IMG_0003 and IMG_0004 (known negatives) now pass motion detection when they were previously filtered by insufficient motion/no motion. IMG_0002 was already being detected in previous runs.

## Test: 2025-07-01 20:49 (Log: wildcams_20250701_204932.log)
**Parameters:** Full run, variance-threshold=40 
**Result:** Both videos 10,11 now fail motion detection (insufficient motion). Intermediate threshold too restrictive.

## Test: 2025-07-02 (Scientific Logging + Temporal Continuity) (Log: wildcams_20250702_063215.log)

**Parameters:**
```bash
process -v 4 8 -e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.40 --min-motion-area 300 --filter-motion-var-threshold 32 --analysis-motion-var-threshold 45 --min-track-duration 0.1 --accepted-rtdetr-overlap 0.1
```

**Changes:**
- Implemented scientific logging format (EVAL/ENSEMBLE/TRACK headers with ✅/❌ icons)
- Added temporal continuity validation using existing `--max-skip-frames 3` parameter
- Track validation now requires: confidence + min_frames + temporal_continuity

**Results:**
- **IMG_0004:** ❌ Failed (false positive correctly filtered)
- **IMG_0008:** ❌ Failed - `temporal_pass=false` (failed temporal continuity check)

**Analysis:** 
- IMG_0004: Correctly failed (known false positive)
- IMG_0008: Failed due to new temporal continuity validation - gaps between validated detections exceeded `max_skip_frames=3`
- **Issue:** IMG_0008 is a true positive (contains animals) but failing temporal continuity. May need to adjust `--max-skip-frames` or frame sampling strategy.

**Next Step:** Need to investigate why IMG_0008 true positive is failing temporal continuity validation. Options:
1. Increase `--max-skip-frames` for more lenient temporal requirements
2. Adjust frame sampling strategy to get more contiguous detections
3. Review if temporal continuity requirement is too strict for real animal behavior

## Test: 2025-07-02 09:11 (Fixed Ensemble + Split Temporal Parameters)

**Parameters:**
```bash
process -v 4 8 -e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.40 --min-motion-area 300 --filter-motion-var-threshold 32 --analysis-motion-var-threshold 45 --min-track-duration 0.1 --accepted-rtdetr-overlap 0.1 --detection-validation-gap-seconds 0.2
```

**Changes:**
1. **Fixed temporal continuity bug**: Removed incorrect "+1" in gap validation logic
2. **Fixed ensemble scoring**: Replaced complex confidence-weighted tiers with simple normalized sum (favors multiple models vs single model)
3. **Split temporal parameters**: 
   - `--motion-tracking-gap-seconds 1.0` (default, for Step 1 motion linking - lenient)
   - `--detection-validation-gap-seconds 0.2` (for Step 3 ML validation - strict)
4. **Added comprehensive parameter logging**: All CLI arguments now logged at start

**Expected Results:**
- IMG_0004: Should FAIL temporal validation (0.24s gap between detections > 0.2s threshold)
- IMG_0008: Should PASS temporal validation (tighter detection clustering)

**Actual Results:**
- ❌ **IMG_0004:** PASSED (`temporal_pass=True`) - **BUG FOUND!**
  - Valid detections at 0.23s and 0.47s (gap = 0.24s > 0.2s threshold)
  - Should have failed temporal validation but didn't
- ✅ **IMG_0008:** PASSED (`temporal_pass=True`) - Expected behavior

**Actual Results (FIXED):**
- ✅ **IMG_0004:** FAILED (`temporal_pass=False`) - **BUG FIXED!**
  - Gap = 0.233s > 0.200s threshold correctly triggered failure
  - Debug: `TEMPORAL_DEBUG | FAILED: gap 0.233s > threshold 0.200s`
  - Result: `processing_status": "no_animals"` ✅
- ✅ **IMG_0008:** PASSED (`temporal_pass=True`) - Expected behavior
  - Gap = 0.167s < 0.200s threshold passed validation
  - Result: Animals detected ✅

**Bug Fixes Applied:**
1. **Missing timestamp**: Added `det['timestamp'] = timestamp` to detection dictionaries
2. **Parameter logging**: Moved logging after processor initialization 
3. **Debug logging**: Added TEMPORAL_DEBUG to show actual gaps vs thresholds

**Analysis:** Temporal continuity validation now working correctly. The false positive IMG_0004 is properly filtered while true positive IMG_0008 passes validation. Split temporal parameters provide appropriate strictness levels for motion tracking vs detection validation.

## Test: 2025-07-02 09:30 (Full Batch - Fixed Temporal Validation)

**Parameters:**
```bash
process -e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.40 --min-motion-area 300 --filter-motion-var-threshold 32 --analysis-motion-var-threshold 45 --min-track-duration 0.1 --accepted-rtdetr-overlap 0.1 --detection-validation-gap-seconds 0.2
```

**Results:**

**✅ Successful (5):** IMG_0007, IMG_0008, IMG_0009, IMG_0012, IMG_0018
- All have strong ensemble scores and temporal consistency
- Good mix of single-track (IMG_0008) and multi-track (IMG_0007, IMG_0012) detections

**❌ Failed Motion Filter (8):**
- **Insufficient motion (3):** IMG_0001, IMG_0005, IMG_0006 (score=88 < 100)
- **Camera handling (5):** IMG_0013, IMG_0014, IMG_0015, IMG_0016, IMG_0017, IMG_0019 (score > 3,000,000)

**❌ Failed ML Validation (7):** IMG_0002, IMG_0003, IMG_0004, IMG_0010, IMG_0011, IMG_0020
- Motion passed but no consistent animal detections in full-frame analysis
- IMG_0004 now correctly filtered by temporal validation (gap=0.233s > 0.2s)

**Analysis:** 
- **Great improvement**: IMG_0004 false positive now correctly filtered by temporal validation
- **Known issues**: IMG_0010, IMG_0011 (true positives) still failing Step 3/4 validation
- **Ground truth verification needed**: Videos 2, 3, 20 status unclear

**Individual Problems to Address:**
1. **IMG_0010, IMG_0011**: True positives failing ML validation - investigate Step 3/4 scoring
2. **Videos 2, 3, 20**: Determine ground truth status (animals vs false positives)
3. **Motion threshold tuning**: Consider adjusting insufficient motion threshold (currently 100)

## Test: 2025-07-02 19:17 (Frame-First Algorithm + Coherent Object Tracking + Bug Fixes)

**Rationale:** 
Major architectural refactoring to implement coherent object tracking with full video coverage and frame-first processing algorithm. Previous experiments showed IMG_0010 and IMG_0011 failing Step 3/4 validation despite having good motion detection. This experiment tests the new architecture with proper spatial overlap validation and track extension.

**Parameters:**
```bash
process -v 10 11 -e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.4 --min-motion-area 300 --motion-var-threshold 32 --min-track-duration 0.1 --spatial-overlap-threshold 0.1
```

**Major Changes:**
1. **Frame-First Algorithm**: Refactored Step 3 to process frames first, then models, avoiding duplicate ML runs
2. **Coherent Object Tracking**: Replaced motion sequences with extended bbox tracks covering full video duration
3. **Backfill/Forward-Fill**: Tracks extended to video start/end using first/last known positions for spatial validation
4. **Spatial Overlap Validation**: Enhanced validation with explicit vs implicit overlap logging  
5. **Bug Fixes**: Fixed tuple-to-list conversion errors in motion region handling
6. **Parameter Renaming**: `--accepted-rtdetr-overlap` → `--spatial-overlap-threshold`
7. **Enhanced Stacktraces**: Added full error logging for better debugging

**Results:**
- **✅ IMG_0010:** **SUCCESS** - Motion passed (score=730), animals detected (conf=0.378, combined=13.149)
- **✅ IMG_0011:** **SUCCESS** - Motion passed (score=4619), animals detected (conf=0.369, combined=26.849)

**Performance:**
- IMG_0010: 44.5s processing time
- IMG_0011: 43.6s processing time  
- Average: 44.1s per video

**Model Contributions:**
- **RT-DETR-L**: 6000 detections, max_conf=0.469 (primary detector)
- **YOLO12M**: 14 detections, max_conf=0.020 (minimal contribution)
- **YOLO12X**: 14 detections, max_conf=0.037 (minimal contribution) 
- **MDV6-YOLOv10-E**: 0 detections (no contribution)

**Analysis:**
🎯 **MAJOR SUCCESS**: Both previously failing videos (IMG_0010, IMG_0011) now successfully detect animals with the new coherent object tracking architecture. The frame-first algorithm and extended track structure resolved the spatial validation issues.

**Key Insights:**
1. **RT-DETR Dominance**: RT-DETR-L is the primary performing model (6000 detections vs 14 each for YOLO models)
2. **Spatial Overlap Works**: New spatial validation correctly validates detections against motion regions
3. **Extended Tracks**: Backfill/forward-fill provides full video coverage for spatial validation
4. **Architecture Fixed**: Previous Step 3/4 validation failures resolved by coherent tracking approach

**Next Steps:**
1. Test full batch (all 20 videos) with new architecture
2. Evaluate if ensemble can be simplified (RT-DETR + 1-2 YOLO models)
3. Investigate MegaDetector v6 low contribution (0 detections)
4. Optimize confidence thresholds based on model performance distribution

## Test: 2025-07-03 05:26 (Inverted Camera Handling Detection) (Log: wildcams_20250703_052556.log)

**Change:** Inverted camera handling logic: spatial_dispersion × motion_sparsity (higher = camera handling). `--composite-motion-threshold` default: 3,000,000 → 8.0.

**Results:**
- **✅ IMG_0007 (bird):** Score=3.35 → PASSED to full analysis
- **❌ IMG_0018 (camera handling):** Score=10.83 → BLOCKED as camera handling

**Analysis:** Camera handling detection fixed. Video 18 scored 3.2x higher than Video 7, correctly identifying dispersed camera movement vs concentrated animal movement.

## Test: 2025-07-03 05:33 (Full 20-Video Test - Fixed Camera Handling) (Log: wildcams_20250703_053345.log)

**Change:** Full test of all 20 videos with fixed camera handling detection (threshold 8.0).

**Results:**
- **✅ Animals Detected (11):** IMG_0004, IMG_0007, IMG_0008, IMG_0010, IMG_0011, IMG_0013, IMG_0014, IMG_0015, IMG_0016, IMG_0017, IMG_0019
- **❌ No Animals (9):** IMG_0001, IMG_0002, IMG_0003, IMG_0005, IMG_0006, IMG_0009, IMG_0012, IMG_0018, IMG_0020

**Analysis:** Camera handling detection broken - videos 13-19 (camera handling) are incorrectly detecting animals instead of being filtered. Only IMG_0018 correctly filtered. Major regression from camera handling fix.

## Test: 2025-07-03 07:26 (Frame Coverage + Fixed Ensemble Scoring) (Log: wildcams_20250703_072622.log)

**Change:** Fixed camera handling detection (frame coverage analysis) + fixed confidence=1.0 bug (proper ensemble scoring). `--composite-motion-threshold` default: 8.0 → 0.5.

**Results:**
- **✅ Animals Detected (11):** IMG_0002, IMG_0003, IMG_0004, IMG_0007, IMG_0008, IMG_0009, IMG_0010, IMG_0011, IMG_0012, IMG_0018, IMG_0019
- **❌ No Animals (9):** IMG_0001, IMG_0005, IMG_0006, IMG_0013, IMG_0014, IMG_0015, IMG_0016, IMG_0017, IMG_0020

**Analysis:** Major improvement on camera handling detection. Videos 13-17 correctly filtered (vs previously passing). IMG_0018 still incorrectly detecting animals. Confidence scores now realistic (0.01-0.03 range vs artificial 1.0). Ground truth issues: videos 2,3,4,9,12 detecting animals (should investigate if legitimate).

## Test: 2025-07-03 09:43 (Simple Sum Ensemble Scoring) (Log: wildcams_20250703_094326.log)

**Changes:**
1. **Simple Sum Ensemble**: Replaced complex weight redistribution with `sum(model_contributions.values())`
2. **Higher Confidence Threshold**: `--conf 0.8` to account for new 0.0-4.0 scoring range
3. **Natural Multi-Model Reward**: More models = higher scores inherently

**Parameters:**
```bash
process -e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.80 --min-motion-area 300 --filter-motion-var-threshold 32 --analysis-motion-var-threshold 45 --min-track-duration 0.1 --spatial-overlap-threshold 0.1 --min-consecutive-detection-seconds 0.2 --enable-track-infilling --infill-max-gap-seconds 0.7 --infill-max-distance-pixels 350
```

**Results:**
- **✅ Animals Detected (10):** IMG_0002, IMG_0003, IMG_0004, IMG_0007, IMG_0008, IMG_0009, IMG_0010, IMG_0012, IMG_0018, IMG_0019
- **❌ No Animals (10):** IMG_0001, IMG_0005, IMG_0006, IMG_0011, IMG_0013, IMG_0014, IMG_0015, IMG_0016, IMG_0017, IMG_0020

**Key Findings:**

**✅ Simple Sum Works!**
- **IMG_0008 best frame**: YOLO12X=0.130 + YOLO12M=0.473 + RT-DETR=1.000 = `ensemble_score=1.603` (3 models, ✅ passed)
- **IMG_0004 best frame**: RT-DETR=0.572 only = `ensemble_score=0.572` (1 model, ❌ failed)

**❌ But `--conf` threshold NOT working as expected:**
- **Problem 1**: Videos still showing confidence < 0.8 in final output (conf=0.024 for IMG_0018)
- **Problem 2**: `--conf 0.8` applies at track level, but final video confidence uses different calculation
- **Problem 3**: Track validation passed with `temporal_pass=False` for IMG_0008 track_0, but `temporal_pass=True` for track_3

**Track vs Video Confidence Disconnect:**
- Track ensemble scores: 1.603, 0.970 (correctly high for multi-model)
- Final video confidence: 0.024 (incorrectly low - not using ensemble scores)

**Analysis:** Simple sum ensemble scoring works perfectly at track level - multi-model consensus gets higher scores than single-model detection. However, the final video confidence calculation is still broken and not using the new ensemble scores. The `--conf` threshold applies correctly at track level but final output shows different confidence values.

## Test: 2025-07-03 10:14 (Fixed Video Confidence Calculation) (Log: wildcams_20250703_101456.log)

**Changes:**
1. **Fixed Video Confidence Bug**: Video confidence now uses actual ensemble scores instead of recalculated avg_conf
2. **Added Ensemble Score Storage**: Validated sequences now store `ensemble_score` from track validation
3. **Eliminated avg_conf Mess**: Removed broken confidence recalculation entirely

**Parameters:**
```bash
process -v 4 8 -e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.80 --min-motion-area 300 --filter-motion-var-threshold 32 --analysis-motion-var-threshold 45 --min-track-duration 0.1 --spatial-overlap-threshold 0.1 --min-consecutive-detection-seconds 0.2 --enable-track-infilling --infill-max-gap-seconds 0.7 --infill-max-distance-pixels 350
```

**Results:**
- **✅ IMG_0008**: Animals detected (conf=0.000, combined=1.228)
- **✅ IMG_0004**: Animals detected (conf=0.000, combined=1.584)

**Track-Level Validation (WORKING CORRECTLY):**
- **IMG_0008 best frame**: 3 models = `ensemble_score=1.603` → ✅ passed 0.8 threshold
- **IMG_0004 best frame**: 1 model = `ensemble_score=0.572` → ❌ failed 0.8 threshold

**❌ STILL BROKEN - Video Confidence = 0.000:**
Both videos show `conf=0.000` despite having high ensemble scores. The fix didn't work because:

**Root Cause**: The validated sequence doesn't get the ensemble score stored properly, or `best_sequence.get('ensemble_score', 0.0)` is returning the default 0.0.

**Key Insight**: Track-level filtering with `--conf 0.8` IS working correctly:
- IMG_0008: 1.603 ensemble score passed → video processed
- IMG_0004: All frame ensemble scores failed (0.572 max) → but track still validated via temporal/other criteria

**Next Fix Needed**: Debug why `ensemble_score` isn't being stored in validated sequences or extract it differently.

**Analysis:** The simple sum ensemble scoring works perfectly and `--conf 0.8` filtering works at track level, but video confidence is still broken (showing 0.0). The ensemble score storage/retrieval needs debugging.

## Test: 2025-07-02 19:50 (IoU Overlap Calculation + Frame-First Algorithm) (Log: wildcams_20250702_195044.log)

**Rationale:** 
Major fix to spatial overlap validation using IoU instead of motion containment. Previous experiments showed RT-DETR generating massive false positive detections with 100% "overlap" scores when huge detections contained small motion regions. Implemented frame-first algorithm and switched from motion containment to Intersection over Union (IoU) for more accurate spatial validation.

**Parameters:**
```bash
process -v 10 11 -e yolo12x,yolo12m,MDV6-yolov10-e,rtdetr-l --conf 0.4 --min-motion-area 300 --motion-var-threshold 32 --min-track-duration 0.1 --spatial-overlap-threshold 0.1
```

**Major Changes:**
1. **IoU Overlap Calculation**: Replaced motion containment with Intersection over Union
   - **Old**: `intersection / motion_area` (motion region fully contained = 100%)
   - **New**: `intersection / union_area` (proper IoU scoring)
2. **Frame-First Algorithm**: Implemented correct loop order (frame → model → track)
3. **Spatial Validation Fix**: Large detections now get low IoU scores instead of false 100% matches
4. **Enhanced Logging**: Proper EVAL format with individual detection logging per track

**Results:**
- **✅ IMG_0011:** **SUCCESS** - Motion passed, animals detected
- **❌ IMG_0010:** **FAILED** - No consistent animal movement (Step 2 filter)

**IoU Overlap Improvements:**
- **Previous**: RT-DETR detections showing `ovlp:1.000` for massive frames containing tiny motion regions
- **Current**: Realistic IoU scores: `ovlp:0.244`, `ovlp:0.133`, `ovlp:0.290` for appropriately sized detections
- **False Positive Filtering**: 12 detection groups filtered out for `no_overlap` (huge detections with low IoU)

**Model Contributions:**
- **RT-DETR-L**: 600 detections, max_conf=0.268 (primary detector, but realistic overlap scores)
- **YOLO12M**: 10 detections, max_conf=0.020 (minimal contribution)
- **YOLO12X**: 4 detections, max_conf=0.007 (minimal contribution)
- **MDV6-YOLOv10-E**: 0 detections (no contribution)

**Analysis:**
🎯 **MAJOR IMPROVEMENT**: IoU-based overlap calculation dramatically improved spatial validation accuracy. RT-DETR detections now show realistic overlap scores (0.1-0.3) instead of false 100% matches. The frame-first algorithm successfully processes each frame once across all models, with proper per-track evaluation.

**Key Insights:**
1. **IoU Eliminates False Positives**: Massive frame-spanning detections now correctly score as low overlap
2. **Realistic Spatial Scoring**: Overlap scores now reflect actual geometric relationships
3. **Algorithm Correctness**: Frame-first implementation follows specification exactly
4. **Filtering Effectiveness**: 12 detection groups properly filtered for insufficient overlap

**Before/After Comparison:**
- **Before**: `bbox:204,-3,1254,896` vs `motn:419,735,580,884` = `ovlp:1.000` (FALSE)
- **After**: Similar massive detection would score `ovlp:0.024` (CORRECT IoU)

**Performance:**
- Processing maintained at ~45s per video
- Logging volume significantly reduced due to proper false positive filtering
- Spatial validation now mathematically sound

**Next Steps:**
1. Run full batch test (all 20 videos) with corrected IoU overlap
2. Analyze if `--spatial-overlap-threshold 0.1` is optimal for IoU scoring
3. Investigate why IMG_0010 fails Step 2 motion filter (should be true positive)
4. Consider ensemble simplification given RT-DETR dominance and YOLO low contribution

## Test: 2025-07-02 20:15 (Track-Based Ensemble + Synthetic Detections) (Log: wildcams_20250702_201558.log)

**Changes:**
1. **Combine overlapping detections**: Multiple RT-DETR detections per track → ONE synthetic detection with consensus boosting
2. **Track-based ensemble**: Per-track scoring instead of frame-global 
3. **Fixed ensemble math**: `ensemble = best_per_model / num_models` instead of summing all detections

**Results:**
- IMG_0010: Failed (no change)
- IMG_0011: Success

**Fixed Issues:**
- RT-DETR was logging 14 detections per track, inflating ensemble scores
- Ensemble score was 0.071 (wrong), now 0.020 (correct)
- Each track now evaluated independently

**Problem:** Still using simple averaging instead of dynamic weighting. RT-DETR conf 0.078 → ensemble 0.078/4 = 0.0195 < 0.4 threshold.

## Test: 2025-07-02 20:27 (Full Batch with Fixed Ensemble Weighting) (Log: wildcams_20250702_202756.log)

**Changes:**
1. **Restored confidence-weighted ensemble** (accidentally deleted in refactoring)
2. **Fixed weighting algorithm**: Non-detecting models 10% each, detecting models get 15% floor + confidence-proportional share
3. **Added parameter TODOs**: Hardcoded weights (10%, 15%) need CLI configuration

**Results:** FULL BATCH (all 20 videos)
- **✅ Animals detected: 2 videos** 
- **❌ No animals: 18 videos**

**Critical Issue: 3 False Negatives**
- **Videos 7, 9, 12**: Known true positives failed detection (should contain animals)
- **Video 10**: Marginal false negative (borderline case)
- **Successful videos**: IMG_0008, IMG_0011

**IMG_0007 Failure Analysis:**
- **Motion Detection**: ✅ 11 tracks, 347 regions, 36.7% frames with motion
- **ML Detection**: ✅ 131-149 detections per track, conf up to 1.000
- **Confidence**: ✅ All tracks pass conf_pass=True, frames_pass=True  
- **FAILURE**: ❌ temporal_pass=False on ALL tracks (gaps > 0.2s between detections)
- **Root Cause**: `--detection-validation-gap-seconds 0.2` too strict for real animal movement

**Performance:**
- Total processing time: ~16 minutes (20 videos)
- Average: ~48s per video

**Key insight:** Fixed ensemble weighting but system now failing to detect known animals. Major regression in sensitivity.

## Refactoring Bug Fixes: 2025-07-04 05:39 (Phase 1 Config System)

**Issue:** After Phase 1 refactoring (config system), two critical bugs emerged:

**Problem A: conf=0.000 in final output**
- Videos 9 and 10 showed `conf=0.000` but `combined=1.596/1.358`  
- **Root Cause:** `ensemble_score` field missing from validated sequence structure
- **Fix:** Added proper ensemble score tracking from frame-level evaluation to final output

**Problem B: Video 8 motion filter regression**  
- Video 8 failed motion filter (`score=0.053 > 0`) when it should pass
- **Root Cause:** `composite_motion_threshold` type was `int` instead of `float`, truncating 0.5 → 0
- **Fix:** Changed ProcessingConfig field from `int` to `float`

**Changes Made:**
1. Fixed `composite_motion_threshold: int` → `composite_motion_threshold: float` in ProcessingConfig
2. Removed `int()` conversion in ConfigurationManager 
3. Added `ensemble_score` field to validated track results using actual ensemble algorithm output
4. Tracked best ensemble score per track from frame-level evaluation

**Test Command:** `./process.py -v 8 9 10`

## Step 4: Animal Classification Implementation: 2025-07-08 (Log: wildcams_20250708_051034.log)

**Changes:** Implemented real BioCLIP and DeepFaune models replacing dummy classification
1. **BioCLIP engine**: OpenCLIP model with Costa Rican wildlife classes
2. **DeepFaune engine**: Binary animal vs non-animal classification  
3. **Step 4 integration**: Optional classification filtering post-validation

**Results:**
- **✅ Animals detected (1):** IMG_0007 (woodpecker species identified)
- **❌ No animals (19):** All others filtered by Step 4 classification
- **Filter impact:** 90% reduction from previous 10 videos → 1 video

**Issues:** DeepFaune URL broken (404), using fallback dummy classification

## DeepFaune Model Loading Fix: 2025-07-14 (Log: wildcams_20250714_041733.log)

**Changes:** Fixed DeepFaune model architecture mismatch
1. **Correct model**: `deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt` from `Deepfaune_v1.3`
2. **Proper structure**: Custom `Model` class with `base_model` attribute matching demo code
3. **Fixed preprocessing**: CROP_SIZE=182, proper normalization

**Results:**
- **✅ Animals detected (2):** IMG_0007 (woodpecker, conf=0.746), IMG_0012 (conf=0.998)
- **❌ No animals (18):** All others filtered
- **Performance**: Precision 100%, Recall 40% (2/5 expected animals)

## Ensemble Logic Fix + Raw Model Outputs: 2025-07-14 (Log: wildcams_20250714_205352.log)

**Changes:** 
1. **Fixed ensemble logic**: Replaced weighted averaging with "either model passes" strategy
2. **Raw model outputs**: Exposed 34-species DeepFaune probabilities and BioCLIP top-5 predictions
3. **Enhanced logging**: Added model approval tracking with `approved_by` field

**Results:**
- **✅ Animals detected (4):** IMG_0007, IMG_0009, IMG_0011, IMG_0012 (all approved by DeepFaune)
- **❌ No animals (16):** All others filtered
- **Performance**: Recall improved 40% → 80% (4/5 expected animals)

**Key Findings:**
- DeepFaune proved decisive in all successful cases
- BioCLIP threshold (0.30) may be too high (most predictions 0.08-0.29)
- Models give confident predictions on garbage crops (robustness issue)
- Ensemble fix eliminated major false negative (IMG_0009 with 99.4% DeepFaune confidence)
## Repo Recovery + Guardrails + Analytics Cleanup: 2026-07-01 (no processing run)

**Context:** The `oss/wildcams` checkout was non-importable. The `90ec2b1` refactor
deleted the old flat layout but three `core/` support modules (`constants.py`,
`data_types.py`, `functional_utils.py`), imported across ~15 files, were never
committed and were absent, so `./process.py` failed at import. Recovered from the
original working copy at `/expanse/code/proj/wildcams` (same commit, files untracked).

**Changes (all left uncommitted for review; no detection-behavior change):**
1. **Recovery**: restored the 3 missing `core/` modules; ported uncommitted proj fixes:
   DeepFaune `torch.load(weights_only=False)` (torch 2.6 fix), motion bg-subtractor
   leak fix + `gc.collect()`, per-video/per-step timing logs, sd_watcher udisks mount
   retry/backoff, and the uv-only env rework (`.envrc` drops Nix + watcher auto-start;
   new `watch.sh`/`stop-watcher.sh`/`process-videos.sh`). Deleted `flake.nix`/`flake.lock`.
2. **Guardrails**: added `tests/` (import smoke test = module-graph guardrail; config +
   functional-utils unit tests), `.github/workflows/ci.yml`, and fixed `pyproject.toml`
   to declare the real ML deps (was "SD card downloader" with only python-dotenv).
3. **Docs**: reconciled theory.md/procedures.md/codestructure.md/CLAUDE.md (3-step→4-step,
   `--composite-motion-threshold` default 8.0→0.5, stale flag names, deleted-file refs).
4. **Analytics (safe, no behavior change)**: fixed a list-aliasing bug in
   `session_manager.record_video_success` (Step-4 contributions were mutating the stored
   Step-3 record); replaced a hardcoded analysis path and two hardcoded "0.5" threshold
   log strings with config values; replaced O(n²) `next(...)` scans with dict lookups;
   removed a dead variable in `classification_coordinator`; added div-by-zero guards to
   `postprocessing` coordinate transforms and dropped a redundant `.copy()`.

**Model currency (research, see models.md):** DeepFaune is Europe-only and a poor fit
for Costa Rica; SpeciesNet + `CRI` geofencing is the top recommended addition; BioCLIP 2
is a near-free `pybioclip` bump; YOLO12 is superseded by YOLO26; MegaDetector v6 is
current. None applied yet — each is a behavior-changing experiment for the user to run.

**Deferred behavior-changing items (need ground-truth re-validation):**
- `fullframe_validator.py`: `frames_passed` counts detections, not distinct frames, so a
  track in one frame can satisfy `min_track_frames` via multiple models.
- `fullframe_validator.py`: `temporal_continuity_passed = True` is stubbed (no-op) yet
  ANDed into `validation_passed` and logged as if real.
- Composite-score constants (temporal/consensus/motion multipliers, duration bonus, fps
  fallback 30.0, consensus boost 0.1) are hardcoded and should become CLI-tunable.
- BioCLIP threshold 0.30 likely too high (most predictions 0.08-0.29) — prior finding.

## Config-Gated Scoring Fixes: 2026-07-02 (no processing run; needs ground-truth re-validation)

Implemented the deferred behavior-changing analytics items, each config-gated so
defaults preserve current behavior (82 unit tests pass; detection quality still needs a
ground-truth video run to confirm).

1. **frames_passed bug fixed** (`fullframe_validator.py`): now counts distinct passed
   frames, not synthetic detections (one per model per frame). Identical to the old
   behavior at the default `--min-track-frames 1`; only differs when the threshold is raised.
2. **Temporal continuity implemented** behind `--enable-temporal-continuity-check`
   (default OFF = prior always-pass). When on, the largest gap between a track's passed
   frames must be ≤ `--temporal-continuity-max-gap-seconds` (default 1.0).
3. **Composite-score constants promoted to CLI** (defaults = old hardcoded values):
   `--default-fps` (30.0), `--consensus-boost-per-detection` (0.1),
   `--composite-temporal-multiplier-cap` (2.0), `--composite-consensus-boost-per-model` (0.2),
   `--composite-motion-multiplier-base` (0.5), `--composite-motion-multiplier-span` (1.5),
   `--composite-duration-bonus-base` (0.8), `--composite-duration-bonus-cap` (1.5),
   `--composite-duration-bonus-divisor` (6.0).

**Suggested first experiments once a golden set is available:** raise
`--min-track-frames` to 2-3 (now that frame counting is correct) and try
`--enable-temporal-continuity-check` to cut false positives from erratic single-frame hits.

## Golden-set / model audit: BLOCKED (2026-07-02)

The documented golden standard set (IMG_0001-0020, labels in knowledge/CLAUDE.md and this
file) is **no longer on disk**. `VIDEO_DIR` (/home/rfirmin/Videos/wildcams) now holds 34
`DSCF*.MP4` files (Nov 2025, unlabeled) plus a `may 2025/` archive of 26 species-prefixed
files (ag/an/co/pa/pe = likely agouti/anteater?/coati/paca/peccary), which is undocumented
in the knowledge base. No label CSV/readme exists for either set.

The requested model-currency audit (DeepFaune vs BioCLIP 2 vs SpeciesNet+CRI, see
models.md) cannot run against the golden set until the videos are located or a new labeled
set with documented ground truth is chosen. Pending user decision.

## Model audit setup + puma1 smoke: 2026-07-02

**Golden set replaced:** the IMG_0001-0020 set is gone from disk. Assembled a new audit set
of 28 species-labeled videos from the user's naming convention: puma×2 (top dir) plus
may-2025 ag×5 (agouti), an×1 (anteater), co×14 (coati), pa×4 (paca), pe×2 (peccary). Used
as detection-level true positives (each contains the named animal); user will supply
false-positive/miss cases later.

**puma1 single-video smoke (current default config):**
- Pipeline runs end-to-end on real footage (validates the recovery). ~3m38s/video on CPU,
  ~4.2 GB peak RAM (first run included model downloads).
- **puma1 = FALSE NEGATIVE**, skipped `filtered_by_animal_classification`, but Step 4 received
  **0 input sequences** — it actually failed at **Step 3**: all detections logged
  `conf:0.000 | ovlp:0.000 | threshold_failed`, motion pinned to the frame bottom edge
  (y≈1100-1296 of a 1296-tall frame). The classifiers never saw a crop.
- **Implication:** for videos that fail Step 3, swapping the Step-4 classifier can't help.
  The classifier audit only has signal on videos that reach Step 4 with a crop. Running the
  full 28-video baseline to measure detection recall per species and count how many reach Step 4.

**Also found:** the "BioCLIP" engine is actually OpenAI CLIP, not BioCLIP (see models.md).

## Baseline audit results (28 species videos, current config): 2026-07-02

Ran the current default pipeline over the 28 species-labeled videos (47 min; puma1 reused
its prior .processed result). Detection recall on known positives:

| Species | Detected / Total | Recall |
|---------|------------------|--------|
| agouti (ag) | 3/5 | 60% |
| anteater (an) | 1/1 | 100% |
| coati (co) | 4/14 | 29% |
| paca (pa) | 4/4 | 100% |
| peccary (pe) | 0/2 | 0% |
| puma | 1/2 | 50% |
| **Total** | **13/28** | **46%** |

**Where the 15 failures occur (the decisive finding):**
- **14 failed at Step 3** — 0 crops reached the classifier (0 input to Step 4).
- **1 failed at Step 4** — co12 produced 1 crop that the classifier rejected.

So the Step-4 classifier is responsible for ~7% of misses; **detection + spatial-overlap
validation (Step 3) causes ~93%.** Conclusion: swapping in BioCLIP 2 or SpeciesNet cannot
materially improve recall on this footage — the classifier almost never runs. **The
highest-impact work is Step 3, not the classifier roadmap.**

**Step 3 mechanism (from co1, a 3840×2160 clip):** Step 2 passed; yolo12x/yolo12m produced
2-11 detections per track, but every one logged `conf:0.000 | ovlp:0.000` and failed as
`no_overlap` or `threshold_failed`. Detections exist yet report zero confidence and zero
overlap with motion regions. Next step is a focused Step-3 debug: is confidence being
dropped between detection and the validator, or are motion regions landing away from the
animal (large 4K frames, motion tracks scattered) so overlap legitimately fails? This is a
systematic-debugging task, separate from the (now deprioritized) classifier work.

## Recall tuning: 46% -> 71% via three stacked gates: 2026-07-02

Root-caused the Step-3 recall loss (see prior entry): the pipeline stacks several gates in
series, each calibrated too high for this low-confidence 4K jungle footage. On these clips
detectors emit ~0.02-0.06 confidence per detection; real animals are detected weakly but
consistently across frames.

**The three binding gates (in order the crop hits them):**
1. **Step-3 per-frame gate** — `_process_frame` reused `--confidence-threshold` (0.8) as a
   per-frame pass bar, so a track validated only if a single frame's summed confidence hit
   0.8. Fixed by the new `--frame-pass-confidence-threshold` (defaults to confidence_threshold;
   set very low to make it a coverage check and let the per-track total be the confidence arbiter).
2. **BioCLIP threshold** (0.30) — CLIP scored a coati 0.14-0.29.
3. **Ensemble animal-confidence threshold** (0.5) — rejected the low ensemble confidence.

**Recipe that recovered co1 and lifted the set:**
`--frame-pass-confidence-threshold 0.001 --bioclip-threshold 0.1 --animal-confidence-threshold 0.1`

**Result over the 28 species videos:**

| Species | Baseline | Tuned |
|---------|----------|-------|
| agouti | 3/5 | 4/5 |
| anteater | 1/1 | 1/1 |
| coati | 4/14 | 8/14 |
| paca | 4/4 | 4/4 |
| peccary | 0/2 | 1/2 |
| puma | 1/2 | 2/2 |
| **Total** | **13/28 (46%)** | **20/28 (71%)** |

**Remaining 8 failures all still fail at Step 3 with conf_pass=False** — the detector barely
sees the animal (e.g. co3: one 19.7s track, 1 detection at conf 0.012, motion_align 0.602).
Catching these needs `--confidence-threshold` ~0.01, which would pass almost anything. That
is a detection-quality ceiling, not a threshold to push blind.

**Caveats / next levers:**
- PRECISION IS UNMEASURED. The old false-positive golden set (videos 1-6) is gone; these 28
  are all true positives. Lowering three gates trivially raises recall; the user's forthcoming
  negatives are needed to set the precision/recall balance before changing any DEFAULT.
- Defaults were NOT changed (behavior-preserving). The recipe is a CLI override to validate.
- The real fix for the remaining detection-quality ceiling is likely **crop-based detection**:
  run detectors on motion-region crops (zoomed in) instead of full 4K frames, so small animals
  are larger and score higher confidence. This is the architectural lever (README notes crop
  analysis was previously "eliminated" in favor of full-frame; that trade-off hurts 4K recall).

## Typed detection migration + data-model tightening: 2026-07-02

Executed the plan at docs/superpowers/plans/2026-07-02-typed-detection-migration.md (10 tasks,
tests-first). Finished the abandoned dict->dataclass migration and shrank the data model.

**Data model: 37 -> 30 dataclasses.** Removed: EnsembleResult, ProcessingError, the 3
single-field step `*Data` wrappers, BioCLIPResult + DeepFauneResult (folded into InferenceResult
with an optional `all_predictions`), and the clustering trio (FeatureVector/ClusterResult/
ClusteringAnalysis). Added: `StepResult` base (kw_only) for the shared step-result fields, and
`ScoredDetection` (a Detection plus the per-frame boosted_confidence/motion_overlap/overlap_type
the validator used to bolt onto a dict).

**Detection hot path now typed.** The inference engines (yolo/rtdetr/megadetector) return
`List[Detection]` with `bbox` a `BoundingBox`; postprocessing/ensemble/fullframe_validator use
attribute access; `_calculate_composite_score` returns a widened `CompositeScore` dataclass
instead of a dict. Geometry helpers (`calculate_iou`, coordinate transforms) still take raw
`[x1,y1,x2,y2]` lists (lowest churn; motion-track bboxes are lists).

**Clustering cut** (see models.md): the ResNet18+DBSCAN feature-extraction/clustering chain was
removed end-to-end (feature_extractor.py, the model_manager ResNet load, ensemble_wrapper
feature methods, wildlife_processor.extract_features_from_detection, config flags).

**Verification.** Characterization tests pin the composite-score math, IoU, NMS, area/confidence
filtering, and coordinate transforms; all 95 tests pass with identical asserted numbers.
End-to-end proof: re-running co1 (tuned thresholds) post-migration produced byte-identical Step-3
scores (track_1 ensemble=1.491/composite=2.155; track_6 ensemble=1.430/composite=2.061) and the
same VIDEO SUCCESS outcome as pre-migration. Behavior-neutral, confirmed.

## Lint/format/type gates + cleanup: 2026-07-02

Added ruff (lint+format) and mypy to the dev deps and CI, and let them drive a cleanup pass.
- **ruff** (`select = E,F,I,UP,W`): fixed ~1665 issues automatically (trailing/blank-line whitespace,
  `typing.List/Dict/Optional` -> builtin `list`/`dict`/`X | None` on 3.13, import sorting, unused
  imports, redundant f-strings) plus manual fixes (unused locals, a bare `except`, E402 import
  ordering). All source now passes `ruff check` + `ruff format --check`.
- **Circular import fixed**: ruff's import-sort exposed a latent `config <-> core` cycle (config
  package re-export order was load-bearing). Fixed properly by importing `ProcessingConfig` from its
  submodule (`config.processing_config`) in the modules that closed the cycle — order-independent.
- **mypy** (scoped to `config` + `core/data_types.py`, `follow_imports=silent` as a green baseline):
  found and fixed a `list[int|str]` annotation and removed dead `ConfigurationUpdate` +
  `update_config` (whose `to_dict()` call would have failed). Widen the mypy `files` list as more
  modules are annotated. (A broader mypy run surfaces ~109 findings across the untyped tree,
  including a real `ValidationResult.validated_sequences` mislabel in animal_classification_step —
  future work.)
- Dropped now-unused `scikit-learn` (clustering) from pyproject + the process.py PEP723 header.

CI now gates on: ruff check, ruff format --check, mypy, pytest. All green; 95 tests pass.

## Canonical-Python cleanup: lint/type gates + module splits: 2026-07-02

Cleaned up the codebase toward modern canonical Python. All changes behavior-neutral.

- **mypy to zero across the whole tree** (was 108 errors): fixed via annotations + a few design
  corrections (VideoAnalysis gained real `_step3_data`/`_step4_data`/`_is_failure` fields;
  AnimalClassificationStep retyped to FullFrameValidationResult; PipelineStep's over-rigid
  `process` abstractmethod signature dropped). mypy now covers config/core/ml/motion/pipeline/
  video_io/process.py and is a CI gate (`follow_imports` normal, `ignore_missing_imports=true`).
- **Dead/broken code removed** (surfaced by mypy): a broken `clear_processed_status` (called a
  non-existent method), 7 dead `ensemble_wrapper` backward-compat passthroughs (one with a broken
  signature), an orchestrator hack that hand-built a fake `ValidationResult` for Step 4 (now passes
  the real FullFrameValidationResult), and unused `SessionSummary`/`TimingStatistics` dataclasses.
- **Big files split:**
  - `pipeline/fullframe_validator.py` 669 -> 569; pure scoring extracted to `pipeline/scoring.py`
    (calculate_composite_score / calculate_bbox_overlap / check_temporal_continuity).
  - The extended-track dict-holdout is now typed (`TrackFrameBBox`/`BBoxTrack`/`ExtendedTrack`) —
    the dict->dataclass migration is fully finished.
  - `core/session_manager.py` 601 -> 287; reporting/logging extracted to `core/session_reporter.py`.
- CI gates: ruff check + ruff format --check + mypy + pytest. 97 tests pass.

Behavior verified: co1 (tuned) produced byte-identical Step-3 scores after each phase, and a full
28-video end-to-end run reproduced the pre-refactor result (see next entry for the diff).

## Full 28-video refactor validation: IDENTICAL: 2026-07-02

Ran the full 28-video set (tuned thresholds) after all cleanup phases (mypy-to-zero, dead-code
removal, fullframe_validator + session_manager splits, extended-track typing) and diffed against
the pre-refactor tuned baseline:
- Detected: 20/28 (unchanged).
- Success set: byte-identical (diff empty).
- Every validated TRACK ensemble/composite score: byte-identical (diff empty).
Conclusion: the whole refactor is behavior-neutral end-to-end, not just on unit tests.
