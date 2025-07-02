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

**CRITICAL BUG:** Temporal continuity validation is still not working correctly. Despite setting `--detection-validation-gap-seconds 0.2`, IMG_0004 passed with a 0.24s gap.

**Analysis:** The temporal validation logic bug was not fully fixed. Need to investigate the actual implementation.