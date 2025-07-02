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

## Next Test
Need different approach - current motion detection cannot distinguish real animals from false positives:
```bash
process -v 10 11 --conf 0.15 --motion-var-threshold 32 --min-motion-area 300
```
Lower confidence threshold to help IMG_0010 pass full-frame analysis.