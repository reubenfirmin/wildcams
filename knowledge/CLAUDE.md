# Wildlife Camera Trap Video Processing System

## Project Overview
Automated wildlife video processing system for Costa Rican jungle camera footage. Detects animals using ensemble ML models, filters false positives, and clusters videos based on visual similarity of detected animals.

## TODO List
* ✅ Camera handling detection fixed (inverted logic implemented)
* ✅ Full-frame analysis prioritized (crop analysis eliminated)
* Tune motion sensitivity for small animals (test MIN_MOTION_AREA 200-500, MOTION_VAR_THRESHOLD 15-35)
* Test full video set (20 videos) with current pipeline to validate performance

## Current Status
- **3-Step Pipeline**: Motion detection → Camera handling filter → Full-frame ensemble analysis
- **Camera Handling Detection**: Fixed inverted logic (spatial dispersion + motion sparsity)
- **Spatial Overlap Validation**: Full-frame detections must correlate with motion regions
- **Default Ensemble**: yolo12x, yolo12m, MDV6-yolov10-e, rtdetr-l

## Ground Truth Reference
- **Videos 7,8,9,11,12**: True positives (should detect animals)
- **Videos 1-6**: False positives (should NOT detect animals)  
- **Videos 13-19**: Camera handling (should be filtered in Step 2)

## Key Commands
```bash
# Test specific videos
./process.py -v 7 18

# Full processing with current defaults
./process.py

# Camera handling detection tuning
./process.py --composite-motion-threshold 8.0

# Motion sensitivity tuning
./process.py --min-motion-area 300 --motion-var-threshold 32
```

## Document Index

### Core Documentation
- **knowledge/CLAUDE.md** - This file: minimal project context and procedures
- **knowledge/theory.md** - Technical architecture, model details, validation algorithms
- **knowledge/procedures.md** - Step-by-step procedures for common tasks
- **knowledge/codestructure.md** - Code structure and organization
- **experiments.md** - Experimental results and parameter tuning history


## Critical Rules
- **NO hardcoded constants**: All parameters must be CLI configurable
- **User will always perform tests**: Provide test commands (using process command from flake) for user to test with
- **New experiments should change only one or two parameters**: Follow scientific process and do not change multiple parameters at once
- **Document experiments**: Log all parameter changes in experiments.md
- **Consult procedures.md**: Follow standard procedures for common tasks

## Debugging Methodology
- **NO GUESSING**: When you encounter missing files, URLs, or errors - search for exact names/messages first
- **USE PROVIDED SOURCES**: Reference the actual code/repos provided by user, not improvised solutions
- **ASK FOR SPECIFICS**: If you need information, ask user exactly what to look for instead of improvising
- **SYSTEMATIC DEBUGGING**: Debug step-by-step using actual source code, not random trial-and-error fixes
- **SEARCH BEFORE FIXING**: For missing files/models, search for the exact filename before trying alternatives
- **NO LAZY SHORTCUTS**: Don't skip research steps or assume solutions - do the actual work to find correct answers
- **EXHAUSTIVE INVESTIGATION**: Read the full source code, check all relevant files, search thoroughly before concluding
- **FIX, DON'T DELETE**: When code is wrong, replace it with correct implementation - don't just remove functionality
- **UNDERSTAND INTENT**: Before removing code, understand what it was trying to accomplish and preserve that functionality