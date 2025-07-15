# Common Procedures

## Analyze Experiment
**Task**: Review experimental results and document findings

**Process**:
1. **Identify log file**: Find most recent `wildcams_YYYYMMDD_HHMMSS.log`
2. **Extract results**: Parse processing outcomes for each video
3. **Compare to ground truth**: 
   - Videos 7,8,9,10,11,12: Should detect animals
   - Videos 1-6: Should NOT detect animals
   - Videos 13-19: Should be filtered as camera handling
4. **Document in experiments.md**: Use terse format with log filename
5. **Update TODO list** if needed

**Required Files**: `experiments.md`, latest log file

## Find Bug
**Task**: Debug processing failures or unexpected results

**Process**:
1. **Read latest log file**: Look for error patterns, stack traces, unexpected outcomes
2. **Check video results**: Compare `.processed` files vs expected outcomes  
3. **Trace through pipeline**: Follow Steps 1→2→3 for problematic videos
4. **Identify root cause**: Motion filter? Spatial validation? Model issues?
5. **Test fix** with minimal video set before full runs

**Required Files**: Latest log file, `.processed` result files, `process.py`

## Add Code Feature  
**Task**: Implement new functionality or parameters

**Process**:
1. **Add CLI parameter**: Update `video_processor_base.py` argument parsing
2. **Add to config class**: Update `ProcessingConfig` in `process.py`
3. **Implement logic**: Add feature code with NO hardcoded constants
4. **Test with 1-2 videos**: Verify functionality works
5. **Document in CLAUDE.md** if significant

**Required Files**: `video_processor_base.py`, `process.py`, test videos

## Tune Parameters
**Task**: Optimize detection thresholds and settings

**Process**:
1. **Identify parameter**: Check CLI help or current defaults
2. **Test range**: Try 3-5 different values with same video set
3. **Compare results**: Document in experiments.md with metrics
4. **Update default** if improvement is significant and consistent
5. **Update CLAUDE.md** if default changed

**Required Files**: Current config, test videos, experiments.md

## Critical Rules
- **NO hardcoded constants**: All values must be CLI parameters
- **Test before commit**: Always test changes with videos 7,18 minimum  
- **Document experiments**: Use terse format in experiments.md
- **Check ground truth**: Know what each test video should contain