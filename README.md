# Wildlife camera processing 

This project uses a locally running CPU feasible ML pipeline to process wildlife camera videos.

The project is setup using *Nix* and *direnv* to supply system level dependencies. *Uv* is used to manage python dependencies. 

## Why?

I wanted:

1) exposure to uv and nix
2) to see how far I could push a 100% claude coded project, which this is
3) to automate a tedious task I do every few weeks when the wildlife cameras come in from the jungle

# Features

## Automatic SD card copying

Direnv starts sd_watcher.py when you change into the working directory, which monitors for new sd cards. 
When an sd card is detected, the card is mounted, and the script searches for video files. Any detected video
files are copied to the video directory. Videos are hashed so that duplicates are not copied.

This allows you to swap cards in and out of the card reader without touching the keyboard, just waiting for the copying to complete each time.

## 4 Step ML Pipeline

The codebase has four steps, which run for each video:

1) Motion detection. Look for temporally consistent motion tracks across videos. This uses opencv to find motion regions. Sequences of insufficient length are dropped, and then sequences that feature roughly the same area of motion are infilled and merged together (on the theory that an animal can move, stop, and then continue moving.) Finally, sequences are outfilled (meaning that videos near the start or end of the video are extended to the start/end of the video, again on the assumption that an animal could start or end still.) This step is valid because the footage being processed is from wildlife camera, which are triggered by fairly crude motion detectors in the first place.
2) Camera handling rejection. Videos full of incoherent movement are dropped. Sometimes the cameras will be left on while they are carried back out of the jungle, which results in dozens or more of full frame video movement.
3) Object detection in movement regions from step 1. Frames are sampled, and per frame an ensemble of object detection models is used to score whether there is an object detected within each movement sequence that crosscuts the frame. Each sequence's scores are aggregated once all frames are processed, and sufficiently high scoring sequences proceed to step 4.
4) Animal detection. Finally, a small ensemble of two dedicated animal classification models are used to specifically filter for animals within frames. The idea here is to eliminate movement from non-animal objects. Unfortunately there are no (readily available) animal detection models which a) work well and b) include Costa Rican animals, but the signal from these two videos is sufficient to remove some false positives.

# Code Quality / Vibe Coding Experience

This was 100% written by Claude Code, with around $200 of prompting; one of my goals was to push Claude Code. Combined, it probably took around 4-5 days of work (in early mornings, evenings, and weekends over a couple of weeks). I've worked adjacent to analytics teams, but have never written ML pipeline (or video processing) code in Python or any other language. When I started this, I didn't know what the architecture should be; I didn't know what models to use; I didn't know what techniques were appropriate. Is it cutting edge? Certainly not. Does it work? Reasonably. Could I have coded this in an equivalent amount of time myself, or using a conversational AI? Probably, but it would have been a higher hurdle to clear in order to get started.

Claude can be extremely lazy, particularly as the context gets jumbled. I ended up creating the knowledge folder to allow me to clear the context when this started happening, and get the new Claude up to speed reasonably quickly. Having a file which indexes into other files seemed reasonably effective, though periodically the Claude would forget that these files existed and would have to be reminded. Sometimes when I would ask it to refactor or create a new code path, it would use dummy values, take shortcuts, comment out code, etc - if you are too hands off, you end up with bad results.

By default its code quality is not great. Setting code standards at the start (via e.g. a knowledge/codestandards.md) would probably have helped.
