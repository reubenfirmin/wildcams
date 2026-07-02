# Model Currency & Roadmap (assessed 2026-07)

This document records the state of the ML models the pipeline uses versus what is
current in 2026, and a prioritized roadmap. It is a planning document: nothing here
changes pipeline behavior until an experiment is run and validated against the
ground-truth video set (per the project's one-parameter-at-a-time rule).

> **AUDIT RESULT (2026-07-02): the classifier is not the bottleneck — Step 3 is.**
> A baseline run over 28 species-labeled videos detected 13/28 (46% recall). Of the 15
> misses, **14 failed at Step 3 (no crop ever reached the classifier) and only 1 was
> rejected by the classifier itself.** Therefore the classifier-upgrade roadmap below
> (SpeciesNet, BioCLIP 2, DeepFaune removal) would improve recall on ~1 of 15 misses and
> is **deprioritized** until Step-3 detection/spatial-overlap is fixed. See experiments.md
> (2026-07-02 baseline). The BioCLIP-2 upgrade remains worth doing for *species-label
> quality* on the videos that do pass (and because today's "BioCLIP" is really CLIP), but
> it is not a recall fix. The roadmap below is retained for when Step 3 is addressed.

## Current stack

- **Detection ensemble**: YOLO12x, YOLO12m, MegaDetector v6 (`MDV6-yolov10-e`), RT-DETR-l (via ultralytics).
- **Animal classification (Step 4)**: DeepFaune (`deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt`) and "BioCLIP".

> **Correction (2026-07-02 audit):** the "BioCLIP" engine (`ml/inference/bioclip_inference.py`)
> does **not** actually load BioCLIP. It loads vanilla **OpenAI CLIP** (`open_clip`
> `ViT-B-16`, `pretrained='openai'`) and does zero-shot against a hardcoded Costa-Rica
> class list. `pybioclip` is a declared dependency but is unused. So today's species
> classifier is generic CLIP, not the biology-tuned model — a likely contributor to weak
> Neotropical species ID, and it means "upgrade to BioCLIP 2" is really "adopt real BioCLIP
> for the first time." This is now roadmap item #2's true scope.

## Assessment

### The core gap: no Neotropical-tuned classifier
- **DeepFaune is Europe/Eurasia-only.** Its ~37-class label space is European (lynx, wolf, chamois, ibex, reindeer, etc.) with no Neotropical/Costa Rican species. On Costa Rican footage it can only emit a wrong-continent label or a generic one, so as a *species* classifier it is a poor fit. Its architecture (ViT-L/14 DINOv2) is fine; the label space is wrong.
- **BioCLIP** is a general tree-of-life foundation model, better than DeepFaune for Neotropical zero-shot, but as a general model it is typically less accurate than a geofenced, region-trained classifier on the exact target species.

### MegaDetector — current, keep it
- MegaDetector v6 is still the current generation (no v6.1/v7). `MDV6-yolov10-e` is near the top on recall. Optional swap to `MDV6-apa-rtdetr-e` (slightly higher recall, Apache license) costs more CPU (76M params). Low priority.
- Access has moved to `pip install PytorchWildlife`; the repo reorganized under `microsoft/MegaDetector` / `microsoft/Biodiversity`.

### General COCO detectors — largely redundant
- YOLO12 is superseded by **YOLO26** (Jan 2026: NMS-free, up to ~43% faster CPU). For the person/animal/vehicle *detection* role, a COCO-trained YOLO/RT-DETR adds cost and rarely adds recall over MegaDetector, which is purpose-built for camera-trap imagery. Their residual value is ensemble voting to catch MD misses.

### BioCLIP 2 — near-free upgrade
- BioCLIP 2 (ViT-L/14, TreeOfLife-200M) reports +18.1% species accuracy over BioCLIP 1, and is the **default model in current `pybioclip`**. Bumping the `pybioclip` version gets the upgrade with essentially no code change (validate before trusting species output).

## Roadmap (ranked by impact / effort / CPU cost)

| # | Change | Impact | Effort | CPU cost | Status |
|---|--------|--------|--------|----------|--------|
| 1 | Add **SpeciesNet** (`pip install speciesnet`, v4.0.3) as primary classifier, geofenced to `CRI`, fed MegaDetector crops | Very high | Low–medium | Medium (~seconds/image CPU; batch-friendly) | Proposed |
| 2 | Upgrade **BioCLIP 1 → BioCLIP 2** (bump `pybioclip`) | Medium | Very low | ~same | Proposed |
| 3 | **Demote/remove DeepFaune** from Neotropical species ID | Medium (removes wrong-label noise) | Very low (config) | Frees CPU | Proposed |
| 4 | **Drop redundant COCO YOLO/RT-DETR**, keep MDv6 | Low–medium (frees CPU for classifiers) | Low | Frees CPU | Proposed |
| 5 | Add **TropiCam-AI** if canopy cameras are used (84 Neotropical taxa, trained partly on Costa Rica images) | High for arboreal only | Medium | Medium | Optional |
| 6 | Swap detector to `MDV6-apa-rtdetr-e` | Low | Low | Higher | Optional |

**Single highest-impact change**: add SpeciesNet with `CRI` geofencing as the primary classifier and stop relying on DeepFaune for species ID. It replaces a Europe-only label space with a 2,000-label, Neotropical-trained, Costa-Rica-filtered one.

## Integration notes for SpeciesNet (roadmap item 1)
- Package: `speciesnet`; crop classifier weights `v4.0.3a` run on MegaDetector crops (which the pipeline already produces in Step 3).
- Geofencing: pass country ISO code `CRI`; the ensemble rolls species up to genus/family when confidence is low.
- Would slot into Step 4 as a new inference engine alongside / replacing DeepFaune, behind a `--classification-models speciesnet,bioclip` selection. Add config flags for the geofence country and thresholds (NO hardcoded constants).
- Validate against the ground-truth set before changing defaults.

## Clustering (removed 2026-07)
The video-clustering feature (ResNet18 features + DBSCAN, `--enable-clustering`) was cut:
it was an unsupervised proxy for species grouping, fed by the same poor crops that already
limit the pipeline, off by default, never exercised, and carried a latent crash. The
"organize footage by animal" goal is better served by grouping on Step-4 species labels once
a Neotropical classifier (SpeciesNet+CRI / real BioCLIP 2) lands — strictly better than
anonymous visual clusters. Reintroduce it there, not via ResNet18 clustering.

## Caveats / unverified
- Exact MegaDetector v6 re-release date in 2026 is unconfirmed; "MDv6 current, no v6.1/v7" is solid.
- SpeciesNet's per-species Costa Rica coverage is not published as a list; `CRI` geofencing is confirmed supported. Validate on a labeled subset before trusting species-level output.
- The +18.1% BioCLIP 2 figure is a global species metric, not Costa-Rica-specific. Build a small internal validation set before full rollout.
