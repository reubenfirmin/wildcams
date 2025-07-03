
# BioCLIP Integration Plan

## Overview
BioCLIP is a CLIP-based vision foundation model specifically trained for biological organism classification. It represents the next major enhancement to transform our system from generic "animal detection" to **precise species identification and biodiversity monitoring**.

## What is BioCLIP?
- **Foundation Model**: Vision transformer trained on TreeOfLife-10M dataset (450K+ taxa)
- **Taxonomic Scope**: Covers full Linnaean hierarchy from kingdom to species level
- **Performance**: 16-17% better than baselines on biological classification tasks
- **Training Data**: iNaturalist, BIOSCAN-1M, Encyclopedia of Life (most diverse ML-ready bio dataset)

## Integration Architecture

### Current Pipeline Enhancement
```
Current: 4-Step Pipeline → Animal Detection → Generic Analysis
Enhanced: 4-Step Pipeline → Animal Detection → BioCLIP Classification → Species-Based Analysis
```

### Technical Implementation
```python
# Integration in Step 4 after validation
from pybioclip import predict
import cv2

# Extract animal region from best validated detection
best_detection = final_validated_sequences[0]['best_detection']
x1, y1, x2, y2 = map(int, best_detection['bbox'])
animal_roi = best_frame[y1:y2, x1:x2]

# Resize to BioCLIP requirements (224x224)
roi_resized = cv2.resize(animal_roi, (224, 224))

# Get species classification with confidence scores
species_prediction = predict(roi_resized, format='scientific_name')
# Returns: {'Panthera onca': 0.85, 'Leopardus pardalis': 0.12, ...}
```

## Expected Costa Rica Species
BioCLIP should identify common Costa Rican fauna including:

**Mammals**:
- *Panthera onca* (Jaguar)
- *Choloepus hoffmanni* (Two-toed sloth)
- *Bradypus variegatus* (Three-toed sloth)
- *Nasua narica* (White-nosed coati)
- *Alouatta palliata* (Howler monkey)
- *Cebus imitator* (White-faced capuchin)
- *Ateles geoffroyi* (Spider monkey)
- *Tapirus bairdii* (Baird's tapir)
- *Leopardus pardalis* (Ocelot)
- *Leopardus margay* (Margay)

**Birds**:
- *Ramphastos sulfuratus* (Keel-billed toucan)
- *Ara macao* (Scarlet macaw)
- *Pharomachrus mocinno* (Quetzal)

## Implementation Plan

### Phase 1: Basic Integration
1. Add `pybioclip>=0.1.0` dependency
2. Implement species classification in Step 4 after validation
3. Add species metadata to analysis output
4. Update clustering to group by species rather than generic visual similarity

### Phase 2: Enhanced Output
1. **Species-Specific Thumbnails**: Generate representative images per species
2. **Biodiversity Reports**: Automatic species counts and diversity metrics
3. **Confidence Thresholds**: Filter low-confidence species predictions
4. **Hierarchical Clustering**: Group by family/genus when species confidence is low

### Phase 3: Scientific Integration
1. **iNaturalist Integration**: Cross-reference predictions with local observation data
2. **Temporal Analysis**: Track species activity patterns over time
3. **Conservation Metrics**: Generate biodiversity indicators for research
4. **Export Formats**: eBird, GBIF-compatible outputs for scientific databases

## Expected Performance Impact

### Processing Time
- **BioCLIP Inference**: ~50ms per animal detection
- **4-Step Pipeline Advantage**: Fewer validated detections = faster BioCLIP processing
- **Overall Impact**: <20% increase in total processing time

### Accuracy Improvements
- **Species-Level Precision**: Expected >80% accuracy for common Costa Rican species
- **Hierarchical Fallback**: Family/genus classification when species uncertain
- **False Positive Reduction**: Additional validation layer using biological knowledge

### Output Enhancement
```json
// Current output
{
  "video_file": "IMG_0007.MP4",
  "detection": {"confidence": 0.73, "bbox": [...], "combined_score": 0.65},
  "processing_mode": "next_generation_4step_pipeline",
  "validated_sequences": 1
}

// Enhanced output with BioCLIP
{
  "video_file": "IMG_0007.MP4", 
  "detection": {"confidence": 0.73, "bbox": [...], "combined_score": 0.65},
  "processing_mode": "next_generation_4step_pipeline",
  "species": {
    "scientific_name": "Panthera onca",
    "common_name": "Jaguar", 
    "confidence": 0.85,
    "taxonomy": {
      "kingdom": "Animalia",
      "phylum": "Chordata", 
      "class": "Mammalia",
      "order": "Carnivora",
      "family": "Felidae"
    }
  },
  "cluster": "panthera_onca_001"
}
```
