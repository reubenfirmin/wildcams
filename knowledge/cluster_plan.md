# Wildlife Animal Clustering System

## Overview
Multi-stage clustering system to identify and group unique animal types across Costa Rican camera trap videos. Uses species classification + visual similarity to group videos by animal presence.

## Architecture

### Input: High-Confidence Animal Detections
- Source: Videos validated by 4-step pipeline as containing animals
- Data: High-confidence crops from `validated_sequences` 
- Threshold: `combined_score > validation_threshold`

### Stage 1: Within-Video Animal Extraction
```
For each video with animals:
1. Extract all high-confidence crop detections  
2. Run BioCLIP species classification on each crop
3. Cluster crops by species within video
4. Generate representative "animal instances" per video
```

### Stage 2: Cross-Video Animal Clustering  
```
Across all videos:
1. Compare animal instances using combined features
2. Cluster similar animals across videos
3. Output grouped videos by animal type
```

## Model Stack

### Primary Models
1. **BioCLIP** - Species classification (scientific names, confidence scores)
2. **ResNet18** - Visual similarity features (existing integration)
3. **CLIP** - Semantic understanding for unclear/partial animals
4. **DINOv2** - Self-supervised visual features robust to conditions
5. **DeepFaune** - Animal species classification 

### Model Usage Strategy
- **High BioCLIP confidence (>0.7)**: Use species classification for primary grouping
- **Low BioCLIP confidence (<0.7)**: Fall back to visual similarity clustering
- **Cross-validation**: Combine multiple model outputs for robustness

## Implementation

### Core Pipeline
```python
def extract_animal_instances(video_analysis):
    """Extract high-confidence animal crops from validated videos."""
    animal_instances = []
    for detection in video_analysis['validated_sequences']:
        if detection['combined_score'] > threshold:
            crop = extract_crop(detection['best_frame'], detection['bbox'])
            
            # Multi-model feature extraction
            species = bioclip.classify(crop)
            visual_features = resnet18.extract_features(crop)
            semantic_features = clip.extract_features(crop)
            
            animal_instances.append({
                'video': video_path,
                'crop': crop,
                'bbox': detection['bbox'],
                'confidence': detection['combined_score'],
                'species': species,
                'features': {
                    'visual': visual_features,
                    'semantic': semantic_features
                }
            })
    return animal_instances

def cluster_animals(all_animal_instances):
    """Two-stage clustering: species-first, then visual similarity."""
    
    # Stage 1: Group by species (BioCLIP)
    species_groups = defaultdict(list)
    uncertain_group = []
    
    for instance in all_animal_instances:
        if instance['species']['confidence'] > 0.7:
            species_name = instance['species']['scientific_name']
            species_groups[species_name].append(instance)
        else:
            uncertain_group.append(instance)
    
    # Stage 2: Visual clustering within species + uncertain group
    final_clusters = []
    
    for species, instances in species_groups.items():
        # Cluster by visual similarity within species
        visual_clusters = cluster_by_visual_similarity(instances)
        final_clusters.extend(visual_clusters)
    
    # Handle uncertain classifications with pure visual clustering
    if uncertain_group:
        uncertain_clusters = cluster_by_visual_similarity(uncertain_group)
        final_clusters.extend(uncertain_clusters)
    
    return final_clusters
```

## Output Format

### Clustering Results
```json
{
  "clustering_results": [
    {
      "cluster_id": "jaguar_001", 
      "species": {
        "scientific_name": "Panthera onca",
        "common_name": "Jaguar",
        "confidence": 0.89
      },
      "videos": ["IMG_0007.MP4", "IMG_0012.MP4"],
      "instance_count": 3,
      "representative_crop": "path/to/best_crop.jpg"
    },
    {
      "cluster_id": "bird_small_001",
      "species": {
        "scientific_name": "Unknown",
        "common_name": "Small bird",
        "confidence": 0.23
      },
      "videos": ["IMG_0007.MP4"],
      "instance_count": 1,
      "representative_crop": "path/to/bird_crop.jpg"
    }
  ]
}
```

### Summary Report
```
Animal Clustering Results:
Animal #1 (Jaguar): Videos IMG_0007.MP4, IMG_0012.MP4
Animal #2 (Small bird): Videos IMG_0007.MP4
Animal #3 (Unknown mammal): Videos IMG_0011.MP4
```

## Expected Costa Rica Species

### Mammals
- *Panthera onca* (Jaguar)
- *Choloepus hoffmanni* (Two-toed sloth)
- *Bradypus variegatus* (Three-toed sloth) 
- *Nasua narica* (White-nosed coati)
- *Alouatta palliata* (Howler monkey)
- *Cebus imitator* (White-faced capuchin)
- *Ateles geoffroyi* (Spider monkey)
- *Tapirus bairdii* (Baird's tapir)
- *Leopardus pardalis* (Ocelot)

### Birds
- *Ramphastos sulfuratus* (Keel-billed toucan)
- *Ara macao* (Scarlet macaw)
- *Pharomachrus mocinno* (Quetzal)

## Enhanced Features

### Temporal Analysis
- **Activity patterns**: When are different species most active?
- **Seasonal tracking**: Species presence over time
- **Behavior analysis**: Feeding, traveling, resting patterns

### Quality Metrics
- **Confidence scoring**: Combined BioCLIP + visual similarity confidence
- **Diversity metrics**: Species richness, Shannon diversity index
- **Detection reliability**: Cross-validation with manual annotations

### Research Integration
- **iNaturalist export**: Upload observations for scientific validation
- **eBird integration**: Bird species data sharing
- **GBIF compliance**: Biodiversity data standards

## Implementation Phases

### Phase 1: BioCLIP Integration
- Integrate BioCLIP for species classification
- Test on existing high-confidence crops
- Validate species identification accuracy

### Phase 2: Within-Video Clustering
- Implement within-video species grouping
- Handle multiple instances of same species per video
- Generate representative crops per species per video

### Phase 3: Cross-Video Clustering
- Implement cross-video animal matching
- Generate final clustering output format
- Create summary reports for researchers

### Phase 4: Enhanced Visual Similarity
- Add CLIP for semantic understanding
- Add DINOv2 for robust visual features
- Improve clustering accuracy for uncertain cases

### Phase 5: Research Tools
- Temporal analysis capabilities
- Export formats for scientific databases
- Biodiversity metrics and reporting

## Dependencies

### Required Models
- `pybioclip>=0.1.0` - Species classification
- `openai-clip` - Semantic features
- `dinov2` - Self-supervised visual features
- `scikit-learn` - Clustering algorithms

### Integration Points
- Input: Analysis results from 4-step pipeline
- Output: Clustered animal identifications
- Storage: Crop images and feature vectors
- Reporting: Biodiversity analysis and exports

## Success Metrics

### Accuracy Targets
- **Species Classification**: >80% accuracy for common Costa Rica species
- **Clustering Precision**: >90% same-species grouped correctly
- **Clustering Recall**: >85% different-species separated correctly

### Research Value
- **Species Discovery**: Identify new/unexpected species in footage
- **Biodiversity Metrics**: Generate standard ecological indicators
- **Temporal Insights**: Activity patterns and seasonal presence
- **Conservation Impact**: Provide data for habitat protection decisions