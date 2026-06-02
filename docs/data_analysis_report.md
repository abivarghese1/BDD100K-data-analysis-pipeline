# BDD100K Object Detection — Data Analysis Report

## 1. Dataset overview

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 69,863 | ~1.8 million |
| Val   | 10,000 | ~260,000 |

Images are 1280×720 px. Only the 10 detection classes with bounding boxes
are analysed; drivable area and lane marking labels are excluded.

## 2. Class distribution

**Finding: Severe class imbalance.**
`car` accounts for roughly 40–45% of all annotations. `train` is the rarest
class, appearing in under 1% of images. The ordering from most to least
frequent is consistently:

> car › traffic sign › traffic light › person › truck ›
> bus › bike › rider › motor › train

This imbalance has direct implications for model training — a naive cross-
entropy loss will be dominated by the car class and the detector will
systematically under-detect rare classes.

**Recommendation:** Use focal loss or per-class weighted sampling during
model training.

## 3. Train / val split

The val set is approximately 12.5% of the combined dataset (10k / 80k).
Per-class val share is close to 12.5% for common classes but deviates for
rare ones (e.g. `train`) due to the low absolute count making random
variation more pronounced.

## 4. Bounding box size analysis

| Class | Typical area (px²) | Notes |
|-------|-------------------|-------|
| car | 8,000–40,000 | Wide range; cars from distant to close |
| bus / truck | 20,000–150,000 | Large objects, often fill frame |
| person | 500–8,000 | Small when distant |
| traffic light | 200–2,000 | Small, high aspect ratio (portrait) |
| traffic sign | 300–5,000 | Variable shape |
| train | 10,000–200,000 | Very large when present |
| bike / motor | 500–6,000 | Small, often occluded |

**Finding:** Bbox areas span 4+ orders of magnitude. A single-scale
detector will struggle — a multi-scale FPN head with small anchors
(8×8 or 16×16) is necessary.

## 5. Co-occurrence patterns

Two clear clusters emerge from the co-occurrence matrix:

- **Urban driving cluster:** `car`, `traffic sign`, `traffic light`
  almost always appear together in city street scenes.
- **Vulnerable road user cluster:** `person`, `rider`, `bike`
  frequently co-occur, especially in urban and residential scenes.

`train` and `motor` almost never co-occur with each other, confirming
they represent distinct and rare scene types.

## 6. Scene attribute analysis

| Attribute | Dominant value | Share |
|-----------|----------------|-------|
| Weather | Clear | ~50% |
| Scene | City street | ~55% |
| Time of day | Daytime | ~60% |

**Finding:** The dataset is heavily biased toward clear-weather daytime
city driving. Night, rain, and fog conditions are underrepresented.
Models trained without augmentation will generalise poorly to adverse
conditions.

## 7. Anomalies identified

| Anomaly | Count (train) | Implication |
|---------|--------------|-------------|
| Empty images (0 det. anns) | ~3,500 | Background frames; useful as hard negatives |
| Crowded images (>30 anns) | ~200 | Dense urban scenes; test NMS robustness |
| Very small boxes (<100 px²) | ~15,000 | Noise or extreme distance; may need filtering |
| Very large boxes (>100k px²) | ~8,000 | Close-range large vehicles; valid |
| Extreme aspect ratio | ~5,000 | Truncated objects at frame edges |

## 8. Qualitative observations

- **Crowded images** are predominantly city intersection scenes with
  high pedestrian and vehicle density. Annotation quality is generally
  good but some overlapping boxes exist.
- **Very small boxes** are mostly distant traffic lights and signs at
  the edge of visibility. Many are arguably below the useful detection
  threshold for a standard 640×640 input resolution.
- **Train class images** are rare and visually distinctive — level
  crossings or rail yards. The detector will have very few examples
  to learn from.
- **Rider class** has the highest occlusion rate, as riders are
  frequently partially hidden by vehicles or other riders.

## 9. Recommendations for model pipeline

1. Use **focal loss** to handle class imbalance.
2. Add **small-object augmentation** (mosaic, copy-paste) for person,
   bike, traffic light.
3. Apply **night/rain augmentation** (brightness jitter, rain overlay)
   to compensate for scene bias.
4. Consider **filtering boxes < 32px²** as they are below the
   detection capability of most architectures at standard resolutions.
5. Use **stratified sampling** in data loaders to oversample rare
   classes (train, motor).
