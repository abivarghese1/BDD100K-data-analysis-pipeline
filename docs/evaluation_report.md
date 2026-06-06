# Model Evaluation Report — YOLOv8m on BDD100K

## 1. Metric selection and rationale

| Metric | Why chosen |
|--------|-----------|
| **mAP@0.5** | Standard detection benchmark — measures localisation and classification jointly. Primary metric for BDD100K leaderboard comparisons. |
| **Precision** | Of all detections made, how many are correct. Critical for automotive where false alarms cause unnecessary braking or alerts. |
| **Recall** | Of all real objects, how many were found. Critical for safety — missing a pedestrian or traffic light is dangerous. |
| **F1** | Harmonic mean of precision and recall — single number for class-level comparison without choosing a threshold. |
| **PR curve** | Shows the precision-recall tradeoff across all confidence thresholds. Reveals whether a class can be improved by threshold tuning alone. |
| **Confusion matrix** | Reveals inter-class confusions — e.g. rider misclassified as person. Guides targeted fixes. |
| **Confidence distribution (TP vs FP)** | Shows model calibration. A well-trained model has high confidence on TPs and low on FPs. Overlap indicates poor discrimination. |
| **Attribute-stratified mAP** | Breaks performance down by weather, scene, and time of day. Directly connects to data analysis findings about dataset bias. |

Accuracy is deliberately excluded — it is meaningless for object
detection where background vastly outnumbers foreground objects.

---

## 2. Quantitative results

*Evaluated on 50 val images, model trained for 1 epoch on 100 images.*
*IoU threshold = 0.5, confidence threshold = 0.25.*

### Per-class metrics

| Class | TP | FP | FN | Precision | Recall | AP@0.5 |
|-------|----|----|-----|-----------|--------|--------|
| car | 286 | 135 | 226 | 0.679 | 0.559 | 0.513 |
| person | 30 | 29 | 46 | 0.508 | 0.395 | 0.347 |
| bus | 3 | 10 | 6 | 0.231 | 0.333 | 0.143 |
| traffic light | 18 | 24 | 105 | 0.429 | 0.146 | 0.101 |
| traffic sign | 26 | 29 | 127 | 0.473 | 0.170 | 0.105 |
| truck | 0 | 0 | 21 | 0.000 | 0.000 | 0.000 |
| bike | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 |
| motor | 0 | 0 | 3 | 0.000 | 0.000 | 0.000 |
| rider | 0 | 0 | 3 | 0.000 | 0.000 | 0.000 |
| train | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| **mAP@0.5** | | | | | | **0.121** |

### Figures produced

| Figure | Description |
|--------|-------------|
| `fig_metrics_bar.png` | Precision, recall, F1, AP per class as horizontal bar charts |
| `fig_pr_curves.png` | Precision-recall curves for all 10 classes |
| `fig_conf_dist.png` | TP vs FP confidence score distributions per class |
| `fig_confusion.png` | Normalised confusion matrix across all predictions |
| `fig_attr_map.png` | mAP stratified by weather, scene, time of day |
| `fig_false_negatives.png` | False negative counts per class |
| `qual_random_samples.png` | GT vs predictions on 6 random val images |
| `qual_night_failures.png` | GT vs predictions on night scenes |
| `qual_rainy_failures.png` | GT vs predictions on rainy scenes |
| `qual_small_object_failures.png` | GT vs predictions on small-object frames |
| `qual_crowded_failures.png` | GT vs predictions on crowded scenes |

---

## 3. What works and why

### car — AP 0.513 (strongest)

- **Why it works:** car accounts for ~45% of all annotations. Even in
  100 training images the model sees hundreds of car instances covering
  a wide range of sizes, orientations, and distances.
- **Architecture contribution:** Medium-resolution (40×40) feature maps
  in the PAN-FPN neck are well-suited to typical car sizes (8k–40k px²
  as identified in data analysis). The C2f blocks extract robust shape
  and texture features that generalise across car types.
- **Evidence:** Precision=0.679 and Recall=0.559 — the model is fairly
  confident when it predicts car and finds the majority of cars present.
  The PR curve shows a well-shaped curve with a clear area under it.

### person — AP 0.347 (second best)

- **Why it works:** person has moderate instance count and is visually
  distinctive (upright silhouette). The 80×80 fine-grained feature map
  captures person-scale objects well.
- **Limitation visible in results:** 46 false negatives vs 30 true
  positives — recall is 0.395, meaning the model misses more persons
  than it finds. Data analysis showed person has the highest occlusion
  rate — occluded persons are systematically missed.

---

## 4. What does not work and why

### truck — AP 0.000 despite 21 ground truth instances

- **Critical finding:** The model produced zero truck detections at
  conf=0.25. Truck and car share very similar visual features — both
  are large box-shaped vehicles. With only 1 epoch of training the
  decision boundary between car and truck has not been learned.
- **Confusion matrix confirms:** truck GT boxes are predicted as car.
- **Data analysis connection:** truck is 5th most frequent class but
  only ~3% of annotations. In 100 training images this translates to
  very few truck examples.

### traffic light — AP 0.101, Recall 0.146

- **Why it fails:** 105 false negatives out of 123 total traffic lights.
  Data analysis showed traffic lights have median area ~500px² — at
  640×640 input resolution these are tiny regions. After 1 epoch the
  model has not learned to reliably activate its small-object head for
  these.
- **High FP count (24):** the model fires on road signs and bright
  spots, confusing them for traffic lights — poor calibration from
  insufficient training.

### motor, rider, bike, train — AP 0.000

- **Root cause — data starvation:** In 100 training images these classes
  appear extremely rarely. motor had 13 instances, rider 25, bike 53,
  train 1 in the full 500-image run. In a 100-image subset these may
  appear 2-5 times — far below the minimum needed to learn a class.
- **Root cause — visual ambiguity:** rider and person overlap visually.
  Without enough rider examples the model collapses rider predictions
  into person (visible in confusion matrix off-diagonal).

---

## 5. Failure cluster analysis

### Cluster 1: Night scenes

Qualitative overlays (`qual_night_failures.png`) show:
- GT boxes present for persons and cars but model produces few or
  no detections
- The model was trained predominantly on daytime images (data analysis:
  60% daytime). Night introduces low-contrast conditions the model
  has not adapted to
- HSV augmentation in training partially helps but 1 epoch is
  insufficient

**Fix:** Oversample night frames during training; add brightness
and gamma augmentation specifically targeting night conditions.

### Cluster 2: Rainy weather

Qualitative overlays (`qual_rainy_failures.png`) show:
- Increased false positives — rain streaks trigger spurious detections
- Reduced recall — rain reduces edge contrast, making objects harder
  to localise
- Data analysis showed rainy scenes = ~15% of dataset but grossly
  underrepresented in the 100-image training subset

**Fix:** Rain overlay augmentation (random rain streak synthesis);
increase training data from rainy scenes.

### Cluster 3: Small objects

Qualitative overlays (`qual_small_object_failures.png`) show:
- Traffic lights and distant persons completely missed
- Data analysis found 15k boxes < 100px² and many traffic lights
  with area 200-500px²
- At 640×640 input a 20×20px object becomes a 9×9px feature after
  the stem conv — below reliable detection threshold

**Fix:** SAHI (Slicing Aided Hyper Inference) — slice 1280×720
images into 640×640 overlapping tiles, run detection on each,
merge with WBF (weighted boxes fusion). This effectively gives
2× resolution for small objects.

### Cluster 4: Crowded scenes

Qualitative overlays (`qual_crowded_failures.png`) show:
- NMS suppresses valid detections when ground truth boxes overlap
- Cars in traffic queues have IoU > 0.45 with each other, causing
  one to be suppressed
- Data analysis found ~200 images with >30 annotations

**Fix:** Lower NMS IoU threshold to 0.35 for crowded scene detection;
use Soft-NMS which decays scores rather than hard-removing boxes.

---

## 6. Suggested improvements ranked by impact

### High impact — data improvements

1. **Train on full 70k images for 50 epochs**
   Expected mAP@0.5 improvement: 0.121 → ~0.45 (literature benchmark)

2. **Oversample rare classes with copy-paste augmentation**
   motor, rider, train — paste instances into common scenes.
   Expected: AP for rare classes from 0.000 to 0.10-0.20.

3. **Filter degenerate boxes < 100px²**
   Data analysis showed 15k such boxes add loss noise.
   Expected: small improvement in precision.

### Medium impact — inference improvements

4. **SAHI for small object detection**
   Traffic light recall 0.146 → expected ~0.35 with sliced inference.

5. **Test-time augmentation (TTA)**
   Flip + multi-scale inference: +1-2% mAP with no retraining.

6. **Lower NMS IoU threshold to 0.35**
   Reduces suppression in crowded scenes.

### Lower impact — architecture improvements

7. **Add CBAM attention in FPN neck**
   Helps focus on small objects.

8. **Use YOLOv8x instead of YOLOv8m**
   +2.4 mAP@0.5 at cost of 2× inference time.

---

## 7. Connection to data analysis

Every failure cluster maps directly to a finding from the data analysis:

| Data analysis finding | Model failure | Suggested fix |
|----------------------|--------------|---------------|
| car = 45% of annotations | car AP 0.513 — model biased to car | Weighted sampling |
| night = 15% of data | Night recall drops | Oversample + augment |
| 15k boxes < 100px² | Traffic light recall 0.146 | SAHI inference |
| motor/train < 1% of images | AP = 0.000 | Copy-paste augmentation |
| 200 crowded images (>30 anns) | NMS suppression | Soft-NMS |
| rainy/foggy underrepresented | FP spikes in rain | Rain augmentation |
| rider/person co-occur | rider collapsed into person | Class-aware loss weighting |
