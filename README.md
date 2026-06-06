# BDD100K Object Detection — Data Analysis & Model Pipeline

End-to-end pipeline for the BDD100K dataset covering data analysis,
model training, evaluation, and containerised deployment.
Focused on the 10 detection classes with bounding box annotations.

## Detection classes

`bike` · `bus` · `car` · `motor` · `person` · `rider` ·
`traffic light` · `traffic sign` · `train` · `truck`

---

## Repository structure
.
├── Dockerfile                        # self-contained Docker image
├── docker-compose.yml                # compose config with volume mounts
├── requirements.txt                  # all Python dependencies
│
├── configs/
│   └── bdd100k.yaml                  # Ultralytics dataset config
│
├── data_analysis/
│   ├── init.py
│   ├── bdd_parser.py                 # JSON parser + Frame/Annotation/BBox dataclasses
│   ├── bdd_analysis.py               # statistics + anomaly detection
│   ├── dashboard.py                  # quantitative dashboard figures (fig1–fig4)
│   ├── visualize_samples.py          # qualitative sample visualisation
│   └── main.py                       # data analysis entry point
│
├── model/
│   ├── bdd_dataset.py                # PyTorch Dataset + DataLoader
│   ├── bdd_to_yolo.py                # BDD100K JSON → YOLO format converter
│   ├── train.py                      # training pipeline (full + 1-epoch subset)
│   ├── evaluate.py                   # quantitative + qualitative evaluation
│   └── predict.py                    # single-image inference + visualisation
│
├── docs/
│   ├── data_analysis_report.md       # EDA findings, anomalies, patterns
│   ├── model_report.md               # model selection, architecture deep-dive
│   └── evaluation_report.md          # metrics, failure analysis, improvements
│
└── tests/
├── fixtures.py                   # synthetic BDD100K test data
└── test_bdd_parser.py            # unit tests for parser + dataclasses
---

## Quickstart with Docker

The container is self-contained. Mount your BDD100K data at runtime.

### 1. Build

```bash
docker build -t bdd100k-analysis .
```

### 2. Data analysis

```bash
docker run --rm \
  -v /path/to/bdd100k:/app/bdd100k:ro \
  -v /path/to/bdd100k/labels:/app/bdd100k_labels:ro \
  -v $(pwd)/outputs:/app/outputs \
  bdd100k-analysis
```

Outputs written to `./outputs/`:

| File | Description |
|------|-------------|
| `fig1_distribution.png` | Class counts + val share + annotations per image |
| `fig2_bbox_areas.png` | Per-class bbox area distributions (log scale) |
| `fig3_cooccurrence.png` | Class co-occurrence heatmap |
| `fig4_attributes.png` | Weather / scene / time-of-day breakdown |
| `samples_crowded.png` | Most annotation-dense images |
| `samples_small_boxes.png` | Near-invisible object examples |
| `samples_train_class.png` | Rarest class examples |
| `samples_per_class.png` | One representative image per class |
| `train_stats.json` | Serialised training statistics |

### 3. Train (1-epoch subset, 100 images)

```bash
docker run --rm \
  -v /path/to/bdd100k:/app/bdd100k:ro \
  -v /path/to/bdd100k/labels:/app/bdd100k_labels:ro \
  -v $(pwd)/outputs:/app/outputs \
  bdd100k-analysis \
  python -m model.train --mode subset --max-samples 100 --batch-size 2
```

### 4. Evaluate

```bash
docker run --rm \
  -v /path/to/bdd100k:/app/bdd100k:ro \
  -v /path/to/bdd100k/labels:/app/bdd100k_labels:ro \
  -v $(pwd)/outputs:/app/outputs \
  bdd100k-analysis \
  python -m model.evaluate \
    --model outputs/runs/yolov8_bdd_subset2/weights/best.pt \
    --max-val-images 200 \
    --conf 0.25
```

Outputs written to `./outputs/evaluation/`:

| File | Description |
|------|-------------|
| `fig_metrics_bar.png` | Precision, recall, F1, AP per class |
| `fig_pr_curves.png` | PR curves for all 10 classes |
| `fig_conf_dist.png` | TP vs FP confidence distributions |
| `fig_confusion.png` | Normalised confusion matrix |
| `fig_attr_map.png` | mAP stratified by weather / scene / time |
| `fig_false_negatives.png` | Missed detections per class |
| `qual_random_samples.png` | GT vs predictions (random samples) |
| `qual_night_failures.png` | Failure cluster: night scenes |
| `qual_rainy_failures.png` | Failure cluster: rainy weather |
| `qual_small_object_failures.png` | Failure cluster: small objects |
| `qual_crowded_failures.png` | Failure cluster: crowded scenes |

### 5. Single-image inference

```bash
docker run --rm \
  -v /path/to/bdd100k:/app/bdd100k:ro \
  -v $(pwd)/outputs:/app/outputs \
  bdd100k-analysis \
  python -m model.predict \
    --image bdd100k/images/100k/val/0000f77c-6257be58.jpg \
    --model outputs/runs/yolov8_bdd_subset2/weights/best.pt \
    --output outputs/prediction.png
```

---

## Running locally on Apple Silicon (MPS)

Docker on Mac cannot access the GPU. Run training directly for MPS acceleration:

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export IMAGE_DIR=/path/to/bdd100k
export LABEL_DIR=/path/to/bdd100k/labels

python -m model.train --mode subset --max-samples 500 --batch-size 4
```

MPS gives roughly 5–10× speedup over CPU on M-series chips.

---

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v --cov=data_analysis --cov-report=term-missing
```

---

## Code quality

```bash
black data_analysis/ model/ tests/
pylint data_analysis/ model/ --fail-under=8.0
```

---

## Model summary

**YOLOv8m** — chosen for multi-scale detection capability (PAN-FPN),
anchor-free head, and strong COCO pretrained weights transferable to
BDD100K's 10 classes. See [docs/model_report.md](docs/model_report.md)
for full architecture explanation and rationale.

### Key results (1 epoch, 100 training images)

| Class | AP@0.5 |
|-------|--------|
| car | 0.513 |
| person | 0.347 |
| bus | 0.143 |
| traffic sign | 0.105 |
| traffic light | 0.101 |
| truck / bike / motor / rider / train | 0.000 |
| **mAP@0.5** | **0.121** |

Low mAP is expected for 1-epoch subset training — full training on
70k images for 50 epochs is projected to reach ~0.45 mAP@0.5.
See [docs/evaluation_report.md](docs/evaluation_report.md) for
full analysis and improvement roadmap.

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/data_analysis_report.md](docs/data_analysis_report.md) | EDA, class distribution, bbox analysis, co-occurrence, anomalies |
| [docs/model_report.md](docs/model_report.md) | Model selection rationale, YOLOv8m architecture deep-dive, training config |
| [docs/evaluation_report.md](docs/evaluation_report.md) | Per-class metrics, failure clusters, improvement suggestions |
