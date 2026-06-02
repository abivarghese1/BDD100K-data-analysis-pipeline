# BDD100K Object Detection — Data Analysis

## Overview
This repository contains a fully containerised data analysis pipeline for the
[BDD100K](https://bdd-data.berkeley.edu/) dataset, focused exclusively on the
10 object detection classes with bounding box annotations.

## Detection classes
`car` · `truck` · `bus` · `person` · `rider` · `bike` · `motor` ·
`traffic light` · `traffic sign` · `train`

## Repository structure
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── docs/
│   └── data_analysis_report.md   ← full findings
├── data_analysis/
│   ├── bdd_parser.py             ← dataset parser + dataclasses
│   ├── bdd_analysis.py           ← statistics + anomaly detection
│   ├── dashboard.py              ← quantitative dashboard figures
│   ├── visualize_samples.py      ← qualitative sample figures
│   └── main.py                   ← pipeline entry point
└── tests/
├── fixtures.py
└── test_bdd_parser.py
## Running with Docker (recommended)

```bash
# 1. Build the image
docker build -t bdd100k-analysis .

# 2. Run — mount your BDD100K folder and an outputs folder
docker run --rm \
  -v /absolute/path/to/bdd100k:/app/bdd100k:ro \
  -v $(pwd)/outputs:/app/outputs \
  bdd100k-analysis
```

Figures and stats JSON will appear in `./outputs/` on your host machine.

## Running with docker-compose

```bash
# edit docker-compose.yml to point to your bdd100k folder, then:
docker-compose up --build
```

## Running locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m data_analysis.main
```

## Running tests

```bash
pytest tests/ -v --cov=data_analysis --cov-report=term-missing
```

## Code quality

```bash
black data_analysis/ tests/      # auto-format
pylint data_analysis/            # lint check
```

## Outputs
| File | Description |
|------|-------------|
| `outputs/fig1_distribution.png` | Class counts + val share + annotations/image |
| `outputs/fig2_bbox_areas.png` | Per-class bbox area distributions |
| `outputs/fig3_cooccurrence.png` | Class co-occurrence heatmap |
| `outputs/fig4_attributes.png` | Weather / scene / time-of-day breakdown |
| `outputs/samples_crowded.png` | Most annotation-dense images |
| `outputs/samples_small_boxes.png` | Images with near-invisible objects |
| `outputs/samples_train_class.png` | Rarest class examples |
| `outputs/samples_per_class.png` | One representative image per class |
| `outputs/train_stats.json` | Serialised training statistics |

See [docs/data_analysis_report.md](docs/data_analysis_report.md) for full findings.
