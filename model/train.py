"""
train.py
--------
Training pipeline for YOLOv8 on BDD100K.

Two modes:
1. subset -- converts a small subset to YOLO format and runs 1 epoch
             using the Ultralytics trainer (recommended)
2. full   -- full fine-tuning using Ultralytics YAML config
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def train_ultralytics(
    data_yaml="configs/bdd100k.yaml",
    model_size="yolov8m.pt",
    epochs=10,
    imgsz=640,
    batch=16,
    output_dir="outputs/runs",
):
    """Fine-tune YOLOv8 on BDD100K using the Ultralytics training API.

    Args:
        data_yaml: Path to the BDD100K dataset YAML config.
        model_size: Ultralytics model checkpoint to start from.
        epochs: Number of training epochs.
        imgsz: Input image size (square).
        batch: Batch size.
        output_dir: Directory for run artefacts.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    log.info("Loading model: %s", model_size)
    model = YOLO(model_size)
    log.info(
        "Starting training -- epochs=%d  imgsz=%d  batch=%d",
        epochs, imgsz, batch,
    )
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name="yolov8_bdd100k",
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        mosaic=1.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    log.info("Training complete. Artefacts saved to %s/", output_dir)


def _prepare_subset_dataset(label_dir, image_dir, max_samples, work_dir):
    """Convert a BDD100K subset to YOLO format and write a dataset YAML.

    Args:
        label_dir: Directory containing BDD100K JSON label files.
        image_dir: Root directory of BDD100K images.
        max_samples: Number of training images to use.
        work_dir: Working directory for converted data.

    Returns:
        Path to the generated dataset YAML file as a string.
    """
    import json

    img_out = work_dir / "images" / "train"
    lbl_out = work_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    class_map = {
        "bike": 0, "bus": 1, "car": 2, "motor": 3,
        "person": 4, "rider": 5, "traffic light": 6,
        "traffic sign": 7, "train": 8, "truck": 9,
    }
    img_w, img_h = 1280.0, 720.0

    train_json = os.path.join(
        label_dir, "bdd100k_labels_images_train.json"
    )
    log.info("Reading labels from %s ...", train_json)
    with open(train_json, encoding="utf-8") as fh:
        frames = json.load(fh)[:max_samples]

    copied, skipped = 0, 0
    for frame in frames:
        name = frame["name"]
        stem = Path(name).stem

        src = Path(image_dir) / "images" / "100k" / "train" / name
        dst = img_out / name
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1

        lines = []
        for label in frame.get("labels", []):
            cat = label.get("category", "")
            if cat not in class_map:
                continue
            box = label.get("box2d")
            if not box:
                continue
            cx = (box["x1"] + box["x2"]) / 2.0 / img_w
            cy = (box["y1"] + box["y2"]) / 2.0 / img_h
            w  = (box["x2"] - box["x1"]) / img_w
            h  = (box["y2"] - box["y1"]) / img_h
            if w > 0 and h > 0:
                lines.append(
                    f"{class_map[cat]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
        (lbl_out / f"{stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    log.info("Images -- copied=%d  skipped=%d", copied, skipped)

    yaml_path = work_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {work_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "nc: 10\n"
        "names:\n"
        "  0: bike\n  1: bus\n  2: car\n  3: motor\n  4: person\n"
        "  5: rider\n  6: traffic light\n  7: traffic sign\n"
        "  8: train\n  9: truck\n",
        encoding="utf-8",
    )
    return str(yaml_path)


def train_subset_one_epoch(
    label_dir=None,
    image_dir=None,
    max_samples=500,
    batch_size=4,
    output_dir="outputs",
):
    """Run one training epoch on a small subset using Ultralytics trainer.

    Converts max_samples images to YOLO format then calls model.train
    with epochs=1. Demonstrates the full parse -> format -> train pipeline.

    Args:
        label_dir: BDD100K label directory. Reads LABEL_DIR env var if None.
        image_dir: BDD100K image root. Reads IMAGE_DIR env var if None.
        max_samples: Number of training images to use.
        batch_size: Images per batch.
        output_dir: Where to save run artefacts and final checkpoint.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("Install: pip install ultralytics")
        sys.exit(1)

    label_dir = label_dir or os.environ.get("LABEL_DIR", "/app/bdd100k_labels")
    image_dir = image_dir or os.environ.get("IMAGE_DIR", "/app/bdd100k")

    work_dir = Path(output_dir) / "subset_data"
    work_dir.mkdir(parents=True, exist_ok=True)

    log.info("Preparing subset (%d images) ...", max_samples)
    yaml_path = _prepare_subset_dataset(
        label_dir, image_dir, max_samples, work_dir
    )
    log.info("Dataset YAML: %s", yaml_path)

    log.info("Loading YOLOv8m ...")
    model = YOLO("yolov8m.pt")

    log.info("Training 1 epoch on %d images ...", max_samples)
    model.train(
        data=yaml_path,
        epochs=1,
        imgsz=640,
        batch=batch_size,
        project=str(Path(output_dir) / "runs"),
        name="yolov8_bdd_subset",
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        mosaic=0.5,
        fliplr=0.5,
        verbose=True,
    )

    best = (
        Path(output_dir)
        / "runs"
        / "yolov8_bdd_subset"
        / "weights"
        / "best.pt"
    )
    if best.exists():
        dst = Path(output_dir) / "yolov8m_bdd_subset_1epoch.pt"
        shutil.copy2(best, dst)
        log.info("Checkpoint saved to %s", dst)
    else:
        log.warning("best.pt not found at %s", best)

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 BDD100K training")
    parser.add_argument(
        "--mode", choices=["full", "subset"], default="subset",
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--output-dir",  default="outputs")
    parser.add_argument("--data-yaml",   default="configs/bdd100k.yaml")
    args = parser.parse_args()

    if args.mode == "full":
        train_ultralytics(
            data_yaml=args.data_yaml,
            epochs=args.epochs,
            batch=args.batch_size,
            output_dir=args.output_dir,
        )
    else:
        train_subset_one_epoch(
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
