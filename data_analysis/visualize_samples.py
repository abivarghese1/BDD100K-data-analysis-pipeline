"""
visualize_samples.py
--------------------
Qualitative visualisation of interesting and anomalous samples from the
BDD100K object detection dataset.
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

CLASS_COLORS = {
    "car": "#378ADD",
    "truck": "#D85A30",
    "bus": "#1D9E75",
    "person": "#D4537E",
    "rider": "#7F77DD",
    "bike": "#EF9F27",
    "motor": "#5DCAA5",
    "traffic light": "#E24B4A",
    "traffic sign": "#639922",
    "train": "#888780",
}

# Original BDD100K image resolution
_ORIG_W, _ORIG_H = 1280, 720
# Thumbnail size passed to matplotlib (keeps memory and canvas size sane)
_THUMB = (320, 180)


def draw_frame(ax, frame, title=""):
    """Draw one image with bounding box overlays onto a matplotlib Axes.

    Args:
        ax: Matplotlib Axes to draw onto.
        frame: :class:`~bdd_parser.Frame` whose image and annotations to draw.
        title: Title string shown above the axes.
    """
    img = None
    try:
        img = Image.open(frame.image_path).convert("RGB")
        img.thumbnail(_THUMB)
        ax.imshow(img)
    except FileNotFoundError:
        ax.set_facecolor("#f0f0f0")
        ax.text(
            0.5, 0.5, "image not found",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=8,
        )

    if img is not None:
        thumb_w, thumb_h = img.size
        sx = thumb_w / _ORIG_W
        sy = thumb_h / _ORIG_H

        for ann in frame.annotations:
            b = ann.box
            color = CLASS_COLORS.get(ann.category, "white")
            rect = patches.Rectangle(
                (b.x1 * sx, b.y1 * sy),
                b.width * sx,
                b.height * sy,
                linewidth=1.2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                b.x1 * sx, b.y1 * sy - 2,
                ann.category,
                fontsize=6,
                color=color,
                backgroundcolor=(0, 0, 0, 0.4),
            )

    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _save(fig, path):
    """Save a figure to *path* and close it.

    Args:
        fig: Matplotlib Figure to save.
        path: Destination :class:`~pathlib.Path`.
    """
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def visualize_interesting_samples(data, anomalies, output_dir="outputs"):
    """Generate and save four qualitative sample figures.

    Figures produced:

    - ``samples_crowded.png``    — images with the most annotations.
    - ``samples_small_boxes.png`` — images with near-invisible objects.
    - ``samples_train_class.png`` — examples of the rarest class.
    - ``samples_per_class.png``  — one representative image per class.

    Args:
        data: Dict with ``"train"`` and ``"val"`` lists of
              :class:`~bdd_parser.Frame` objects.
        anomalies: Anomaly dict from
                   :func:`~bdd_analysis.find_anomalies`.
        output_dir: Directory to write figure files into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_frames = {f.name: f for f in data["train"]}
    crowded = sorted(anomalies["crowded_images"], key=lambda x: -x[1])[:6]
    small = anomalies["very_small_boxes"][:6]
    rare = [
        f for f in data["train"]
        if any(a.category == "train" for a in f.annotations)
    ][:6]
    classes = list(CLASS_COLORS.keys())

    # ── crowded images ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Most crowded images (most annotations)", fontsize=12)
    for ax, (name, cnt) in zip(axes.flat, crowded):
        draw_frame(ax, train_frames[name], f"{name[:18]}…  {cnt} anns")
    for ax in axes.flat:
        ax.axis("off")
    _save(fig, out / "samples_crowded.png")

    # ── very small boxes ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Very small bounding boxes (area < 100 px²)", fontsize=12)
    for ax, (name, cat, area) in zip(axes.flat, small):
        draw_frame(ax, train_frames[name], f"{cat}  area={area:.0f}px²")
    for ax in axes.flat:
        ax.axis("off")
    _save(fig, out / "samples_small_boxes.png")

    # ── rare class: train ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Samples containing 'train' class (rarest category)",
                 fontsize=12)
    for ax, frame in zip(axes.flat, rare):
        draw_frame(ax, frame, frame.name[:22])
    for ax in axes.flat:
        ax.axis("off")
    _save(fig, out / "samples_train_class.png")

    # ── one sample per class ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle("Representative sample per detection class", fontsize=12)
    for ax, cls in zip(axes.flat, classes):
        candidates = [
            f for f in data["train"]
            if any(a.category == cls for a in f.annotations)
        ]
        if candidates:
            frame = random.choice(candidates[:200])
            draw_frame(ax, frame, cls)
    for ax in axes.flat:
        ax.axis("off")
    _save(fig, out / "samples_per_class.png")
