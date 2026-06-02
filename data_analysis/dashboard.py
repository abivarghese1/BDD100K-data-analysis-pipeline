"""
dashboard.py
------------
Quantitative visualisation dashboard for BDD100K object detection statistics.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_dashboard(train_stats, val_stats, output_dir="outputs"):
    """Generate and save all four dashboard figures.

    Args:
        train_stats: Statistics dict from compute_stats for train split.
        val_stats: Statistics dict from compute_stats for val split.
        output_dir: Directory to write figure files into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    classes = sorted(train_stats["class_counts"].keys())
    train_counts = [train_stats["class_counts"][c] for c in classes]
    val_counts   = [val_stats["class_counts"][c]   for c in classes]
    x = np.arange(len(classes))
    w = 0.35

    # ── Figure 1: class distribution ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("BDD100K — class distribution", fontsize=14)

    axes[0].bar(x - w/2, train_counts, w, label="train", color="#378ADD")
    axes[0].bar(x + w/2, val_counts,   w, label="val",   color="#D85A30")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=40, ha="right")
    axes[0].set_title("annotation counts (train vs val)")
    axes[0].set_ylabel("count")
    axes[0].legend()

    ratios = [v / (t + v) * 100 if (t + v) > 0 else 0
              for t, v in zip(train_counts, val_counts)]
    axes[1].barh(classes, ratios, color="#1D9E75")
    axes[1].axvline(12.5, color="gray", linestyle="--", linewidth=0.8,
                    label="expected 12.5%")
    axes[1].set_title("val share % per class")
    axes[1].set_xlabel("val %")
    axes[1].legend()

    axes[2].hist(train_stats["annotations_per_image"], bins=40,
                 color="#7F77DD", alpha=0.8, label="train")
    axes[2].hist(val_stats["annotations_per_image"], bins=40,
                 color="#D4537E", alpha=0.6, label="val")
    axes[2].set_title("annotations per image")
    axes[2].set_xlabel("count")
    axes[2].set_ylabel("images")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out / "fig1_distribution.png", dpi=150)
    plt.close("all")
    print(f"  saved {out}/fig1_distribution.png")

    # ── Figure 2: bbox area distributions ────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("BDD100K — bounding box area distributions per class",
                 fontsize=14)

    for i, cls in enumerate(classes):
        ax = axes[i // 5][i % 5]
        areas = train_stats["bbox_areas"][cls]
        ax.hist(np.log1p(areas), bins=40, color="#378ADD")
        ax.set_title(cls, fontsize=11)
        ax.set_xlabel("log(area)")
        if areas:
            median = np.median(areas)
            ax.axvline(np.log1p(median), color="red",
                       linestyle="--", linewidth=0.8)
            ax.text(0.98, 0.95, f"med={median:.0f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="red")

    plt.tight_layout()
    plt.savefig(out / "fig2_bbox_areas.png", dpi=150)
    plt.close("all")
    print(f"  saved {out}/fig2_bbox_areas.png")

    # ── Figure 3: co-occurrence heatmap ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    matrix = np.zeros((len(classes), len(classes)))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            matrix[i][j] = train_stats["cooccurrence"][a][b]
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, where=row_sums != 0)
    sns.heatmap(norm, xticklabels=classes, yticklabels=classes,
                cmap="Blues", ax=ax, fmt=".2f", annot=True,
                annot_kws={"size": 8})
    ax.set_title("Class co-occurrence (normalised by row) — train")
    plt.tight_layout()
    plt.savefig(out / "fig3_cooccurrence.png", dpi=150)
    plt.close("all")
    print(f"  saved {out}/fig3_cooccurrence.png")

    # ── Figure 4: scene attribute breakdown ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("BDD100K — scene attribute distribution (train)",
                 fontsize=14)

    for ax, (attr, key) in zip(axes, [
        ("weather",    "weather_counts"),
        ("scene",      "scene_counts"),
        ("time of day","timeofday_counts"),
    ]):
        counts = train_stats[key]
        ax.pie(counts.values(), labels=counts.keys(),
               autopct="%1.1f%%", startangle=90)
        ax.set_title(attr)

    plt.tight_layout()
    plt.savefig(out / "fig4_attributes.png", dpi=150)
    plt.close("all")
    print(f"  saved {out}/fig4_attributes.png")
