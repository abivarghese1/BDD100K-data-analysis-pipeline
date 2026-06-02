from collections import defaultdict, Counter
import numpy as np
from data_analysis.bdd_parser import load_dataset, DETECTION_CLASSES


def compute_stats(frames: list) -> dict:
    stats = {
        "total_images": len(frames),
        "total_annotations": 0,
        "class_counts": Counter(),
        "images_with_class": Counter(),  # how many images contain >= 1 of class
        "annotations_per_image": [],
        "bbox_areas": defaultdict(list),
        "bbox_aspect_ratios": defaultdict(list),
        "bbox_widths": defaultdict(list),
        "bbox_heights": defaultdict(list),
        "occluded_counts": Counter(),
        "truncated_counts": Counter(),
        "weather_counts": Counter(),
        "scene_counts": Counter(),
        "timeofday_counts": Counter(),
        "cooccurrence": defaultdict(
            Counter
        ),  # class A appears in same image as class B
        "anns_per_image_by_class": defaultdict(list),
    }

    for frame in frames:
        cats_in_frame = [a.category for a in frame.annotations]
        stats["annotations_per_image"].append(len(cats_in_frame))
        stats["weather_counts"][frame.weather] += 1
        stats["scene_counts"][frame.scene] += 1
        stats["timeofday_counts"][frame.timeofday] += 1

        seen_in_frame = set()
        cat_count_in_frame = Counter(cats_in_frame)

        for ann in frame.annotations:
            c = ann.category
            stats["total_annotations"] += 1
            stats["class_counts"][c] += 1
            stats["bbox_areas"][c].append(ann.box.area)
            stats["bbox_aspect_ratios"][c].append(ann.box.aspect_ratio)
            stats["bbox_widths"][c].append(ann.box.width)
            stats["bbox_heights"][c].append(ann.box.height)
            if ann.occluded:
                stats["occluded_counts"][c] += 1
            if ann.truncated:
                stats["truncated_counts"][c] += 1
            seen_in_frame.add(c)

        for c in seen_in_frame:
            stats["images_with_class"][c] += 1
            stats["anns_per_image_by_class"][c].append(cat_count_in_frame[c])

        # co-occurrence matrix
        seen = list(seen_in_frame)
        for i, a in enumerate(seen):
            for b in seen:
                if a != b:
                    stats["cooccurrence"][a][b] += 1

    return stats


def find_anomalies(frames: list, stats: dict) -> dict:
    anomalies = {
        "very_small_boxes": [],  # area < 100 px²
        "very_large_boxes": [],  # area > 100,000 px²
        "extreme_aspect": [],  # aspect ratio > 10 or < 0.1
        "crowded_images": [],  # > 30 annotations in one image
        "empty_images": [],  # 0 detection annotations
        "rare_class_combos": [],  # image with 'train' or 'motor' class
    }
    for frame in frames:
        n = len(frame.annotations)
        if n == 0:
            anomalies["empty_images"].append(frame.name)
        elif n > 30:
            anomalies["crowded_images"].append((frame.name, n))
        for ann in frame.annotations:
            if ann.box.area < 100:
                anomalies["very_small_boxes"].append(
                    (frame.name, ann.category, ann.box.area)
                )
            if ann.box.area > 100_000:
                anomalies["very_large_boxes"].append(
                    (frame.name, ann.category, ann.box.area)
                )
            if ann.box.aspect_ratio > 10 or (
                ann.box.aspect_ratio < 0.1 and ann.box.aspect_ratio > 0
            ):
                anomalies["extreme_aspect"].append(
                    (frame.name, ann.category, ann.box.aspect_ratio)
                )
        cats = {a.category for a in frame.annotations}
        if "train" in cats or "motor" in cats:
            anomalies["rare_class_combos"].append((frame.name, list(cats)))
    return anomalies
