# script to parse the dataset
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def aspect_ratio(self):
        return self.width / self.height if self.height > 0 else 0


@dataclass
class Annotation:
    label_id: int
    category: str
    occluded: bool
    truncated: bool
    box: BBox
    traffic_light_color: Optional[str] = None


@dataclass
class Frame:
    name: str
    split: str  # 'train' or 'val'
    weather: str
    scene: str
    timeofday: str
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def image_path(self):
        """Path to the image file inside the container.

        Controlled by the IMAGE_DIR environment variable.
        Defaults to ``bdd100k`` which matches the Docker volume mount.
        """
        import os
        base = os.environ.get("IMAGE_DIR", "bdd100k")
        return Path(base) / "images" / "100k" / self.split / self.name


DETECTION_CLASSES = {
    "car",
    "truck",
    "bus",
    "person",
    "rider",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
}


def parse_split(json_path: str, split: str) -> list[Frame]:
    with open(json_path) as f:
        raw = json.load(f)

    frames = []
    for entry in raw:
        attrs = entry.get("attributes", {})
        frame = Frame(
            name=entry["name"],
            split=split,
            weather=attrs.get("weather", "unknown"),
            scene=attrs.get("scene", "unknown"),
            timeofday=attrs.get("timeofday", "unknown"),
        )
        for lbl in entry.get("labels", []):
            cat = lbl.get("category", "")
            if cat not in DETECTION_CLASSES:
                continue  # skip segmentation labels
            b = lbl.get("box2d")
            if not b:
                continue  # skip if no bounding box
            ann = Annotation(
                label_id=lbl["id"],
                category=cat,
                occluded=lbl.get("attributes", {}).get("occluded", False),
                truncated=lbl.get("attributes", {}).get("truncated", False),
                box=BBox(b["x1"], b["y1"], b["x2"], b["y2"]),
                traffic_light_color=lbl.get("attributes", {}).get("trafficLightColor"),
            )
            frame.annotations.append(ann)
        frames.append(frame)
    return frames


def load_dataset(
    label_dir: str = None,
) -> dict:
    """Load both train and val splits of the BDD100K detection dataset.

    Args:
        label_dir: Directory containing the label JSON files.
                   Defaults to the LABEL_DIR environment variable,
                   or ``bdd100k_labels`` if unset.

    Returns:
        A dict with keys ``"train"`` and ``"val"``, each mapping to a
        list of :class:`Frame` objects.
    """
    import os
    if label_dir is None:
        label_dir = os.environ.get("LABEL_DIR", "bdd100k_labels")
    return {
        "train": parse_split(
            f"{label_dir}/bdd100k_labels_images_train.json", "train"
        ),
        "val": parse_split(
            f"{label_dir}/bdd100k_labels_images_val.json", "val"
        ),
    }
