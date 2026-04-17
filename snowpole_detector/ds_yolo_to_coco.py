import json
import shutil
from pathlib import Path

import typer
from PIL import Image

from snowpole_detector.ds_split_images import ImageShape, YoloBbox, YoloBboxList, yolo_to_xyxy
from snowpole_detector.settings import get_settings


def _load_yaml_classes(yaml_path: Path) -> list[str]:
    """Parse class names from data.yaml (simple line-by-line, no PyYAML dependency)."""
    names: list[str] = []
    in_names = False
    with open(yaml_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("names:"):
                in_names = True
                inline = stripped[len("names:"):].strip()
                if inline.startswith("["):
                    names = [n.strip().strip("'\"") for n in inline.strip("[]").split(",")]
                    break
                continue
            if in_names:
                if stripped.startswith("-"):
                    names.append(stripped.lstrip("- ").strip().strip("'\""))
                elif stripped and not stripped.startswith(" ") and not stripped.startswith("\t"):
                    break
    return names


def _load_yolo_labels(label_path: Path) -> YoloBboxList:
    bboxes: YoloBboxList = []
    if not label_path.exists():
        return bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_c, y_c, w, h = int(float(parts[0])), *map(float, parts[1:])
            bboxes.append(
                YoloBbox(class_id=class_id, x_center=x_c, y_center=y_c, width=w, height=h)
            )
    return bboxes


def convert_subset(
        img_dir: Path,
        label_dir: Path,
        class_names: list[str],
        output_dir: Path,  # changed: now a directory, not a json path
) -> None:
    """Convert one YOLO subset (train/valid/test) to a COCO annotations JSON, copying images alongside."""
    output_dir.mkdir(parents=True, exist_ok=True)

    coco: dict = {
        "info": {"description": "Converted from YOLO format", "version": "1.0"},
        "licenses": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "none"}
            for i, name in enumerate(class_names)
        ],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    image_files = sorted(
        f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for img_id, img_path in enumerate(image_files, start=1):
        img = Image.open(img_path)
        img_w, img_h = img.size
        img_shape: ImageShape = (img_h, img_w, len(img.getbands()))

        # Copy image to output dir
        dest = output_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)

        coco["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": img_w, "height": img_h}
        )

        label_path = label_dir / f"{img_path.stem}.txt"
        for bbox in _load_yolo_labels(label_path):
            xyxy = yolo_to_xyxy(bbox, img_shape)
            coco_w = xyxy["xmax"] - xyxy["xmin"]
            coco_h = xyxy["ymax"] - xyxy["ymin"]
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": bbox["class_id"],
                    "bbox": [xyxy["xmin"], xyxy["ymin"], coco_w, coco_h],
                    "area": coco_w * coco_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        print(f"  [{img_id}/{len(image_files)}] {img_path.name}")

    output_json = output_dir / "_annotations.coco.json"
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"  → Saved {output_json} "
        f"({len(coco['images'])} images, {len(coco['annotations'])} annotations)"
    )


def convert_dataset() -> None:
    """Convert entire YOLO dataset (train/valid/test) to COCO JSON format."""
    settings = get_settings()
    dir_in = settings.DATASET_FINISHED_YOLO
    dir_out = settings.DATASET_FINISHED_COCO

    yaml_path = dir_in / "data.yaml"
    class_names = _load_yaml_classes(yaml_path)
    if not class_names:
        raise ValueError(f"Could not parse class names from {yaml_path}")
    print(f"Classes: {class_names}")

    for subset in ["train", "valid"]:
        img_dir = dir_in / subset / "images"
        label_dir = dir_in / subset / "labels"
        if not img_dir.exists():
            print(f"Skipping '{subset}' (no images dir found)")
            continue

        output_dir = dir_out / subset
        print(f"\nConverting '{subset}'...")
        convert_subset(img_dir, label_dir, class_names, output_dir)

    print("\nDone. COCO dataset written to:", dir_out)

if __name__ == "__main__":
    typer.run(convert_dataset)