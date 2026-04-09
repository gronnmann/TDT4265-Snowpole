from pathlib import Path
from typing import TypedDict

import typer
from PIL import Image

from snowpole_detector.settings import get_settings

type YoloBboxList = list[YoloBbox]
type XYXYBboxList = list[XYXYBbox]
type ImageShape = tuple[int, int, int]  # H x W x C


class YoloBbox(TypedDict):
    """YOLO formatted bounding box (normalized coordinates)."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class XYXYBbox(TypedDict):
    """Bounding box in pixel coordinates: [xmin, ymin, xmax, ymax]."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


def yolo_to_xyxy(bbox: YoloBbox, img_shape: ImageShape) -> XYXYBbox:
    """Convert YOLO bbox (normalized x_center, y_center, width, height) to XYXY in pixels."""
    H, W = img_shape[:2]
    x_c, y_c, w, h = bbox["x_center"], bbox["y_center"], bbox["width"], bbox["height"]
    return XYXYBbox(
        xmin=(x_c - w / 2) * W,
        ymin=(y_c - h / 2) * H,
        xmax=(x_c + w / 2) * W,
        ymax=(y_c + h / 2) * H,
    )


def xyxy_to_yolo(bbox: XYXYBbox, img_shape: ImageShape, class_id: int) -> YoloBbox:
    """Convert XYXY bbox in pixels to YOLO normalized format."""
    H, W = img_shape[:2]
    xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
    return YoloBbox(
        class_id=class_id,
        x_center=(xmin + xmax) / 2 / W,
        y_center=(ymin + ymax) / 2 / H,
        width=(xmax - xmin) / W,
        height=(ymax - ymin) / H,
    )


def _write_labels(label_path: Path, bboxes: YoloBboxList) -> None:
    with open(label_path, "w") as f:
        for bbox in bboxes:
            f.write(
                f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f}"
                f" {bbox['width']:.6f} {bbox['height']:.6f}\n"
            )


def split_image_and_labels(
    img_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_label_dir: Path,
) -> None:
    """Split one image horizontally into two halves, adjust YOLO labels, and save both."""
    img = Image.open(img_path)
    W, H = img.size
    img_shape: ImageShape = (H, W, len(img.getbands()))

    bboxes: YoloBboxList = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                class_id, x_c, y_c, w, h = map(float, line.strip().split())
                bboxes.append(
                    YoloBbox(
                        class_id=int(class_id),
                        x_center=x_c,
                        y_center=y_c,
                        width=w,
                        height=h,
                    )
                )

    xy_bboxes: XYXYBboxList = [yolo_to_xyxy(b, img_shape) for b in bboxes]
    half_shape: ImageShape = (H, W // 2, img_shape[2])

    # Left half
    left_bboxes: YoloBboxList = []
    for orig, bbox in zip(bboxes, xy_bboxes, strict=True):
        if bbox["xmax"] <= W / 2:
            left_bboxes.append(
                xyxy_to_yolo(
                    XYXYBbox(
                        xmin=bbox["xmin"], ymin=bbox["ymin"], xmax=bbox["xmax"], ymax=bbox["ymax"]
                    ),
                    half_shape,
                    orig["class_id"],
                )
            )
        elif bbox["xmin"] < W / 2:
            # Partial overlap: clip to left edge
            left_bboxes.append(
                xyxy_to_yolo(
                    XYXYBbox(xmin=bbox["xmin"], ymin=bbox["ymin"], xmax=W / 2, ymax=bbox["ymax"]),
                    half_shape,
                    orig["class_id"],
                )
            )

    img.crop((0, 0, W // 2, H)).save(out_img_dir / f"{img_path.stem}_left{img_path.suffix}")
    _write_labels(out_label_dir / f"{img_path.stem}_left.txt", left_bboxes)

    # Right half
    right_bboxes: YoloBboxList = []
    for orig, bbox in zip(bboxes, xy_bboxes, strict=True):
        if bbox["xmin"] >= W / 2:
            right_bboxes.append(
                xyxy_to_yolo(
                    XYXYBbox(
                        xmin=bbox["xmin"] - W / 2,
                        ymin=bbox["ymin"],
                        xmax=bbox["xmax"] - W / 2,
                        ymax=bbox["ymax"],
                    ),
                    half_shape,
                    orig["class_id"],
                )
            )
        elif bbox["xmax"] > W / 2:
            # Partial overlap: clip to right edge
            right_bboxes.append(
                xyxy_to_yolo(
                    XYXYBbox(
                        xmin=0, ymin=bbox["ymin"], xmax=bbox["xmax"] - W / 2, ymax=bbox["ymax"]
                    ),
                    half_shape,
                    orig["class_id"],
                )
            )

    img.crop((W // 2, 0, W, H)).save(out_img_dir / f"{img_path.stem}_right{img_path.suffix}")
    _write_labels(out_label_dir / f"{img_path.stem}_right.txt", right_bboxes)


def process_dataset() -> None:
    """Process entire YOLO dataset, splitting train/valid images horizontally."""
    settings = get_settings()
    dir_in = settings.DATASET_SOURCE
    dir_out = settings.DATASET_RESULTS
    dir_out.mkdir(parents=True, exist_ok=True)

    for subset in ["train", "valid"]:
        img_dir = dir_in / "images" / subset
        label_dir = dir_in / "labels" / subset
        out_img_dir = dir_out / "images" / subset
        out_label_dir = dir_out / "labels" / subset
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            print(f"Processing {img_file}")
            split_image_and_labels(
                img_file, label_dir / f"{img_file.stem}.txt", out_img_dir, out_label_dir
            )


if __name__ == "__main__":
    typer.run(process_dataset)
