import zipfile
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image
from ultralytics import YOLO

from snowpole_detector.settings import get_settings


class InferenceMode(StrEnum):
    normal = "normal"
    split = "split"  # type: ignore[assignment]


def _zip_labels(labels_dir: Path, output_dir: Path) -> Path:
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(labels_dir.glob("*.txt")):
            zf.write(txt_file, txt_file.name)
    count = sum(1 for _ in labels_dir.glob("*.txt"))
    print(f"Zipped {count} prediction files -> {zip_path}")
    return zip_path


def predict_normal(model: YOLO, test_dir: Path, output_dir: Path) -> Path:
    """
    Run inference directly on full images using Ultralytics predict.
    Returns the labels directory.
    """
    model.predict(
        source=str(test_dir),
        project=str(output_dir),
        name="labels_raw",
        save_txt=True,
        save_conf=True,
        exist_ok=True,
    )
    return output_dir / "labels_raw" / "labels"


def predict_split(model: YOLO, test_dir: Path, output_dir: Path) -> Path:
    """
    Split each image into left/right halves, run inference on each half, then remap
    detections back to full-image YOLO coordinates.

    This mirrors the training pipeline in ds_split_images.py, which splits images
    horizontally before training. At inference time we split the same way, run the
    model on both halves, and stitch results back into full-image coordinates.

    Coordinate mapping (half -> full image):
      Left half:
        x_center_full = x_center_half / 2       width_full = width_half / 2
      Right half:
        x_center_full = 0.5 + x_center_half / 2  width_full = width_half / 2
      y_center and height are unchanged in both cases.
    """
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png"}
    img_files = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in img_extensions)

    for img_path in img_files:
        img = Image.open(img_path)
        W, H = img.size

        left_arr = np.array(img.crop((0, 0, W // 2, H)))
        right_arr = np.array(img.crop((W // 2, 0, W, H)))

        left_results = model.predict(source=left_arr, verbose=False)[0]
        right_results = model.predict(source=right_arr, verbose=False)[0]

        detections: list[tuple[int, float, float, float, float, float]] = []

        if left_results.boxes is not None and len(left_results.boxes):
            for box in left_results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x_c, y_c, w, h = box.xywhn[0].tolist()
                detections.append((cls, x_c / 2, y_c, w / 2, h, conf))

        if right_results.boxes is not None and len(right_results.boxes):
            for box in right_results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x_c, y_c, w, h = box.xywhn[0].tolist()
                detections.append((cls, 0.5 + x_c / 2, y_c, w / 2, h, conf))

        with open(labels_dir / f"{img_path.stem}.txt", "w") as f:
            for cls, x_c, y_c, w, h, conf in detections:
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

        print(f"  {img_path.name}: {len(detections)} detection(s)")

    return labels_dir


def run_inference(
    mode: Annotated[
        InferenceMode,
        typer.Option(
            help="'normal': inference on full images. 'split': split each image in half (matches split-trained model)."
        ),
    ] = InferenceMode.normal,
    model_path: Annotated[
        Path | None,
        typer.Option(help="Path to YOLO .pt weights (default: settings.MODEL_PATH)"),
    ] = None,
    test_dir: Annotated[
        Path | None,
        typer.Option(help="Directory of test images (default: settings.TEST_DIR)"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Output directory (default: settings.INFERENCE_OUTPUT)"),
    ] = None,
    zip_output: Annotated[
        bool,
        typer.Option(help="Create submission.zip from label files"),
    ] = True,
) -> None:
    settings = get_settings()

    resolved_model_path = model_path or settings.MODEL_PATH
    resolved_test_dir = test_dir or settings.TEST_DIR
    resolved_output_dir = output_dir or settings.INFERENCE_OUTPUT

    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inference mode : {mode}")
    print(f"Model          : {resolved_model_path}")
    print(f"Test dir       : {resolved_test_dir}")
    print(f"Output dir     : {resolved_output_dir}")

    yolo_model = YOLO(str(resolved_model_path))

    if mode == InferenceMode.normal:
        labels_dir = predict_normal(yolo_model, resolved_test_dir, resolved_output_dir)
    else:
        labels_dir = predict_split(yolo_model, resolved_test_dir, resolved_output_dir)

    if zip_output:
        _zip_labels(labels_dir, resolved_output_dir)


if __name__ == "__main__":
    typer.run(run_inference)
