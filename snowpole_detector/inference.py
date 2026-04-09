import zipfile
from enum import Enum
from pathlib import Path

import numpy as np
import typer
from PIL import Image
from ultralytics import YOLO

from snowpole_detector.settings import get_settings


class InferenceMode(str, Enum):
    normal = "normal"
    split = "split"


def _zip_labels(labels_dir: Path, output_dir: Path) -> Path:
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(labels_dir.glob("*.txt")):
            zf.write(txt_file, txt_file.name)
    print(f"Zipped {sum(1 for _ in labels_dir.glob('*.txt'))} prediction files -> {zip_path}")
    return zip_path


def predict_normal(model: YOLO, test_dir: Path, output_dir: Path) -> Path:
    """
    Run inference directly on full images using Ultralytics predict.
    Returns the labels directory.
    """
    results = model.predict(
        source=str(test_dir),
        project=str(output_dir),
        name="labels_raw",
        save_txt=True,
        save_conf=True,
        exist_ok=True,
    )
    labels_dir = output_dir / "labels_raw" / "labels"
    return labels_dir


def predict_split(model: YOLO, test_dir: Path, output_dir: Path) -> Path:
    """
    Split each image into left/right halves, run inference on each half,
    then remap detections back to full-image YOLO coordinates.

    This mirrors the training pipeline in ds_split_images.py, which split
    images horizontally before training. At inference time we therefore split
    the same way, run the model on both halves, and stitch results back.

    Coordinate mapping (half -> full):
      Left half:
        x_center_full = x_center_half / 2
        y_center_full = y_center_half          (unchanged)
        width_full    = width_half / 2
        height_full   = height_half            (unchanged)
      Right half:
        x_center_full = 0.5 + x_center_half / 2
        y_center_full = y_center_half          (unchanged)
        width_full    = width_half / 2
        height_full   = height_half            (unchanged)
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

        detections: list[tuple] = []

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

        label_path = labels_dir / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            for cls, x_c, y_c, w, h, conf in detections:
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

        print(f"  {img_path.name}: {len(detections)} detection(s)")

    return labels_dir


def run_inference(
    mode: InferenceMode = typer.Option(
        InferenceMode.normal,
        help="'normal': inference on full images. 'split': split each image in half (matches split-trained model).",
    ),
    model_path: Path = typer.Option(None, help="Path to YOLO .pt weights (default: settings.MODEL_PATH)"),
    test_dir: Path = typer.Option(None, help="Directory of test images (default: settings.TEST_DIR)"),
    output_dir: Path = typer.Option(None, help="Output directory (default: settings.INFERENCE_OUTPUT)"),
    zip_output: bool = typer.Option(True, help="Create submission.zip from label files"),
) -> None:
    settings = get_settings()

    model_path = model_path or settings.MODEL_PATH
    test_dir = test_dir or settings.TEST_DIR
    output_dir = output_dir or settings.INFERENCE_OUTPUT

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inference mode : {mode.value}")
    print(f"Model          : {model_path}")
    print(f"Test dir       : {test_dir}")
    print(f"Output dir     : {output_dir}")

    model = YOLO(str(model_path))

    if mode == InferenceMode.normal:
        labels_dir = predict_normal(model, Path(test_dir), output_dir)
    else:
        labels_dir = predict_split(model, Path(test_dir), output_dir)

    if zip_output:
        _zip_labels(labels_dir, output_dir)


if __name__ == "__main__":
    typer.run(run_inference)
