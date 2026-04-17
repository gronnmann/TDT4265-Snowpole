"""
Confidence threshold analysis on the validation set.

Runs model inference at very low confidence (0.001) to collect all candidate
detections, then sweeps thresholds to find the value that maximises F1
(or a precision/recall trade-off you choose).

Usage:
    python -m snowpole_detector.tune_threshold
    python -m snowpole_detector.tune_threshold --model-path runs/detect/train/weights/best.pt
    python -m snowpole_detector.tune_threshold --iou-threshold 0.5 --output threshold_plot.png
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from ultralytics import YOLO

from snowpole_detector.settings import get_settings


def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU between two [x1, y1, x2, y2] pixel boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _yolo_to_xyxy(x_c: float, y_c: float, w: float, h: float, W: int, H: int) -> np.ndarray:
    return np.array([
        (x_c - w / 2) * W,
        (y_c - h / 2) * H,
        (x_c + w / 2) * W,
        (y_c + h / 2) * H,
    ])


def _eval_at_threshold(
    all_preds: list[tuple[np.ndarray, np.ndarray]],
    all_gts: list[np.ndarray],
    threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Return (TP, FP, FN) for the whole dataset at a given confidence threshold."""
    tp = fp = fn = 0
    for (pred_boxes, pred_confs), gt_boxes in zip(all_preds, all_gts):
        mask = pred_confs >= threshold
        preds = pred_boxes[mask]
        n_gt = len(gt_boxes)
        n_pred = len(preds)

        if n_gt == 0 and n_pred == 0:
            continue
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue

        matched_gt: set[int] = set()
        for pred in preds:
            best_iou = iou_threshold  # must exceed this to count as TP
            best_j = -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = _iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += n_gt - len(matched_gt)

    return tp, fp, fn


def main(
    model_path: Annotated[
        Path | None,
        typer.Option(help="Path to YOLO .pt weights (default: settings.MODEL_PATH)"),
    ] = None,
    val_dir: Annotated[
        Path | None,
        typer.Option(help="Validation images dir (default: DATASET_RESULTS/images/valid)"),
    ] = None,
    labels_dir: Annotated[
        Path | None,
        typer.Option(help="Validation labels dir (default: DATASET_RESULTS/labels/valid)"),
    ] = None,
    iou_threshold: Annotated[
        float,
        typer.Option(help="IoU threshold used to count a prediction as TP"),
    ] = 0.5,
    output: Annotated[
        Path,
        typer.Option(help="Output plot path"),
    ] = Path("threshold_analysis.png"),
) -> None:
    settings = get_settings()

    resolved_model = model_path or settings.MODEL_PATH
    resolved_val_dir = val_dir or (settings.DATASET_PREPROCESS_PATH / "valid" / "images")
    resolved_labels_dir = labels_dir or (settings.DATASET_PREPROCESS_PATH / "valid" / "labels")

    print(f"Model      : {resolved_model}")
    print(f"Val images : {resolved_val_dir}")
    print(f"Val labels : {resolved_labels_dir}")
    print(f"IoU thresh : {iou_threshold}")

    model = YOLO(str(resolved_model))

    img_paths = sorted(
        p for p in resolved_val_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"\nRunning inference at conf=0.001 on {len(img_paths)} images...")

    all_preds: list[tuple[np.ndarray, np.ndarray]] = []
    all_gts: list[np.ndarray] = []

    for img_path in img_paths:
        result = model.predict(source=str(img_path), conf=0.001, verbose=False)[0]
        H, W = result.orig_shape

        if result.boxes is not None and len(result.boxes):
            boxes_xyxy: np.ndarray = result.boxes.xyxy.cpu().numpy()
            confs: np.ndarray = result.boxes.conf.cpu().numpy()
        else:
            boxes_xyxy = np.zeros((0, 4))
            confs = np.zeros(0)
        all_preds.append((boxes_xyxy, confs))

        label_path = resolved_labels_dir / f"{img_path.stem}.txt"
        gt_list: list[np.ndarray] = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        _, x_c, y_c, w, h = parts[:5]
                        gt_list.append(_yolo_to_xyxy(x_c, y_c, w, h, W, H))
        all_gts.append(np.array(gt_list) if gt_list else np.zeros((0, 4)))

    # Sweep confidence thresholds
    thresholds = np.linspace(0.01, 0.95, 300)
    precisions = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))
    f1s = np.zeros(len(thresholds))

    print("Sweeping thresholds...")
    for i, t in enumerate(thresholds):
        tp, fp, fn = _eval_at_threshold(all_preds, all_gts, t, iou_threshold)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions[i] = p
        recalls[i] = r
        f1s[i] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    best_idx = int(np.argmax(f1s))
    best_thresh = float(thresholds[best_idx])

    print(f"\n{'='*45}")
    print(f"Best threshold (max F1 @ IoU={iou_threshold}): {best_thresh:.3f}")
    print(f"  Precision : {precisions[best_idx]:.4f}")
    print(f"  Recall    : {recalls[best_idx]:.4f}")
    print(f"  F1        : {f1s[best_idx]:.4f}")
    print(f"{'='*45}")

    # Print table of notable thresholds
    print(f"\n{'Conf':>6}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}")
    print("-" * 36)
    notable = sorted({0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, best_thresh})
    for t_val in notable:
        idx = int(np.argmin(np.abs(thresholds - t_val)))
        marker = " <-- best F1" if abs(thresholds[idx] - best_thresh) < 1e-4 else ""
        print(
            f"{thresholds[idx]:>6.3f}  {precisions[idx]:>9.4f}  "
            f"{recalls[idx]:>7.4f}  {f1s[idx]:>7.4f}{marker}"
        )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P / R / F1 vs threshold
    ax = axes[0]
    ax.plot(thresholds, precisions, label="Precision", color="steelblue")
    ax.plot(thresholds, recalls, label="Recall", color="darkorange")
    ax.plot(thresholds, f1s, label="F1", color="green", linewidth=2)
    ax.axvline(best_thresh, color="red", linestyle="--",
               label=f"Best F1  conf={best_thresh:.3f}")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 vs Confidence  (IoU={iou_threshold})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Right: Precision-Recall curve
    ax2 = axes[1]
    ax2.plot(recalls, precisions, color="purple")
    ax2.scatter(
        [recalls[best_idx]], [precisions[best_idx]],
        color="red", zorder=5, s=80,
        label=f"Best F1  conf={best_thresh:.3f}",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(str(output), dpi=150)
    print(f"\nPlot saved to {output}")


if __name__ == "__main__":
    typer.run(main)
