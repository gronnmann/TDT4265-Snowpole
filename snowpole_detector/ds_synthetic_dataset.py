"""
ds_create_copied.py

Creates a synthetic dataset by:
1. Extracting pole crops from training images using SAM2 segmentation
2. Pasting segmented poles onto random background images
3. Saving new images + YOLO labels

Pole extraction results are cached to disk so SAM2 only runs once.
The cache is invalidated automatically if the source images/labels change.

Setup:
    pip install sam2
    mkdir checkpoints && cd checkpoints
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

Commands:
    python ds_create_copied.py create --num-synthetic 300
    python ds_create_copied.py merge ./dataset-copied
    python ds_create_copied.py create --num-synthetic 300 --merge
    python ds_create_copied.py clear-cache          # wipe cached poles
"""

import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from typing_extensions import Annotated

from snowpole_detector.settings import get_settings

app = typer.Typer()

# ---------------------------------------------------------------------------
# Pole cache — saves crops + masks to disk so SAM2 only runs once
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(".pole_cache")
_CACHE_MANIFEST = _CACHE_DIR / "manifest.json"


def _source_fingerprint(img_dir: Path, lbl_dir: Path) -> str:
    """
    Stable fingerprint of every (image, label) pair in the source split.
    Changes whenever files are added, removed, or modified (mtime + size).
    Fast — no file content is hashed.
    """
    entries = []
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img_stat = img_path.stat()
        lbl_mtime = lbl_path.stat().st_mtime if lbl_path.exists() else 0
        lbl_size  = lbl_path.stat().st_size  if lbl_path.exists() else 0
        entries.append(
            f"{img_path.name}:{img_stat.st_mtime}:{img_stat.st_size}"
            f":{lbl_mtime}:{lbl_size}"
        )
    return hashlib.md5("\n".join(entries).encode()).hexdigest()


def _load_pole_cache(img_dir: Path, lbl_dir: Path) -> list[dict] | None:
    """Return cached poles if the cache exists and matches current sources, else None."""
    if not _CACHE_MANIFEST.exists():
        return None

    manifest = json.loads(_CACHE_MANIFEST.read_text())
    if manifest.get("fingerprint") != _source_fingerprint(img_dir, lbl_dir):
        print("  Cache fingerprint mismatch — will re-segment.")
        return None

    poles = []
    for entry in manifest["poles"]:
        crop = cv2.imread(str(_CACHE_DIR / entry["crop_file"]))
        mask = cv2.imread(str(_CACHE_DIR / entry["mask_file"]), cv2.IMREAD_GRAYSCALE)
        if crop is None or mask is None:
            print("  Cache file missing — will re-segment.")
            return None
        poles.append({
            "crop_bgr": crop,
            "mask": mask,
            "src_cy_norm": entry["src_cy_norm"],
        })

    print(f"  Loaded {len(poles)} poles from cache.")
    return poles


def _save_pole_cache(poles: list[dict], img_dir: Path, lbl_dir: Path) -> None:
    """Persist poles to disk and write a manifest with the source fingerprint."""
    _CACHE_DIR.mkdir(exist_ok=True)

    # Wipe stale files first
    for f in _CACHE_DIR.glob("pole_*"):
        f.unlink()

    manifest_entries = []
    for i, pole in enumerate(poles):
        crop_file = f"pole_{i:05d}_crop.png"
        mask_file = f"pole_{i:05d}_mask.png"
        cv2.imwrite(str(_CACHE_DIR / crop_file), pole["crop_bgr"])
        cv2.imwrite(str(_CACHE_DIR / mask_file), pole["mask"])
        manifest_entries.append({
            "crop_file": crop_file,
            "mask_file": mask_file,
            "src_cy_norm": pole["src_cy_norm"],
        })

    manifest = {
        "fingerprint": _source_fingerprint(img_dir, lbl_dir),
        "poles": manifest_entries,
    }
    _CACHE_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"  Cached {len(poles)} poles to {_CACHE_DIR}/")


# ---------------------------------------------------------------------------
# SAM2 predictor (loaded once, reused for all images)
# ---------------------------------------------------------------------------

def load_sam2_predictor():
    """
    Load SAM2 image predictor with bbox prompting.
    sam2.1_hiera_large = best quality. Use _small/_tiny if VRAM is limited.
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = Path("./sam2.1_hiera_large.pt")
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {checkpoint}.\n"
            "Download it:\n"
            "  mkdir checkpoints && cd checkpoints\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        )

    model = build_sam2(config, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    print(f"  SAM2 loaded on {device}")
    return predictor


# ---------------------------------------------------------------------------
# Step 1: Extract segmented pole crops from labeled training data
# ---------------------------------------------------------------------------

def segment_pole_sam2(
    predictor,
    image_rgb: np.ndarray,
    bbox_yolo: list[float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Segment a pole using SAM2 with a YOLO bbox as the box prompt.
    Returns (crop_bgr, mask_crop) tightly cropped to the bbox, or None on failure.
    """
    h, w = image_rgb.shape[:2]
    cx, cy, bw, bh = bbox_yolo

    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w - 1, int((cx + bw / 2) * w))
    y2 = min(h - 1, int((cy + bh / 2) * h))

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    try:
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2], dtype=np.float32),
            multimask_output=True,   # 3 candidates — pick the best scoring one
        )
        mask_full = masks[scores.argmax()].astype(np.uint8) * 255
    except Exception as e:
        print(f"    SAM2 failed ({e}) — skipping pole")
        return None

    crop_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)[y1:y2, x1:x2]
    mask_crop = mask_full[y1:y2, x1:x2]

    if crop_bgr.size == 0 or mask_crop.sum() == 0:
        return None

    return crop_bgr.copy(), mask_crop.copy()


def extract_all_poles(img_dir: Path, lbl_dir: Path) -> list[dict]:
    """
    Extract all labeled pole instances from img_dir using SAM2.
    Results are cached to .pole_cache/ and reused on subsequent runs.
    """
    # --- Cache hit ---
    cached = _load_pole_cache(img_dir, lbl_dir)
    if cached is not None:
        return cached

    # --- Cache miss: run SAM2 ---
    print("  No valid cache found — running SAM2 segmentation...")
    predictor = load_sam2_predictor()
    poles = []
    img_paths = [p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"  Scanning {len(img_paths)} images...")

    for img_path in img_paths:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for line in lbl_path.read_text().splitlines():
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5:
                continue

            bbox = parts[1:5]  # cx cy w h
            result = segment_pole_sam2(predictor, image_rgb, bbox)
            if result is None:
                continue

            crop, mask = result
            poles.append({
                "crop_bgr": crop,
                "mask": mask,
                "src_cy_norm": bbox[1],   # used for depth-aware scaling
            })

    print(f"  Got {len(poles)} pole instances")
    _save_pole_cache(poles, img_dir, lbl_dir)
    return poles


# ---------------------------------------------------------------------------
# Step 2: Paste segmented poles onto backgrounds
# ---------------------------------------------------------------------------

def paste_pole(bg: np.ndarray, pole: dict) -> tuple[np.ndarray, list[float]] | None:
    """
    Paste a single SAM2-segmented pole onto a background.
    Returns (result_bgr, yolo_bbox) or None if placement is out of bounds.
    """
    result = bg.copy()
    bh, bw = result.shape[:2]

    crop = pole["crop_bgr"].copy()
    mask = pole["mask"].copy()
    src_cy = pole["src_cy_norm"]

    # Place in lower 65% of image — road area, not sky
    paste_cy_norm = random.uniform(0.35, 0.95)
    paste_cx_norm = random.uniform(0.08, 0.92)

    # Depth-aware scale: higher in image = further away = smaller pole
    scale = np.clip(paste_cy_norm / max(src_cy, 0.05), 0.3, 2.5) * random.uniform(0.85, 1.15)
    new_h = max(10, int(crop.shape[0] * scale))
    new_w = max(4, int(crop.shape[1] * scale))
    crop = cv2.resize(crop, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    x1 = int(paste_cx_norm * bw) - new_w // 2
    y1 = int(paste_cy_norm * bh) - new_h // 2

    # Clamp to image bounds
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(bw, x1 + new_w), min(bh, y1 + new_h)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    cx1, cy1 = rx1 - x1, ry1 - y1
    crop_region = crop[cy1: cy1 + (ry2 - ry1), cx1: cx1 + (rx2 - rx1)]
    mask_region = mask[cy1: cy1 + (ry2 - ry1), cx1: cx1 + (rx2 - rx1)]

    # Poisson blending (seamlessly reconciles lighting); alpha-blend fallback
    pasted = False
    if mask_region.sum() > 500:
        try:
            pole_full = np.zeros_like(result)
            pole_full[ry1:ry2, rx1:rx2] = crop_region
            mask_full = np.zeros(result.shape[:2], dtype=np.uint8)
            mask_full[ry1:ry2, rx1:rx2] = mask_region
            center = (rx1 + (rx2 - rx1) // 2, ry1 + (ry2 - ry1) // 2)
            result = cv2.seamlessClone(pole_full, result, mask_full, center, cv2.NORMAL_CLONE)
            pasted = True
        except cv2.error:
            pass

    if not pasted:
        ks = max(3, min(new_w, new_h) // 6 * 2 + 1)
        alpha = cv2.GaussianBlur(mask_region.astype(np.float32), (ks, ks), 0)[..., None] / 255.0
        roi = result[ry1:ry2, rx1:rx2].astype(np.float32)
        result[ry1:ry2, rx1:rx2] = np.clip(
            alpha * crop_region + (1 - alpha) * roi, 0, 255
        ).astype(np.uint8)

    bbox = [
        (rx1 + rx2) / 2 / bw,
        (ry1 + ry2) / 2 / bh,
        (rx2 - rx1) / bw,
        (ry2 - ry1) / bh,
    ]
    return result, bbox


# ---------------------------------------------------------------------------
# Dataset creation & merging
# ---------------------------------------------------------------------------

def create_dataset(output_dir: Path, source_dir: Path, bg_dir: Path, num_synthetic: int) -> None:
    img_out = output_dir / "train" / "images"
    lbl_out = output_dir / "train" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    yaml_src = source_dir / "data.yaml"
    if yaml_src.exists():
        shutil.copy(yaml_src, output_dir / "data.yaml")

    print("Extracting pole instances with SAM2...")
    poles = extract_all_poles(
        source_dir / "train" / "images",
        source_dir / "train" / "labels",
    )
    if not poles:
        print("No poles found — aborting.")
        raise typer.Exit(1)

    bg_paths = [p for p in bg_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not bg_paths:
        print(f"No backgrounds found in {bg_dir} — falling back to training images.")
        bg_paths = list((source_dir / "train" / "images").glob("*.*"))
    print(f"Using {len(bg_paths)} background images from {bg_dir}")

    print(f"Generating {num_synthetic} synthetic images...")
    generated, attempts = 0, 0

    while generated < num_synthetic and attempts < num_synthetic * 5:
        attempts += 1

        bg = cv2.imread(str(random.choice(bg_paths)))
        if bg is None:
            continue

        bboxes = []
        for pole in random.choices(poles, k=random.randint(1, 3)):
            result = paste_pole(bg, pole)
            if result is None:
                continue
            bg, bbox = result
            bboxes.append(bbox)

        if not bboxes:
            continue

        stem = f"synthetic_{generated:05d}"
        cv2.imwrite(str(img_out / f"{stem}.jpg"), bg)
        with open(lbl_out / f"{stem}.txt", "w") as f:
            for bbox in bboxes:
                f.write(f"0 {' '.join(f'{v:.6f}' for v in bbox)}\n")

        generated += 1
        if generated % 50 == 0:
            print(f"  {generated}/{num_synthetic}")

    print(f"Done — {generated} images written to {output_dir}")


def merge_datasets(source_dir: Path, target_dir: Path) -> None:
    """Merge train split of source into target, prefixing filenames to avoid collisions."""
    prefix = source_dir.name

    for split in ["train"]:  # never touch valid
        img_src = source_dir / split / "images"
        lbl_src = source_dir / split / "labels"
        img_dst = target_dir / split / "images"
        lbl_dst = target_dir / split / "labels"

        if not img_src.exists():
            print(f"  {split}/images not found in source — skipping.")
            continue

        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img_path in img_src.glob("*.*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            shutil.copy(img_path, img_dst / f"{prefix}_{img_path.name}")
            lbl_path = lbl_src / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_dst / f"{prefix}_{img_path.stem}.txt")
            copied += 1

        print(f"  Merged {copied} {split} images into {target_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def create(
    num_synthetic: Annotated[int, typer.Option("--num-synthetic", "-n")] = 300,
    merge: Annotated[bool, typer.Option("--merge")] = False,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
) -> None:
    """Create a synthetic copy-paste dataset from training poles + background images."""
    settings = get_settings()
    out = output_dir or settings.DATASET_SYNTHETIC_PATH

    create_dataset(
        output_dir=out,
        source_dir=settings.DATASET_BASE_PATH,
        bg_dir=settings.DATASET_SYNTH_BACKGROUNDS_PATH,
        num_synthetic=num_synthetic,
    )

    if merge:
        print(f"\nMerging into {settings.DATASET_FINISHED_YOLO}...")
        merge_datasets(source_dir=out, target_dir=settings.DATASET_FINISHED_YOLO)


@app.command()
def merge(
    source: Annotated[Path | None, typer.Argument(help="Dataset to merge from (e.g. ./dataset-copied)")],
    target: Annotated[Path | None, typer.Argument(help="Dataset to merge into (default: DATASET_FINISHED_YOLO)")] = None,
) -> None:
    """Merge a synthetic dataset's train split into an existing dataset."""
    settings = get_settings()
    merge_datasets(source_dir=source or settings.DATASET_SYNTHETIC_PATH, target_dir=target or settings.DATASET_FINISHED_YOLO)


@app.command()
def clear_cache() -> None:
    """Delete the cached pole crops so SAM2 will re-segment on the next run."""
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR)
        print(f"Cleared cache at {_CACHE_DIR}/")
    else:
        print("No cache found — nothing to clear.")


if __name__ == "__main__":
    app()