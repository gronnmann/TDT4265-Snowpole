from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw, ImageFont

from snowpole_detector.settings import get_settings

COLORS = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#00C7BE",
    "#30B0C7", "#32ADE6", "#007AFF", "#5856D6", "#AF52DE",
]


def draw_predictions(
    img: Image.Image,
    label_path: Path,
    line_width: int = 3,
) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    if not label_path.exists():
        return img

    lines = label_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        conf = float(parts[5]) if len(parts) > 5 else None

        x1 = int((x_c - w / 2) * W)
        y1 = int((y_c - h / 2) * H)
        x2 = int((x_c + w / 2) * W)
        y2 = int((y_c + h / 2) * H)

        color = COLORS[cls % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"cls={cls}" + (f" {conf:.2f}" if conf is not None else "")
        # Small filled box behind text for readability
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((x1, y1 - 22), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 22), label, fill="white", font=font)

    return img


def run_visualize(
    test_dir: Annotated[
        Path | None,
        typer.Option(help="Directory of test images (default: settings.TEST_DIR)"),
    ] = None,
    labels_dir: Annotated[
        Path | None,
        typer.Option(help="Directory of .txt label files (default: settings.INFERENCE_OUTPUT/labels)"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Where to save visualized images (default: settings.INFERENCE_OUTPUT/visualized)"),
    ] = None,
    max_images: Annotated[
        int,
        typer.Option(help="Max number of images to render (0 = all)"),
    ] = 0,
    scale: Annotated[
        float,
        typer.Option(help="Scale factor for output images (e.g. 0.5 for half size)"),
    ] = 0.5,
) -> None:
    settings = get_settings()

    resolved_test_dir = test_dir or settings.TEST_DIR
    resolved_labels_dir = labels_dir or (settings.INFERENCE_OUTPUT / "labels")
    resolved_output_dir = output_dir or (settings.INFERENCE_OUTPUT / "visualized")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png"}
    img_files = sorted(p for p in resolved_test_dir.iterdir() if p.suffix.lower() in img_extensions)

    if max_images > 0:
        img_files = img_files[:max_images]

    print(f"Test images  : {resolved_test_dir}")
    print(f"Labels dir   : {resolved_labels_dir}")
    print(f"Output dir   : {resolved_output_dir}")
    print(f"Images       : {len(img_files)}" + (f" (capped at {max_images})" if max_images else ""))

    for img_path in img_files:
        label_path = resolved_labels_dir / f"{img_path.stem}.txt"
        img = Image.open(img_path)

        if scale != 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

        annotated = draw_predictions(img, label_path)
        out_path = resolved_output_dir / f"{img_path.stem}_pred.jpg"
        annotated.save(out_path, quality=90)
        n_dets = len(label_path.read_text().strip().splitlines()) if label_path.exists() else 0
        print(f"  {img_path.name}: {n_dets} detection(s) -> {out_path.name}")

    print(f"\nDone. {len(img_files)} images saved to {resolved_output_dir}")


if __name__ == "__main__":
    typer.run(run_visualize)