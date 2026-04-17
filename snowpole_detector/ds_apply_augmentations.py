import shutil
from pathlib import Path

import albumentations as A
import cv2
import typer

from snowpole_detector.settings import get_settings


def get_augmentation_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.Blur(blur_limit=3, p=0.1),
                    A.Defocus(radius=(1, 3), p=0.2),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.RandomRain(
                        slant_range=(-10, 10),
                        drop_length=10,
                        drop_width=1,
                        blur_value=3,
                        p=0.3,
                    ),
                    A.RandomSnow(
                        snow_point_range=(0.1, 0.3),
                        brightness_coeff=1.5,
                        p=0.4,
                    ),
                    A.RandomFog(
                        fog_coef_range=(0.1, 0.3),
                        alpha_coef=0.1,
                        p=0.3,
                    ),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.RandomShadow(
                        shadow_roi=(0, 0, 1, 1),
                        num_shadows_limit=(1, 2),
                        shadow_dimension=5,
                        p=0.3,
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_range=(0, 1),
                        src_radius=100,
                        p=0.2,
                    ),
                    A.RandomToneCurve(scale=0.2, p=0.3),
                ],
                p=0.3,
            ),
            # Mean brightness is low (93.8/255)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5,
            ),
            # Keep poles mostly vertical
            A.ShiftScaleRotate(
                shift_limit=0.1625,
                scale_limit=0.3,
                rotate_limit=15,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(60, 100), p=0.3),
                    A.GaussNoise(p=0.3),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(50, 100),
                hole_width_range=(50, 100),
                fill="inpaint_ns",
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            # coord_format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            clip=True,
        ),
    )


def process_augmentation(source_dir: Path, output_dir: Path, multiplier: int = 3) -> None:
    transform = get_augmentation_pipeline()

    for subset in ["train", "valid"]:
        img_in = source_dir / subset / "images"
        lbl_in = source_dir / subset / "labels"
        img_out = output_dir / subset / "images"
        lbl_out = output_dir / subset / "labels"

        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        yaml_file = source_dir / "data.yaml"
        yaml_out = output_dir / "data.yaml"
        shutil.copyfile(yaml_file, yaml_out)

        if not img_in.exists():
            continue

        print(f"Processing {subset}...")

        for img_path in img_in.glob("*.*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            label_path = lbl_in / f"{img_path.stem}.txt"

            if subset == "train":
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Warning: could not read {img_path}, skipping")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                bboxes = []
                class_labels = []
                if label_path.exists():
                    with open(label_path) as f:
                        for line in f:
                            parts = list(map(float, line.strip().split()))
                            if not parts:
                                continue
                            class_labels.append(int(parts[0]))
                            bboxes.append(parts[1:])

                shutil.copy(img_path, img_out / img_path.name)
                if label_path.exists():
                    shutil.copy(label_path, lbl_out / label_path.name)

                for i in range(multiplier):
                    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    save_stem = f"{img_path.stem}_aug_{i}"

                    cv2.imwrite(
                        str(img_out / f"{save_stem}.jpg"),
                        cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR),
                    )

                    with open(lbl_out / f"{save_stem}.txt", "w") as f:
                        for cls, box in zip(
                            transformed["class_labels"], transformed["bboxes"], strict=True
                        ):
                            f.write(f"{cls} {' '.join([f'{c:.6f}' for c in box])}\n")
            else:
                # Valid — copy files directly without augmentation
                shutil.copy(img_path, img_out / img_path.name)
                if label_path.exists():
                    shutil.copy(label_path, lbl_out / label_path.name)


def main(multiplier: int = 4) -> None:
    print(f"Doing augmentations with multiplier: {multiplier}")
    settings = get_settings()
    process_augmentation(settings.DATASET_PREPROCESS_PATH, settings.DATASET_FINISHED_YOLO, multiplier)
    print("Augmentation complete.")


if __name__ == "__main__":
    typer.run(main)
