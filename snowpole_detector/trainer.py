from enum import StrEnum
from typing import Annotated

import typer
from rfdetr import RFDETRMedium
from ultralytics import YOLO

from snowpole_detector.settings import get_settings


class ModelType(StrEnum):
    yolo = "yolo"
    rfdetr = "rfdetr"


default_variants: dict[ModelType, str] = {
    ModelType.yolo: "yolo26m.pt",
    ModelType.rfdetr: "idk enda",
}


def run_training(
    model: Annotated[ModelType, typer.Option(help="'yolo' or 'rfdetr'")] = ModelType.yolo,
    model_variant: Annotated[str, typer.Option(help="Model variant to use")] = "__default__",
):
    if model_variant == "__default__":
        model_variant = default_variants[model]

    settings = get_settings()

    if model == ModelType.yolo:
        yolo = YOLO(model_variant)
        yolo.train(
            data=settings.DATASET_FINISHED_YOLO / "data.yml",
            epochs=1000,
            imgsz=1280,
            batch=16,
            device=0,
            rect=True,
            mosaic=1.0,
            close_mosaic=10,
            box=10.0,
            cls=0.5,
            project="TDT4265-Snowpole",
            save_period=50,
            patience=50,
            cos_lr=True,
            lrf=0.01,
        )

    elif model == ModelType.rfdetr:
        rfdetr = RFDETRMedium() # https://mintlify.wiki/roboflow/rf-detr/train/training-parameters#param-checkpoint-interval
        rfdetr.train(
            dataset_dir=str(settings.DATASET_FINISHED_COCO),
            epochs=1000,
            batch_size=16,
            grad_accum_steps=1,
            wandb=True,
            save_period=50,
        )


if __name__ == "__main__":
    typer.run(run_training)