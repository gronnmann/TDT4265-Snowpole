from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATASET_BASE_PATH: Path = Path("./ds-v0/01-raw-and-synthetic")
    DATASET_SYNTHETIC_PATH: Path = Path("./ds-v0/01-synthetic/")
    DATASET_PREPROCESS_PATH: Path = Path("./ds-v0/02-split/")
    DATASET_FINISHED_YOLO: Path = Path("./ds-v0/03-finished-yolo/")
    DATASET_FINISHED_COCO: Path = Path("./ds-v0/03-finished-coco-coco/")

    DATASET_SYNTH_BACKGROUNDS_PATH: Path = Path("./dataset-synthetic-bg/")
    # BACKGROUND FOR SYNTHETIC DATASET

    # Inference settings
    MODEL_PATH: Path = Path("./runs/detect/train/weights/best.pt")
    TEST_DIR: Path = Path("/datasets/tdt4265/Poles2025/roadpoles_v1/test/images")
    INFERENCE_OUTPUT: Path = Path("./inference-output/")

    model_config = SettingsConfigDict(
        env_file=".env",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
