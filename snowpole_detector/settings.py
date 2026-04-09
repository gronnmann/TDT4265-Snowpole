from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATASET_SOURCE: Path = Path("/datasets/tdt4265/Poles2025/rgb")
    DATASET_RESULTS: Path = Path("./dataset-processed/")
    DATASET_AUGMENTOR_RESULTS: Path = Path("./dataset-augmented/")

    # Inference settings
    MODEL_PATH: Path = Path("./runs/detect/train/weights/best.pt")
    TEST_DIR: Path = Path("/datasets/tdt4265/Poles2025/rgb/test/images")
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
