from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    DATASET_SOURCE: Path = "/datasets/tdt4265/Poles2025/rgb"

    DATASET_RESULTS: Path = "./dataset-processed/"

    DATASET_AUGMENTOR_RESULTS: Path = "./dataset-augmented/"

    # Inference settings
    MODEL_PATH: Path = "./runs/detect/train/weights/best.pt"
    TEST_DIR: Path = "/datasets/tdt4265/Poles2025/roadpoles_v1/test/images"
    INFERENCE_OUTPUT: Path = "./inference-output/"

    model_config = SettingsConfigDict(
        env_file=".env",
    )

_settings = None
def get_settings() -> BaseSettings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
