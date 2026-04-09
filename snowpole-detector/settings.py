
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    DATASET_SOURCE: str = "/datasets/tdt4265/Poles2025/"


    model_config = SettingsConfigDict(
        env_file=".env",
    )