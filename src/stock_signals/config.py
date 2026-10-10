from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "stock_signals"
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    data_dir: Path = Path("data")
    history_years: int = 10
    # Symbols fetched per download call.
    batch_size: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
