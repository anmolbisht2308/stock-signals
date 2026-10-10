from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from stock_signals.sizing import SizingConfig


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

    swing_capital: float = 200_000
    positional_capital: float = 300_000
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 20.0
    max_per_sector: int = 3
    max_positions: int = 10
    slippage_pct: float = 0.1
    brokerage_per_order: float = 0.0  # delivery brokerage (₹0 at most discount brokers)

    def capital(self) -> dict[str, float]:
        return {"swing": self.swing_capital, "positional": self.positional_capital}

    def sizing(self) -> SizingConfig:
        return SizingConfig(
            risk_pct=self.risk_per_trade_pct,
            max_position_pct=self.max_position_pct,
            max_per_sector=self.max_per_sector,
            max_positions=self.max_positions,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
