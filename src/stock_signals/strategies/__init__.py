from typing import Protocol

import pandas as pd

from stock_signals.models import Candidate, Position
from stock_signals.regime import Regime


class Strategy(Protocol):
    book: str
    version: str

    def candidates(self, snap: pd.DataFrame, regime: Regime, week_end: bool) -> list[Candidate]:
        """Rule hits on the snapshot (index = symbol), best first."""
        ...

    def max_new(self, regime: Regime) -> int:
        """How many new signals may be issued today."""
        ...

    def close_exit(self, pos: Position, row: pd.Series, week_end: bool) -> str | None:
        """Exit reason decided on today's close (executed at the next open), or None."""
        ...
