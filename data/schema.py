from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Candle:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

