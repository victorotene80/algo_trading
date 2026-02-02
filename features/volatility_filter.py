from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class VolDecision:
    allow: bool
    reason: str


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


class VolatilityFilter:
    """
    Blocks entries when:
    - ATR too low
    - ATR too high
    - Vol spike (zscore) too high
    """

    def __init__(self, cfg: Dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.atr_min = float(cfg.get("atr_min", 0.0))
        self.atr_max = float(cfg.get("atr_max", 1e9))
        self.block_on_spike = bool(cfg.get("block_on_spike", True))
        self.vol_spike_z = float(cfg.get("vol_spike_z", 2.5))

    def decide(self, atr: float, vol_z: Optional[float] = None) -> VolDecision:
        if not self.enabled:
            return VolDecision(True, "vol_disabled")

        atr = _safe_float(atr)
        if atr < self.atr_min:
            return VolDecision(False, f"atr_too_low<{self.atr_min}")
        if atr > self.atr_max:
            return VolDecision(False, f"atr_too_high>{self.atr_max}")

        if self.block_on_spike and vol_z is not None:
            vol_z = _safe_float(vol_z)
            if vol_z >= self.vol_spike_z:
                return VolDecision(False, f"vol_spike_z>={self.vol_spike_z}")

        return VolDecision(True, "vol_ok")
