from dataclasses import dataclass
from typing import Dict
import math


@dataclass
class TrendGuardDecision:
    allow_long: bool
    allow_short: bool
    reason: str


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


class TrendGuard:
    """
    Counter-trend protection:
    - Blocks longs if price <= ema_trend
    - Blocks shorts if price >= ema_trend
    """

    def __init__(self, cfg: Dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.req_above_for_long = bool(cfg.get("require_price_above_trend_for_long", True))
        self.req_below_for_short = bool(cfg.get("require_price_below_trend_for_short", True))

    def decide(self, price: float, ema_trend: float) -> TrendGuardDecision:
        if not self.enabled:
            return TrendGuardDecision(True, True, "trend_guard_disabled")

        price = _safe_float(price)
        ema_trend = _safe_float(ema_trend)

        allow_long = True
        allow_short = True

        if self.req_above_for_long and price <= ema_trend:
            allow_long = False
        if self.req_below_for_short and price >= ema_trend:
            allow_short = False

        return TrendGuardDecision(allow_long, allow_short, "trend_guard_ok")
