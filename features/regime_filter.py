from dataclasses import dataclass
from typing import Optional, Dict
import math


@dataclass
class RegimeDecision:
    regime: str  # "trend" | "range" | "none"
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


class RegimeFilter:
    """
    Lightweight regime classifier:
    - Uses EMA trend baseline slope vs ATR as a trend/range proxy.
    - Optionally uses ADX (not wired yet; you pass adx=None).
    """

    def __init__(self, cfg: Dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.slope_lookback = int(cfg.get("slope_lookback", 8))
        self.slope_min_atr_frac = float(cfg.get("slope_min_atr_frac", 0.05))
        self.adx_min_trend = float(cfg.get("adx_min_trend", 18))
        self.allow_range_trades = bool(cfg.get("allow_range_trades", False))

    def decide(
        self,
        price: float,
        ema_trend: float,
        ema_trend_prev: Optional[float],
        atr: float,
        adx: Optional[float] = None,
        ema_diff: Optional[float] = None,
    ) -> RegimeDecision:
        if not self.enabled:
            d = _safe_float(ema_diff, 0.0)
            return RegimeDecision("none", allow_long=(d > 0), allow_short=(d < 0), reason="regime_disabled")

        price = _safe_float(price)
        ema_trend = _safe_float(ema_trend)
        atr = max(_safe_float(atr), 1e-9)

        if ema_trend_prev is None:
            ema_trend_prev = ema_trend

        ema_trend_prev = _safe_float(ema_trend_prev)

        # 1-bar slope proxy (since we only pass prev). If you want true lookback slope later,
        # pass a series or precompute a slope feature.
        slope = (ema_trend - ema_trend_prev)
        slope_ok = abs(slope) >= (self.slope_min_atr_frac * atr)

        adx_ok = True
        if adx is not None:
            adx_ok = _safe_float(adx) >= self.adx_min_trend

        is_trend = slope_ok and adx_ok
        regime = "trend" if is_trend else "range"

        direction = _safe_float(ema_diff) if ema_diff is not None else (price - ema_trend)

        if regime == "range" and not self.allow_range_trades:
            return RegimeDecision("range", False, False, "range_blocked")

        return RegimeDecision(
            regime,
            allow_long=(direction > 0),
            allow_short=(direction < 0),
            reason=f"{regime}: slope_ok={slope_ok}, adx_ok={adx_ok}",
        )
