from typing import Optional

from features.regime_filter import RegimeFilter
from features.volatility_filter import VolatilityFilter
from strategy.trend_guard import TrendGuard


def should_open_long(p_up: float, ema_diff: float, prob_threshold: float) -> bool:
    # legacy
    return (p_up >= prob_threshold) and (ema_diff > 0)


def should_open_short(p_up: float, ema_diff: float, prob_threshold: float) -> bool:
    # legacy
    return ((1.0 - p_up) >= prob_threshold) and (ema_diff < 0)


def should_open_long_v2(
    p_up: float,
    ema_diff: float,
    prob_threshold: float,
    *,
    price: float,
    ema_trend: float,
    ema_trend_prev: Optional[float],
    atr: float,
    adx: Optional[float],
    vol_z: Optional[float],
    cfg: dict,
) -> bool:
    base = (p_up >= prob_threshold) and (ema_diff > 0)
    if not base:
        return False

    r = RegimeFilter(cfg["filters"]["regime"]).decide(
        price=price,
        ema_trend=ema_trend,
        ema_trend_prev=ema_trend_prev,
        atr=atr,
        adx=adx,
        ema_diff=ema_diff,
    )
    if not r.allow_long:
        return False

    v = VolatilityFilter(cfg["filters"]["volatility"]).decide(atr=atr, vol_z=vol_z)
    if not v.allow:
        return False

    g = TrendGuard(cfg["guards"]["counter_trend"]).decide(price=price, ema_trend=ema_trend)
    if not g.allow_long:
        return False

    return True


def should_open_short_v2(
    p_up: float,
    ema_diff: float,
    prob_threshold: float,
    *,
    price: float,
    ema_trend: float,
    ema_trend_prev: Optional[float],
    atr: float,
    adx: Optional[float],
    vol_z: Optional[float],
    cfg: dict,
) -> bool:
    base = ((1.0 - p_up) >= prob_threshold) and (ema_diff < 0)
    if not base:
        return False

    r = RegimeFilter(cfg["filters"]["regime"]).decide(
        price=price,
        ema_trend=ema_trend,
        ema_trend_prev=ema_trend_prev,
        atr=atr,
        adx=adx,
        ema_diff=ema_diff,
    )
    if not r.allow_short:
        return False

    v = VolatilityFilter(cfg["filters"]["volatility"]).decide(atr=atr, vol_z=vol_z)
    if not v.allow:
        return False

    g = TrendGuard(cfg["guards"]["counter_trend"]).decide(price=price, ema_trend=ema_trend)
    if not g.allow_short:
        return False

    return True
