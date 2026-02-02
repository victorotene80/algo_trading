import numpy as np
import pandas as pd
from typing import Optional, Dict


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame, cfg: Optional[Dict] = None) -> pd.DataFrame:
    """
    Builds model + execution features.

    Added (for filters/guards):
    - ema_trend, ema_trend_prev
    - vol_z (rolling vol z-score)
    """
    c = df["close"]

    r1 = c.pct_change(1)
    r3 = c.pct_change(3)
    r6 = c.pct_change(6)
    r12 = c.pct_change(12)

    # Configurable periods (safe defaults)
    ema_fast = int(cfg.get("features", {}).get("ema_fast", 20)) if cfg else 20
    ema_slow = int(cfg.get("features", {}).get("ema_slow", 50)) if cfg else 50

    ema_trend_period = int(
        (cfg.get("filters", {}).get("regime", {}).get("ema_trend_period", 200) if cfg else 200)
    )

    atr_period = int(
        (cfg.get("filters", {}).get("volatility", {}).get("atr_period", 14) if cfg else 14)
    )

    vol20_window = int(cfg.get("features", {}).get("vol_window", 20)) if cfg else 20
    vol_z_window = int(cfg.get("filters", {}).get("volatility", {}).get("vol_window", 30)) if cfg else 30

    # EMAs
    ema_fast_s = ema(c, ema_fast)
    ema_slow_s = ema(c, ema_slow)

    # Existing directional feature
    ema_diff = (ema_fast_s - ema_slow_s) / c

    # NEW: slow baseline for trend guard / regime
    ema_trend = ema(c, ema_trend_period)
    ema_trend_prev = ema_trend.shift(1)

    # Volatility + ATR
    vol20 = r1.rolling(vol20_window).std()
    a14 = atr(df, atr_period)
    atr_norm = a14 / c

    # NEW: volatility z-score (for spike blocking)
    vol = r1.rolling(vol_z_window).std()
    vol_mean = vol.rolling(vol_z_window).mean()
    vol_std = vol.rolling(vol_z_window).std().replace(0, np.nan)
    vol_z = (vol - vol_mean) / vol_std

    trend_strength = (ema_diff.abs() / (vol20.replace(0, np.nan))).fillna(0)

    out = pd.DataFrame(
        {
            "r1": r1,
            "r3": r3,
            "r6": r6,
            "r12": r12,
            "ema_diff": ema_diff,
            "ema_trend": ema_trend,
            "ema_trend_prev": ema_trend_prev,
            "vol20": vol20,
            "atr14": a14,
            "atr_norm": atr_norm,
            "vol_z": vol_z,
            "trend_strength": trend_strength,
        },
        index=df.index,
    )

    return out.dropna()


def forward_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    return (close.shift(-horizon_bars) / close) - 1.0


def make_labels(df: pd.DataFrame, horizon_bars: int, threshold: float = 0.0) -> pd.Series:
    c = df["close"]
    fwd_ret = forward_return(c, horizon_bars)
    y = (fwd_ret > threshold).astype(int)
    return y
