import numpy as np
import pandas as pd

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    r1 = c.pct_change(1)
    r3 = c.pct_change(3)
    r6 = c.pct_change(6)
    r12 = c.pct_change(12)

    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    ema_diff = (ema20 - ema50) / c

    vol20 = r1.rolling(20).std()
    a14 = atr(df, 14)
    atr_norm = a14 / c

    trend_strength = (ema_diff.abs() / (vol20.replace(0, np.nan))).fillna(0)

    out = pd.DataFrame({
        "r1": r1,
        "r3": r3,
        "r6": r6,
        "r12": r12,
        "ema_diff": ema_diff,
        "vol20": vol20,
        "atr14": a14,
        "atr_norm": atr_norm,
        "trend_strength": trend_strength,
    }, index=df.index)

    return out.dropna()

def forward_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    return (close.shift(-horizon_bars) / close) - 1.0

def make_labels(df: pd.DataFrame, horizon_bars: int, threshold: float = 0.0) -> pd.Series:
    c = df["close"]
    fwd_ret = forward_return(c, horizon_bars)
    y = (fwd_ret > threshold).astype(int)
    return y
