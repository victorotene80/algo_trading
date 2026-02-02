import joblib
import pandas as pd
from typing import Optional, Dict
from features.build_features import build_features


class ModelPredictor:
    def __init__(self, model_path: str, cfg: Optional[Dict] = None):
        self.model = joblib.load(model_path)
        self.cfg = cfg

    def predict_last(self, pair: str, candles: pd.DataFrame):
        feats = build_features(candles, cfg=self.cfg)
        if feats.empty:
            raise RuntimeError("Not enough data for features.")

        ts = feats.index[-1]
        x = feats.iloc[[-1]].copy()

        # NOTE: if you add more pairs later, replace this with a mapping in config.
        x["pair_id"] = 0 if pair == "EUR/USD" else 1

        p_up = float(self.model.predict_proba(x)[0, 1])

        ctx = {
            "ema_diff": float(x["ema_diff"].iloc[0]),
            "ema_trend": float(x["ema_trend"].iloc[0]),
            "ema_trend_prev": float(x["ema_trend_prev"].iloc[0]),
            "atr14": float(x["atr14"].iloc[0]),
            "vol_z": float(x["vol_z"].iloc[0]),
            "close": float(candles["close"].loc[ts]),
        }
        return ts, p_up, ctx
