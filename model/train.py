# model/train.py
import joblib
import pandas as pd
from typing import Optional, Dict, Tuple, List
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from features.build_features import build_features, forward_return


def _pair_id(pair: str) -> int:
    # must match predict.py exactly
    return 0 if pair == "EUR/USD" else 1


def _feature_schema() -> List[str]:
    """
    Canonical feature order used for BOTH training and inference.
    Must match build_features() output + pair_id.
    """
    return [
        "r1", "r3", "r6", "r12",
        "ema_diff",
        "ema_trend", "ema_trend_prev",
        "vol20",
        "atr14", "atr_norm",
        "vol_z",
        "trend_strength",
        "pair_id",
    ]


def _ensure_schema(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")
    return X.loc[:, cols]


def _build_dataset(
    candles_by_pair: dict[str, pd.DataFrame],
    horizon_bars: int,
    label_threshold: float,
    drop_middle: bool,
    cfg: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    X_all, y_all = [], []
    cols = _feature_schema()

    for pair, df in candles_by_pair.items():
        if df is None or df.empty:
            continue

        # IMPORTANT: config-driven features to match runtime
        feats = build_features(df, cfg=cfg)
        if feats.empty:
            continue

        close = df["close"].loc[feats.index]
        fwd_ret = forward_return(close, horizon_bars)

        if drop_middle and label_threshold > 0:
            keep = (fwd_ret > label_threshold) | (fwd_ret < -label_threshold)
            keep = keep.fillna(False)
            feats = feats.loc[keep]
            fwd_ret = fwd_ret.loc[feats.index]

        y = (fwd_ret > label_threshold).astype(int)

        valid = y.dropna().index
        feats = feats.loc[valid]
        y = y.loc[valid]

        if len(feats) < 500:
            continue

        feats = feats.copy()
        feats["pair_id"] = _pair_id(pair)

        feats = _ensure_schema(feats, cols)

        X_all.append(feats)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No training data available yet.")

    X = pd.concat(X_all, axis=0)
    y = pd.concat(y_all, axis=0)
    return X, y


def _make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def train_model(
    candles_by_pair: dict[str, pd.DataFrame],
    horizon_bars: int,
    min_rows: int,
    out_path: str,
    calibrate: bool,
    label_threshold: float,
    drop_middle: bool,
    cfg: Optional[Dict] = None,
) -> dict:
    X, y = _build_dataset(
        candles_by_pair=candles_by_pair,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
        drop_middle=drop_middle,
        cfg=cfg,
    )

    if len(X) < min_rows:
        raise RuntimeError(f"Not enough rows to train. Have {len(X)}, need {min_rows}.")

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    base = _make_lgbm()
    base.fit(X_train, y_train)

    model = base
    if calibrate:
        model = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        model.fit(X_test, y_test)

    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)

    joblib.dump(model, out_path)

    # save schema next to model (helps prevent future mismatch headaches)
    schema_path = out_path.replace(".joblib", ".features.joblib")
    joblib.dump(_feature_schema(), schema_path)

    return {
        "model_type": "lgbm",
        "calibrated": bool(calibrate),
        "rows": int(len(X)),
        "auc": float(auc),
        "features": _feature_schema(),
        "label_threshold": float(label_threshold),
        "drop_middle": bool(drop_middle),
        "feature_schema_path": schema_path,
    }
