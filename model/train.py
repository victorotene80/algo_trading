import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from features.build_features import build_features, forward_return

def _build_dataset(
    candles_by_pair: dict[str, pd.DataFrame],
    horizon_bars: int,
    label_threshold: float,
    drop_middle: bool,
):
    X_all, y_all = [], []

    for pair, df in candles_by_pair.items():
        if df is None or df.empty:
            continue

        feats = build_features(df)
        if feats.empty:
            continue

        # Align close series to feats timestamps
        close = df["close"].loc[feats.index]
        fwd_ret = forward_return(close, horizon_bars)

        if drop_middle and label_threshold > 0:
            keep = (fwd_ret > label_threshold) | (fwd_ret < -label_threshold)
            keep = keep.fillna(False)
            feats = feats.loc[keep]
            fwd_ret = fwd_ret.loc[feats.index]

        # binary label: up if fwd_ret > threshold else 0
        y = (fwd_ret > label_threshold).astype(int)

        # drop last horizon NaNs
        valid = y.dropna().index
        feats = feats.loc[valid]
        y = y.loc[valid]

        if len(feats) < 500:
            continue

        # simple pooled model feature: pair_id
        pair_id = 0 if pair == "EUR/USD" else 1
        feats = feats.copy()
        feats["pair_id"] = pair_id

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
) -> dict:
    X, y = _build_dataset(
        candles_by_pair=candles_by_pair,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
        drop_middle=drop_middle,
    )

    if len(X) < min_rows:
        raise RuntimeError(f"Not enough rows to train. Have {len(X)}, need {min_rows}.")

    # Time split (walk-forward style)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    base = _make_lgbm()
    base.fit(X_train, y_train)

    model = base
    if calibrate:
        # Calibrate probabilities on "later" data (simple approach for MVP)
        model = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        model.fit(X_test, y_test)

    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)

    joblib.dump(model, out_path)

    return {
        "model_type": "lgbm",
        "calibrated": bool(calibrate),
        "rows": int(len(X)),
        "auc": float(auc),
        "features": list(X.columns),
        "label_threshold": float(label_threshold),
        "drop_middle": bool(drop_middle),
    }
