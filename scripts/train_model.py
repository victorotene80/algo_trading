# scripts/train_model.py
import os

from app.bootstrap import bootstrap
from storage.sqlite import load_candles
from model.train import train_model


def main():
    boot = bootstrap()
    cfg, con, _, logger = boot[:4]

    pairs = cfg["trading"]["pairs"]
    horizon = int(cfg["model"]["horizon_bars"])
    min_rows = int(cfg["model"]["min_train_rows"])

    calibrate = bool(cfg["model"]["calibrate"])
    label_threshold = float(cfg["model"]["label_threshold"])
    drop_middle = bool(cfg["model"]["drop_middle"])

    # keep consistent with TwelveData max (and your fetch_history cap)
    lookback_cfg = int(cfg.get("backtest", {}).get("lookback_candles", 8000))
    lookback = min(lookback_cfg, 5000)

    candles_by_pair = {}
    for p in pairs:
        df = load_candles(con, p, limit=lookback)
        if df is not None and not df.empty:
            # ensure tz-naive index
            try:
                df.index = df.index.tz_localize(None)
            except TypeError:
                pass
        candles_by_pair[p] = df
        logger.info("Train dataset %s rows=%d", p, 0 if df is None else len(df))

    os.makedirs("model/artifacts", exist_ok=True)
    out_path = "model/artifacts/gb_model.joblib"

    meta = train_model(
        candles_by_pair=candles_by_pair,
        horizon_bars=horizon,
        min_rows=min_rows,
        out_path=out_path,
        calibrate=calibrate,
        label_threshold=label_threshold,
        drop_middle=drop_middle,
        cfg=cfg,  # ✅ CRITICAL: train features match runtime features
    )

    logger.info("Model saved: %s", out_path)
    logger.info("Meta: %s", meta)


if __name__ == "__main__":
    main()
