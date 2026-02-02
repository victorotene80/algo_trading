# scripts/train_model_upto.py
from datetime import datetime, timedelta, timezone
import os

from app.bootstrap import bootstrap
from storage.sqlite import load_candles
from model.train import train_model


def main():
    boot = bootstrap()
    cfg, con, _, logger = boot[:4]

    test_days = int(cfg.get("backtest", {}).get("test_days", 7))
    pairs = cfg["trading"]["pairs"]

    # cutoff = now - test_days (tz-aware initially)
    cutoff = datetime.now(timezone.utc) - timedelta(days=test_days)

    # ✅ make cutoff tz-naive to match df.index dtype
    cutoff = cutoff.replace(tzinfo=None)

    # keep consistent with TwelveData max (and your fetch_history cap)
    lookback_cfg = int(cfg.get("backtest", {}).get("lookback_candles", 8000))
    lookback = min(lookback_cfg, 5000)

    candles_by_pair = {}
    for p in pairs:
        df = load_candles(con, p, limit=lookback)
        if df is None or df.empty:
            candles_by_pair[p] = df
            logger.warning("Train set %s rows=0 (empty)", p)
            continue

        # ✅ ensure df index is datetime & tz-naive
        try:
            df.index = df.index.tz_localize(None)
        except TypeError:
            pass

        # ✅ Filter strictly before cutoff (prevents leakage)
        df = df[df.index < cutoff]

        candles_by_pair[p] = df
        logger.info("Train set %s rows=%d cutoff=%s", p, len(df), cutoff.isoformat())

    os.makedirs("model/artifacts", exist_ok=True)
    out_path = "model/artifacts/gb_model_upto_lastweek.joblib"

    meta = train_model(
        candles_by_pair=candles_by_pair,
        horizon_bars=int(cfg["model"]["horizon_bars"]),
        min_rows=int(cfg["model"]["min_train_rows"]),
        out_path=out_path,
        calibrate=bool(cfg["model"]["calibrate"]),
        label_threshold=float(cfg["model"]["label_threshold"]),
        drop_middle=bool(cfg["model"]["drop_middle"]),
        cfg=cfg,  # ✅ CRITICAL
    )

    logger.info("Model saved: %s", out_path)
    logger.info("Meta: %s", meta)


if __name__ == "__main__":
    main()
