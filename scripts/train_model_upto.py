from datetime import datetime, timedelta, timezone
import os

from app.bootstrap import bootstrap
from storage.sqlite import load_candles
from model.train import train_model

def main():
    cfg, con, _, logger = bootstrap()

    test_days = int(cfg.get("backtest", {}).get("test_days", 7))
    pairs = cfg["trading"]["pairs"]

    # cutoff = now - test_days (tz-aware initially)
    cutoff = datetime.now(timezone.utc) - timedelta(days=test_days)

    # 🔥 FIX: make cutoff tz-naive to match df.index dtype
    cutoff = cutoff.replace(tzinfo=None)

    candles_by_pair = {}
    for p in pairs:
        df = load_candles(con, p, limit=8000)

        # 🔥 Ensure df index is datetime & tz-naive
        df.index = df.index.tz_localize(None)

        # Filter before cutoff (now compatible)
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
    )

    logger.info("Model saved: %s", out_path)
    logger.info("Meta: %s", meta)

if __name__ == "__main__":
    main()
