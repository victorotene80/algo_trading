import os
from app.bootstrap import bootstrap
from storage.sqlite import load_candles
from model.train import train_model

def main():
    cfg, con, _, logger = bootstrap()

    pairs = cfg["trading"]["pairs"]
    horizon = int(cfg["model"]["horizon_bars"])
    min_rows = int(cfg["model"]["min_train_rows"])

    calibrate = bool(cfg["model"]["calibrate"])
    label_threshold = float(cfg["model"]["label_threshold"])
    drop_middle = bool(cfg["model"]["drop_middle"])

    candles_by_pair = {p: load_candles(con, p, limit=8000) for p in pairs}

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
    )

    logger.info("Model saved: %s", out_path)
    logger.info("Meta: %s", meta)

if __name__ == "__main__":
    main()
