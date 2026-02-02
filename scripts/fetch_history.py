# scripts/fetch_history.py
from app.bootstrap import bootstrap
from storage.sqlite import upsert_candles

MAX_OUTPUTSIZE = 5000   # TwelveData API limit

def main():
    boot = bootstrap()
    cfg, con, td, logger = boot[:4]

    pairs = cfg["trading"]["pairs"]
    interval = cfg["trading"]["timeframe"]

    requested = int(cfg["backtest"]["lookback_candles"])
    outputsize = min(requested, MAX_OUTPUTSIZE)

    logger.info("Requested %d candles, using API limit %d", requested, outputsize)

    for pair in pairs:
        df = td.fetch_time_series(pair, interval=interval, outputsize=outputsize)

        upsert_candles(con, pair, df)

        logger.info(
            "Fetched history pair=%s rows=%d latest_ts=%s",
            pair,
            len(df),
            str(df.index[-1]) if len(df) else None
        )

    logger.info("History fetch complete.")

if __name__ == "__main__":
    main()
