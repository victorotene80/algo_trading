from app.bootstrap import bootstrap
from storage.sqlite import upsert_candles

def main():
    cfg, con, td, logger = bootstrap()
    pairs = cfg["trading"]["pairs"]
    interval = cfg["trading"]["timeframe"]

    for pair in pairs:
        df = td.fetch_time_series(pair, interval=interval, outputsize=5000)
        upsert_candles(con, pair, df)
        logger.info("Fetched %d candles for %s (latest=%s)", len(df), pair, str(df.index[-1]))

if __name__ == "__main__":
    main()
