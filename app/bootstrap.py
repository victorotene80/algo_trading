import yaml
from dotenv import load_dotenv

from data.twelvedata_client import TwelveDataClient
from storage.sqlite import connect, init_db
from utils.logger import setup_logger

def load_config(path: str = "config/config.yaml") -> dict:
    return yaml.safe_load(open(path, "r"))

def bootstrap():
    load_dotenv()
    logger = setup_logger()

    cfg = load_config()
    con = connect(cfg["storage"]["db_path"])
    init_db(con)

    td = TwelveDataClient.from_env(cfg["twelvedata"]["base_url"])

    logger.info("Bootstrapped app (db=%s, pairs=%s, tf=%s)",
                cfg["storage"]["db_path"], cfg["trading"]["pairs"], cfg["trading"]["timeframe"])
    return cfg, con, td, logger
