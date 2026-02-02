# app/bootstrap.py
import yaml
from dotenv import load_dotenv

from data.twelvedata_client import TwelveDataClient
from storage.sqlite import connect, init_db
from utils.logger import setup_logger
from risk.risk_manager import RiskManager  # <-- IMPORTANT


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def bootstrap():
    load_dotenv()
    logger = setup_logger()

    cfg = load_config()
    con = connect(cfg["storage"]["db_path"])
    init_db(con)

    td = TwelveDataClient.from_env(cfg["twelvedata"]["base_url"])

    rm = RiskManager(
        starting_equity=cfg["risk"]["starting_equity"],
        daily_max_loss=cfg["risk"]["daily_max_loss"],
        risk_per_trade=cfg["risk"]["risk_per_trade"],
        cluster_cfg=cfg["guards"]["clustered_entries"],
    )


    logger.info(
        "Bootstrapped app (db=%s, pairs=%s, tf=%s)",
        cfg["storage"]["db_path"],
        cfg["trading"]["pairs"],
        cfg["trading"]["timeframe"],
    )

    return cfg, con, td, logger, rm
