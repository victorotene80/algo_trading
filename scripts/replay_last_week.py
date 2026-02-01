# scripts/replay_last_week.py
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from typing import Union
import pandas as pd

from app.bootstrap import bootstrap
from storage.sqlite import (
    load_candles,
    save_signal,
    insert_trade_open,
    close_trade,
)
from model.predict import ModelPredictor
from risk.risk_manager import RiskManager
from execution.paper import PaperExecutor
from strategy.signal_engine import should_open_long, should_open_short


def _ensure_tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    SQLite/pandas typically returns tz-naive DatetimeIndex (dtype=datetime64[ns]).
    Ensure we stay tz-naive everywhere to avoid tz-aware vs tz-naive comparisons.
    """
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.tz_localize(None)
        except TypeError:
            pass
    return df


def _ts_db_format(ts: Union[pd.Timestamp, datetime]) -> str:
    """
    DB stores timestamps like 'YYYY-MM-DD HH:MM:SS' (space, no 'T').
    This must match for DELETE comparisons because SQLite compares TEXT lexicographically.
    """
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def main():
    cfg, con, _, logger = bootstrap()

    test_days = int(cfg.get("backtest", {}).get("test_days", 7))
    pairs: List[str] = cfg["trading"]["pairs"]
    max_open = int(cfg["trading"]["max_open_positions"])

    # cutoff = now - test_days (tz-aware -> tz-naive)
    cutoff = datetime.now(timezone.utc) - timedelta(days=test_days)
    cutoff = cutoff.replace(tzinfo=None)
    cutoff_db = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    # Optional: clear prior replay window results
    if os.getenv("CLEAR_REPLAY") == "1":
        con.execute("DELETE FROM trades  WHERE entry_ts >= ?", (cutoff_db,))
        con.execute("DELETE FROM signals WHERE ts       >= ?", (cutoff_db,))
        con.commit()
        logger.info("Cleared trades/signals from cutoff=%s", cutoff_db)

    model_path = "model/artifacts/gb_model_upto_lastweek.joblib"
    predictor = ModelPredictor(model_path)

    rm = RiskManager(
        starting_equity=float(cfg["risk"]["starting_equity"]),
        daily_max_loss=float(cfg["risk"]["daily_max_loss"]),
        risk_per_trade=float(cfg["risk"]["risk_per_trade"]),
    )

    executor = PaperExecutor(
        sl_atr_mult=float(cfg["risk"]["sl_atr_mult"]),
        tp_r_mult=float(cfg["risk"]["tp_r_mult"]),
        time_stop_bars=int(cfg["risk"]["time_stop_bars"]),
    )

    prob_threshold = float(cfg["model"]["prob_threshold"])
    cooldown = int(cfg["model"]["cooldown_bars_after_trade"])
    last_trade_bar_idx: Dict[str, int] = {p: -10_000 for p in pairs}

    # Load full candles
    full = {p: _ensure_tz_naive_index(load_candles(con, p, limit=8000)) for p in pairs}

    # Replay window (>= cutoff)
    replay = {p: df[df.index >= cutoff].copy() for p, df in full.items()}
    for p, df in replay.items():
        logger.info("Replay set %s rows=%d from=%s", p, len(df), cutoff_db)

    if any(df.empty for df in replay.values()):
        missing = [p for p, df in replay.items() if df.empty]
        logger.error("Replay has no rows for: %s", missing)
        return

    # Unified timeline across pairs
    all_ts = sorted(set().union(*[set(df.index) for df in replay.values()]))

    # “Seen” buffer starts as history before cutoff
    seen = {p: full[p][full[p].index < cutoff].copy() for p in pairs}
    for p in pairs:
        seen[p] = _ensure_tz_naive_index(seen[p])

    logger.info(
        "Starting replay (model=%s, days=%d, total_bars=%d, pairs=%s)",
        model_path, test_days, len(all_ts), pairs
    )

    last_day = None
    trades_opened = 0
    trades_closed = 0

    for ts in all_ts:
        # Append this timestamp’s bars (if any) into each pair’s seen df
        for pair in pairs:
            df_pair = replay[pair]
            if ts in df_pair.index:
                seen[pair] = pd.concat([seen[pair], df_pair.loc[[ts]]]).sort_index()

        # Day rollover
        day = ts.strftime("%Y-%m-%d")
        if last_day is None:
            last_day = day
            rm.reset_day()
            logger.info("Initialized replay day=%s equity=%.2f", day, rm.equity)
        elif day != last_day:
            logger.info("New day rollover %s -> %s (reset daily limits)", last_day, day)
            last_day = day
            rm.reset_day()

        # 1) Update open positions first (and persist CLOSE)
        for pair in pairs:
            candles = seen[pair]
            if candles.empty or ts not in candles.index:
                continue

            bar = candles.loc[ts]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])

            _, closed = executor.update_bar(pair=pair, ts=ts, high=high, low=low, close=close)
            for pos, exit_ts, exit_price, pnl, reason in closed:
                # persist close to DB
                close_trade(
                    con,
                    trade_id=int(pos.trade_id),
                    exit_ts=_ts_db_format(exit_ts),
                    exit_price=float(exit_price),
                    pnl=float(pnl),
                    reason=str(reason),
                )
                rm.update_equity(rm.equity + float(pnl))
                trades_closed += 1
                logger.info(
                    "CLOSE trade_id=%s pair=%s side=%s reason=%s exit=%.5f pnl=%.2f equity=%.2f",
                    pos.trade_id, pair, pos.side, reason, float(exit_price), float(pnl), rm.equity
                )

        # If halted, don’t open new trades
        if not rm.can_trade():
            logger.warning("Trading halted by daily loss limit (day_dd=%.2f%%)", rm.day_dd() * 100)
            continue

        if executor.total_open() >= max_open:
            continue

        # 2) Entry decisions (and persist SIGNAL + OPEN)
        for pair in pairs:
            candles = seen[pair]
            if len(candles) < 250:
                continue

            pred_ts, p_up, ctx = predictor.predict_last(pair, candles)

            # persist signal
            save_signal(con, pair, _ts_db_format(pred_ts), float(p_up), "lgbm-calibrated")

            this_bar_idx = len(candles)
            if this_bar_idx - last_trade_bar_idx[pair] <= cooldown:
                continue

            open_long = should_open_long(float(p_up), float(ctx["ema_diff"]), prob_threshold)
            open_short = should_open_short(float(p_up), float(ctx["ema_diff"]), prob_threshold)

            # If both somehow true (rare), skip to avoid conflicting decisions.
            if open_long and open_short:
                continue

            if not open_long and not open_short:
                continue

            risk_amount = rm.risk_amount()
            side = "LONG" if open_long else "SHORT"

            # INSERT trade row first (gets AUTOINCREMENT id)
            trade_id = insert_trade_open(
                con,
                pair=pair,
                entry_ts=_ts_db_format(pred_ts),
                side=side,
                entry_price=float(ctx["close"]),
                size_units=0.0,
                sl_price=0.0,
                tp_price=0.0,
            )

            # Open in executor (computes sl/tp/units)
            if side == "LONG":
                pos = executor.open_long(
                    trade_id=int(trade_id),
                    pair=pair,
                    ts=pred_ts,
                    price=float(ctx["close"]),
                    atr14=float(ctx["atr14"]),
                    risk_amount=risk_amount,
                )
            else:
                pos = executor.open_short(
                    trade_id=int(trade_id),
                    pair=pair,
                    ts=pred_ts,
                    price=float(ctx["close"]),
                    atr14=float(ctx["atr14"]),
                    risk_amount=risk_amount,
                )

            if pos is None:
                # mark it closed immediately (NO_ATR)
                close_trade(
                    con,
                    trade_id=int(trade_id),
                    exit_ts=_ts_db_format(pred_ts),
                    exit_price=float(ctx["close"]),
                    pnl=0.0,
                    reason="NO_ATR",
                )
                logger.warning("OPEN aborted trade_id=%s pair=%s side=%s (NO_ATR)", trade_id, pair, side)
                continue

            # Update row with real size/sl/tp
            con.execute(
                "UPDATE trades SET size_units=?, sl_price=?, tp_price=? WHERE id=?",
                (float(pos.units), float(pos.sl), float(pos.tp), int(trade_id))
            )
            con.commit()

            last_trade_bar_idx[pair] = this_bar_idx
            trades_opened += 1

            logger.info(
                "OPEN trade_id=%s pair=%s side=%s p_up=%.3f entry=%.5f sl=%.5f tp=%.5f units=%.2f equity=%.2f",
                trade_id, pair, side, float(p_up),
                float(pos.entry_price), float(pos.sl), float(pos.tp), float(pos.units), rm.equity
            )

    logger.info(
        "Replay done. equity=%.2f opened=%d closed=%d open_now=%d halted=%s",
        rm.equity, trades_opened, trades_closed, executor.total_open(), rm.halted
    )


if __name__ == "__main__":
    main()
