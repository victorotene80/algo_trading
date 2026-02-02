import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Union

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
from strategy.signal_engine import should_open_long_v2, should_open_short_v2


def _ensure_tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    SQLite/pandas typically returns tz-naive DatetimeIndex (datetime64[ns]).
    Ensure we stay tz-naive everywhere to avoid tz-aware vs tz-naive comparisons.
    """
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.tz_localize(None)
        except TypeError:
            # already tz-naive
            pass
    return df


def _ts_db_format(ts: Union[pd.Timestamp, datetime]) -> str:
    """
    DB stores timestamps like 'YYYY-MM-DD HH:MM:SS' (space, no 'T').
    This must match for SQLite TEXT comparisons.
    """
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def main():
    # ✅ bootstrap now returns (cfg, con, td, logger, rm)
    boot = bootstrap()
    cfg, con, _, logger = boot[:4]

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

    # Use model trained "up to last week" for replay window
    model_path = "model/artifacts/gb_model_upto_lastweek.joblib"
    predictor = ModelPredictor(model_path, cfg=cfg)  # ✅ important

    # Risk manager with cluster guard config
    rm = RiskManager(
        starting_equity=float(cfg["risk"]["starting_equity"]),
        daily_max_loss=float(cfg["risk"]["daily_max_loss"]),
        risk_per_trade=float(cfg["risk"]["risk_per_trade"]),
        cluster_cfg=cfg.get("guards", {}).get("clustered_entries", {"enabled": False}),
    )

    executor = PaperExecutor(
        sl_atr_mult=float(cfg["risk"]["sl_atr_mult"]),
        tp_r_mult=float(cfg["risk"]["tp_r_mult"]),
        time_stop_bars=int(cfg["risk"]["time_stop_bars"]),
    )

    prob_threshold = float(cfg["model"]["prob_threshold"])

    # Load candles (API limit is 5000; DB may store more over time—use config lookback)
    lookback = int(cfg.get("backtest", {}).get("lookback_candles", 8000))
    full: Dict[str, pd.DataFrame] = {
        p: _ensure_tz_naive_index(load_candles(con, p, limit=lookback)) for p in pairs
    }

    # Replay window (>= cutoff)
    replay: Dict[str, pd.DataFrame] = {p: df[df.index >= cutoff].copy() for p, df in full.items()}
    for p, df in replay.items():
        logger.info("Replay set %s rows=%d from=%s", p, len(df), cutoff_db)

    if any(df.empty for df in replay.values()):
        missing = [p for p, df in replay.items() if df.empty]
        logger.error("Replay has no rows for: %s", missing)
        return

    # Unified timeline across pairs
    all_ts = sorted(set().union(*[set(df.index) for df in replay.values()]))

    # Seen buffer starts as history before cutoff
    seen: Dict[str, pd.DataFrame] = {p: full[p][full[p].index < cutoff].copy() for p in pairs}
    for p in pairs:
        seen[p] = _ensure_tz_naive_index(seen[p])

    logger.info(
        "Starting replay (model=%s, days=%d, total_bars=%d, pairs=%s)",
        model_path,
        test_days,
        len(all_ts),
        pairs,
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

                # ✅ update cluster guard state after a close
                if hasattr(rm, "on_trade_closed"):
                    # bar_index = number of candles seen for that pair at this ts
                    bar_idx = len(candles)
                    rm.on_trade_closed(pair=pair, side=str(pos.side), bar_index=bar_idx, pnl=float(pnl))

                logger.info(
                    "CLOSE trade_id=%s pair=%s side=%s reason=%s exit=%.5f pnl=%.2f equity=%.2f",
                    pos.trade_id,
                    pair,
                    pos.side,
                    reason,
                    float(exit_price),
                    float(pnl),
                    rm.equity,
                )

        # If halted, don’t open new trades
        if not rm.can_trade():
            logger.warning("Trading halted by daily loss limit (day_dd=%.2f%%)", rm.day_dd() * 100.0)
            continue

        if executor.total_open() >= max_open:
            continue

        # 2) Entry decisions (persist SIGNAL + OPEN)
        for pair in pairs:
            candles = seen[pair]
            if len(candles) < 250:
                continue

            pred_ts, p_up, ctx = predictor.predict_last(pair, candles)

            # persist signal
            save_signal(con, pair, _ts_db_format(pred_ts), float(p_up), "lgbm-calibrated")

            # compute bar index for guards
            bar_idx = len(candles)

            side_to_open = None

            # ✅ use v2 logic with filters/guards
            if should_open_long_v2(
                float(p_up),
                float(ctx["ema_diff"]),
                prob_threshold,
                price=float(ctx["close"]),
                ema_trend=float(ctx["ema_trend"]),
                ema_trend_prev=float(ctx["ema_trend_prev"]),
                atr=float(ctx["atr14"]),
                adx=None,  # not computed yet
                vol_z=float(ctx["vol_z"]),
                cfg=cfg,
            ):
                side_to_open = "LONG"

            elif should_open_short_v2(
                float(p_up),
                float(ctx["ema_diff"]),
                prob_threshold,
                price=float(ctx["close"]),
                ema_trend=float(ctx["ema_trend"]),
                ema_trend_prev=float(ctx["ema_trend_prev"]),
                atr=float(ctx["atr14"]),
                adx=None,
                vol_z=float(ctx["vol_z"]),
                cfg=cfg,
            ):
                side_to_open = "SHORT"

            if side_to_open is None:
                continue

            # ✅ Cluster guard check (cooldown / clustering / loss pause)
            if hasattr(rm, "can_enter"):
                if not rm.can_enter(pair=pair, side=side_to_open, bar_index=bar_idx):
                    continue

            risk_amount = rm.risk_amount()

            # INSERT trade row first (gets AUTOINCREMENT id)
            trade_id = insert_trade_open(
                con,
                pair=pair,
                entry_ts=_ts_db_format(pred_ts),
                side=side_to_open,
                entry_price=float(ctx["close"]),
                size_units=0.0,
                sl_price=0.0,
                tp_price=0.0,
            )

            # Open in executor (computes sl/tp/units)
            if side_to_open == "LONG":
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
                logger.warning("OPEN aborted trade_id=%s pair=%s side=%s (NO_ATR)", trade_id, pair, side_to_open)
                continue

            # Update row with real size/sl/tp
            con.execute(
                "UPDATE trades SET size_units=?, sl_price=?, tp_price=? WHERE id=?",
                (float(pos.units), float(pos.sl), float(pos.tp), int(trade_id)),
            )
            con.commit()

            trades_opened += 1
            logger.info(
                "OPEN trade_id=%s pair=%s side=%s p_up=%.3f entry=%.5f sl=%.5f tp=%.5f units=%.2f equity=%.2f",
                trade_id,
                pair,
                side_to_open,
                float(p_up),
                float(pos.entry_price),
                float(pos.sl),
                float(pos.tp),
                float(pos.units),
                rm.equity,
            )

    logger.info(
        "Replay done. equity=%.2f opened=%d closed=%d open_now=%d halted=%s",
        rm.equity,
        trades_opened,
        trades_closed,
        executor.total_open(),
        rm.halted,
    )


if __name__ == "__main__":
    main()
