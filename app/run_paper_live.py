import time

from storage.sqlite import (
    upsert_candles,
    load_candles,
    save_signal,
    insert_trade_open,
    close_trade,
)
from model.predict import ModelPredictor
from risk.risk_manager import RiskManager
from execution.paper import PaperExecutor
from strategy.signal_engine import should_open_long_v2, should_open_short_v2


def run(cfg, con, td, logger, model_path: str):
    pairs = cfg["trading"]["pairs"]
    interval = cfg["trading"]["timeframe"]

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

    predictor = ModelPredictor(model_path, cfg=cfg)

    prob_threshold = float(cfg["model"]["prob_threshold"])
    max_open = int(cfg["trading"]["max_open_positions"])

    sleep_seconds = 240
    last_day = None

    logger.info(
        "Starting paper live loop (sleep=%ss, max_open=%s, risk_per_trade=%.2f%%, daily_max_loss=%.2f%%, model=%s)",
        sleep_seconds,
        max_open,
        float(cfg["risk"]["risk_per_trade"]) * 100.0,
        float(cfg["risk"]["daily_max_loss"]) * 100.0,
        model_path,
    )

    while True:
        try:
            # 1) Fetch + store latest candles
            for pair in pairs:
                df_new = td.fetch_time_series(pair, interval=interval, outputsize=800)
                upsert_candles(con, pair, df_new)
                logger.info(
                    "Fetched candles pair=%s rows=%d latest_ts=%s",
                    pair,
                    len(df_new),
                    str(df_new.index[-1]) if len(df_new) else "None",
                )

            # 2) Establish clock
            clock = load_candles(con, pairs[0], limit=50)
            if clock.empty:
                logger.warning("No data yet, sleeping...")
                time.sleep(30)
                continue

            now_ts = clock.index[-1]
            day = now_ts.strftime("%Y-%m-%d")

            # 3) Day rollover reset
            if last_day is None:
                last_day = day
                rm.reset_day()
                logger.info("Initialized trading day=%s equity=%.2f", day, rm.equity)

            if day != last_day:
                logger.info("New day rollover from %s -> %s (reset daily limits)", last_day, day)
                last_day = day
                rm.reset_day()

            # 4) Per pair loop
            for pair in pairs:
                candles = load_candles(con, pair, limit=5000)
                if len(candles) < 250:
                    logger.warning("Not enough candles for pair=%s (have=%d)", pair, len(candles))
                    continue

                ts, p_up, ctx = predictor.predict_last(pair, candles)
                save_signal(con, pair, str(ts), p_up, "lgbm-calibrated")

                p_down = 1.0 - p_up
                logger.debug(
                    "Signal pair=%s ts=%s p_up=%.4f p_down=%.4f ema_diff=%.6f atr14=%.6f close=%.5f ema_trend=%.5f vol_z=%.3f",
                    pair,
                    str(ts),
                    p_up,
                    p_down,
                    ctx["ema_diff"],
                    ctx["atr14"],
                    ctx["close"],
                    ctx["ema_trend"],
                    ctx["vol_z"],
                )

                # Update open positions on this bar
                bar = candles.loc[ts]
                _, closed = executor.update_bar(
                    pair=pair,
                    ts=ts,
                    high=float(bar["high"]),
                    low=float(bar["low"]),
                    close=float(bar["close"]),
                )

                bar_idx = len(candles)

                # Persist closures and feed cluster guard
                for pos, exit_ts, exit_price, pnl, reason in closed:
                    close_trade(con, pos.trade_id, str(exit_ts), float(exit_price), float(pnl), reason)

                    rm.update_equity(rm.equity + float(pnl))
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

                # Daily risk guardrail
                if not rm.can_trade():
                    logger.warning("Trading halted by daily loss limit (day_dd=%.2f%%)", rm.day_dd() * 100.0)
                    continue

                # Position limit guardrail
                if executor.total_open() >= max_open:
                    logger.info("Max open positions reached (%d), skipping entries", executor.total_open())
                    continue

                # Entry decision via v2 filters/guards
                side_to_open = None

                if should_open_long_v2(
                    p_up,
                    ctx["ema_diff"],
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
                    p_up,
                    ctx["ema_diff"],
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

                # Cluster guard (replaces old cooldown logic)
                if not rm.can_enter(pair=pair, side=side_to_open, bar_index=bar_idx):
                    logger.info("Entry blocked by clustered-entry guard pair=%s side=%s bar_idx=%d", pair, side_to_open, bar_idx)
                    continue

                risk_amount = rm.risk_amount()

                trade_id = insert_trade_open(
                    con,
                    pair=pair,
                    entry_ts=str(ts),
                    side=side_to_open,
                    entry_price=float(ctx["close"]),
                    size_units=0.0,
                    sl_price=0.0,
                    tp_price=0.0,
                )

                if side_to_open == "LONG":
                    pos = executor.open_long(
                        trade_id=trade_id,
                        pair=pair,
                        ts=ts,
                        price=float(ctx["close"]),
                        atr14=float(ctx["atr14"]),
                        risk_amount=risk_amount,
                    )
                else:
                    pos = executor.open_short(
                        trade_id=trade_id,
                        pair=pair,
                        ts=ts,
                        price=float(ctx["close"]),
                        atr14=float(ctx["atr14"]),
                        risk_amount=risk_amount,
                    )

                if pos is None:
                    close_trade(con, trade_id, str(ts), float(ctx["close"]), 0.0, "NO_ATR")
                    logger.warning("OPEN aborted trade_id=%s pair=%s side=%s (NO_ATR)", trade_id, pair, side_to_open)
                    continue

                con.execute(
                    "UPDATE trades SET size_units=?, sl_price=?, tp_price=? WHERE id=?",
                    (float(pos.units), float(pos.sl), float(pos.tp), int(trade_id)),
                )
                con.commit()

                logger.info(
                    "OPEN trade_id=%s pair=%s side=%s p_up=%.3f p_down=%.3f entry=%.5f sl=%.5f tp=%.5f units=%.2f equity=%.2f",
                    trade_id,
                    pair,
                    side_to_open,
                    p_up,
                    p_down,
                    pos.entry_price,
                    pos.sl,
                    pos.tp,
                    pos.units,
                    rm.equity,
                )

            # Status
            logger.info(
                "Status equity=%.2f day_dd=%.2f%% open=%d halted=%s",
                rm.equity,
                rm.day_dd() * 100.0,
                executor.total_open(),
                rm.halted,
            )

            time.sleep(sleep_seconds)

        except Exception:
            logger.exception("Unhandled exception in live loop")
            time.sleep(30)
