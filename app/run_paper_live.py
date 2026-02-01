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
from strategy.signal_engine import should_open_long, should_open_short


def run(cfg, con, td, logger, model_path: str):
    pairs = cfg["trading"]["pairs"]
    interval = cfg["trading"]["timeframe"]

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

    predictor = ModelPredictor(model_path)

    prob_threshold = float(cfg["model"]["prob_threshold"])
    cooldown = int(cfg["model"]["cooldown_bars_after_trade"])
    max_open = int(cfg["trading"]["max_open_positions"])

    # cooldown tracking (per pair)
    last_trade_bar_idx = {p: -10_000 for p in pairs}

    # Run loop timing (your original)
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
            # 1) Fetch + store latest candles for all pairs
            for pair in pairs:
                df_new = td.fetch_time_series(pair, interval=interval, outputsize=800)
                upsert_candles(con, pair, df_new)
                logger.info(
                    "Fetched candles pair=%s rows=%d latest_ts=%s",
                    pair,
                    len(df_new),
                    str(df_new.index[-1]) if len(df_new) else "None",
                )

            # 2) Establish the "clock" timestamp using first pair
            clock = load_candles(con, pairs[0], limit=50)
            if clock.empty:
                logger.warning("No data yet, sleeping...")
                time.sleep(30)
                continue

            now_ts = clock.index[-1]
            day = now_ts.strftime("%Y-%m-%d")

            # 3) Day rollover / daily risk reset
            if last_day is None:
                last_day = day
                rm.reset_day()
                logger.info("Initialized trading day=%s equity=%.2f", day, rm.equity)

            if day != last_day:
                logger.info("New day rollover from %s -> %s (reset daily limits)", last_day, day)
                last_day = day
                rm.reset_day()

            # 4) Per pair: predict, save signal, update positions, maybe open new position
            for pair in pairs:
                candles = load_candles(con, pair, limit=5000)
                if len(candles) < 250:
                    logger.warning("Not enough candles for pair=%s (have=%d)", pair, len(candles))
                    continue

                # --- Predict current bar
                ts, p_up, ctx = predictor.predict_last(pair, candles)

                # Save signal (signals table has PK(pair, ts))
                save_signal(con, pair, str(ts), p_up, "lgbm-calibrated")

                p_down = 1.0 - p_up

                logger.debug(
                    "Signal pair=%s ts=%s p_up=%.4f p_down=%.4f ema_diff=%.6f atr14=%.6f close=%.5f",
                    pair,
                    str(ts),
                    p_up,
                    p_down,
                    ctx["ema_diff"],
                    ctx["atr14"],
                    ctx["close"],
                )

                # --- Update open positions using this bar
                bar = candles.loc[ts]
                _, closed = executor.update_bar(
                    pair=pair,
                    ts=ts,
                    high=float(bar["high"]),
                    low=float(bar["low"]),
                    close=float(bar["close"]),
                )

                # Persist closures to DB + update equity
                for pos, exit_ts, exit_price, pnl, reason in closed:
                    close_trade(con, pos.trade_id, str(exit_ts), float(exit_price), float(pnl), reason)
                    rm.update_equity(rm.equity + pnl)
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

                # --- Risk guardrails
                if not rm.can_trade():
                    logger.warning("Trading halted by daily loss limit (day_dd=%.2f%%)", rm.day_dd() * 100.0)
                    continue

                # --- Position limit guardrails
                if executor.total_open() >= max_open:
                    logger.info("Max open positions reached (%d), skipping entries", executor.total_open())
                    continue

                # --- Cooldown logic
                bar_idx = len(candles)
                if bar_idx - last_trade_bar_idx[pair] <= cooldown:
                    continue

                # 5) Entry decision: LONG first, else SHORT
                side_to_open = None
                if should_open_long(p_up, ctx["ema_diff"], prob_threshold):
                    side_to_open = "LONG"
                elif should_open_short(p_up, ctx["ema_diff"], prob_threshold):
                    side_to_open = "SHORT"

                if side_to_open is None:
                    continue

                risk_amount = rm.risk_amount()

                # Insert trade first (so we get db trade_id)
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

                # Create position in executor
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
                    # SHORT leg (requires open_short in PaperExecutor)
                    pos = executor.open_short(
                        trade_id=trade_id,
                        pair=pair,
                        ts=ts,
                        price=float(ctx["close"]),
                        atr14=float(ctx["atr14"]),
                        risk_amount=risk_amount,
                    )

                if pos is None:
                    # No ATR / invalid
                    close_trade(con, trade_id, str(ts), float(ctx["close"]), 0.0, "NO_ATR")
                    logger.warning("OPEN aborted trade_id=%s pair=%s side=%s (NO_ATR)", trade_id, pair, side_to_open)
                    continue

                # Persist executor values into DB
                con.execute(
                    "UPDATE trades SET size_units=?, sl_price=?, tp_price=? WHERE id=?",
                    (float(pos.units), float(pos.sl), float(pos.tp), int(trade_id)),
                )
                con.commit()

                last_trade_bar_idx[pair] = bar_idx

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

            # 6) Status log
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
