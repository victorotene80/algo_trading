from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd

@dataclass
class PaperPosition:
    trade_id: int
    pair: str
    entry_ts: pd.Timestamp
    side: str  # "LONG" | "SHORT"
    entry_price: float
    units: float
    sl: float
    tp: float
    bars_held: int = 0


class PaperExecutor:
    def __init__(self, sl_atr_mult: float, tp_r_mult: float, time_stop_bars: int):
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_r_mult = float(tp_r_mult)
        self.time_stop_bars = int(time_stop_bars)
        self.positions: Dict[str, List[PaperPosition]] = {}

    def total_open(self) -> int:
        return sum(len(v) for v in self.positions.values())

    def open_long(
        self,
        trade_id: int,
        pair: str,
        ts: pd.Timestamp,
        price: float,
        atr14: float,
        risk_amount: float
    ) -> Optional[PaperPosition]:
        sl_dist = self.sl_atr_mult * atr14
        if sl_dist <= 0:
            return None

        sl = price - sl_dist
        tp = price + (self.tp_r_mult * sl_dist)
        units = risk_amount / sl_dist  # positive

        pos = PaperPosition(trade_id, pair, ts, "LONG", price, units, sl, tp)
        self.positions.setdefault(pair, []).append(pos)
        return pos

    def open_short(
        self,
        trade_id: int,
        pair: str,
        ts: pd.Timestamp,
        price: float,
        atr14: float,
        risk_amount: float
    ) -> Optional[PaperPosition]:
        sl_dist = self.sl_atr_mult * atr14
        if sl_dist <= 0:
            return None

        # SHORT: SL above entry, TP below entry
        sl = price + sl_dist
        tp = price - (self.tp_r_mult * sl_dist)
        units = risk_amount / sl_dist  # positive

        pos = PaperPosition(trade_id, pair, ts, "SHORT", price, units, sl, tp)
        self.positions.setdefault(pair, []).append(pos)
        return pos

    def update_bar(
        self,
        pair: str,
        ts: pd.Timestamp,
        high: float,
        low: float,
        close: float
    ) -> Tuple[List[PaperPosition], List[Tuple[PaperPosition, pd.Timestamp, float, float, str]]]:
        if pair not in self.positions:
            return [], []

        still: List[PaperPosition] = []
        closed = []

        for pos in self.positions[pair]:
            pos.bars_held += 1
            exit_price = None
            reason = None

            if pos.side == "LONG":
                # conservative ordering: SL first if both touched
                if low <= pos.sl:
                    exit_price = pos.sl
                    reason = "SL"
                elif high >= pos.tp:
                    exit_price = pos.tp
                    reason = "TP"
            else:  # SHORT
                # conservative ordering: SL first if both touched
                if high >= pos.sl:
                    exit_price = pos.sl
                    reason = "SL"
                elif low <= pos.tp:
                    exit_price = pos.tp
                    reason = "TP"

            if exit_price is None and pos.bars_held >= self.time_stop_bars:
                exit_price = close
                reason = "TIME"

            if exit_price is None:
                still.append(pos)
            else:
                # pnl formula depends on side
                if pos.side == "LONG":
                    pnl = (exit_price - pos.entry_price) * pos.units
                else:
                    pnl = (pos.entry_price - exit_price) * pos.units  # SHORT profit when price falls

                closed.append((pos, ts, float(exit_price), float(pnl), reason))

        self.positions[pair] = still
        return still, closed
