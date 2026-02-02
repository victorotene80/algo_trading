from typing import Optional, Dict
from risk.entry_cluster_guard import EntryClusterGuard


class RiskManager:
    """
    Expanded Risk Manager with:
    - daily drawdown halt
    - risk-per-trade sizing
    - clustered-entry protection (cooldown, same-side limit, loss streak pause)
    """

    def __init__(
        self,
        starting_equity: float,
        daily_max_loss: float,
        risk_per_trade: float,
        *,
        cluster_cfg: Optional[Dict] = None,
    ):
        self.starting_equity = float(starting_equity)
        self.daily_max_loss = float(daily_max_loss)
        self.risk_per_trade = float(risk_per_trade)

        self.start_equity_day = self.starting_equity
        self.equity = self.starting_equity
        self.halted = False

        self.cluster_guard = EntryClusterGuard(cluster_cfg or {"enabled": False})

    def reset_day(self):
        self.start_equity_day = self.equity
        self.halted = False

    def update_equity(self, new_equity: float):
        self.equity = float(new_equity)
        if self.day_dd() <= -self.daily_max_loss:
            self.halted = True

    def day_dd(self) -> float:
        return (self.equity - self.start_equity_day) / self.start_equity_day

    def can_trade(self) -> bool:
        if self.halted:
            return False
        if self.day_dd() <= -self.daily_max_loss:
            self.halted = True
            return False
        return True

    def risk_amount(self) -> float:
        return self.equity * self.risk_per_trade

    # ---- NEW: clustered entry checks ----

    def can_enter(self, pair: str, side: str, bar_index: int) -> bool:
        if not self.can_trade():
            return False
        decision = self.cluster_guard.can_enter(pair=pair, side=side, bar_index=bar_index)
        return decision.allow

    def on_trade_closed(self, pair: str, side: str, bar_index: int, pnl: float):
        self.cluster_guard.on_trade_closed(pair=pair, side=side, bar_index=bar_index, pnl=pnl)
