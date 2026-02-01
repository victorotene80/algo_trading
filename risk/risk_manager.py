class RiskManager:
    def __init__(self, starting_equity: float, daily_max_loss: float, risk_per_trade: float):
        self.starting_equity = float(starting_equity)
        self.daily_max_loss = float(daily_max_loss)
        self.risk_per_trade = float(risk_per_trade)

        self.start_equity_day = self.starting_equity
        self.equity = self.starting_equity
        self.halted = False

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
