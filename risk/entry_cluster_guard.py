from dataclasses import dataclass
from typing import Dict


@dataclass
class ClusterDecision:
    allow: bool
    reason: str


class EntryClusterGuard:
    """
    Prevents clustered entries:
    - cooldown bars after any trade
    - max consecutive same-side entries in a window
    - pause after loss streak
    """

    def __init__(self, cfg: Dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.cooldown_bars = int(cfg.get("cooldown_bars_after_trade", 2))
        self.max_same_side = int(cfg.get("max_same_side_entries", 2))
        self.window_bars = int(cfg.get("window_bars", 12))
        self.block_after_losses = int(cfg.get("block_after_losses", 2))
        self.pause_bars_after_loss_streak = int(cfg.get("pause_bars_after_loss_streak", 8))

        self._last_trade_bar: Dict[str, int] = {}
        self._last_side: Dict[str, str] = {}
        self._same_side_count: Dict[str, int] = {}
        self._same_side_window_start: Dict[str, int] = {}
        self._loss_streak: Dict[str, int] = {}
        self._pause_until_bar: Dict[str, int] = {}

    def can_enter(self, pair: str, side: str, bar_index: int) -> ClusterDecision:
        if not self.enabled:
            return ClusterDecision(True, "cluster_guard_disabled")

        pause_until = self._pause_until_bar.get(pair)
        if pause_until is not None and bar_index < pause_until:
            return ClusterDecision(False, f"paused_until_bar={pause_until}")

        last_bar = self._last_trade_bar.get(pair)
        if last_bar is not None and (bar_index - last_bar) <= self.cooldown_bars:
            return ClusterDecision(False, f"cooldown_active({bar_index-last_bar}<={self.cooldown_bars})")

        prev_side = self._last_side.get(pair)

        if prev_side == side:
            start = self._same_side_window_start.get(pair, bar_index)
            if (bar_index - start) > self.window_bars:
                self._same_side_window_start[pair] = bar_index
                self._same_side_count[pair] = 1
            else:
                self._same_side_count[pair] = self._same_side_count.get(pair, 1) + 1

            if self._same_side_count[pair] > self.max_same_side:
                return ClusterDecision(False, f"same_side_cluster>{self.max_same_side}")
        else:
            self._same_side_window_start[pair] = bar_index
            self._same_side_count[pair] = 1

        return ClusterDecision(True, "cluster_ok")

    def on_trade_closed(self, pair: str, side: str, bar_index: int, pnl: float):
        if not self.enabled:
            return

        self._last_trade_bar[pair] = int(bar_index)
        self._last_side[pair] = str(side)

        if pnl < 0:
            self._loss_streak[pair] = self._loss_streak.get(pair, 0) + 1
            if self._loss_streak[pair] >= self.block_after_losses:
                self._pause_until_bar[pair] = int(bar_index) + self.pause_bars_after_loss_streak
        else:
            self._loss_streak[pair] = 0
