def should_open_long(p_up: float, ema_diff: float, prob_threshold: float) -> bool:
    return (p_up >= prob_threshold) and (ema_diff > 0)


def should_open_short(p_up: float, ema_diff: float, prob_threshold: float) -> bool:
    # p_down = 1 - p_up
    return ((1.0 - p_up) >= prob_threshold) and (ema_diff < 0)
