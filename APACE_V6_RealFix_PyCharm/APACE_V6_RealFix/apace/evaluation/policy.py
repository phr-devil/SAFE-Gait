from __future__ import annotations


def choose_action(net_values: dict[str, float]) -> tuple[str, float]:
    if not net_values:
        return "STOP", 0.0
    action, value = max(net_values.items(), key=lambda kv: kv[1])
    if value <= 0:
        return "STOP", 0.0
    return action, float(value)
