from __future__ import annotations

from functools import lru_cache
from typing import Callable, Hashable, Mapping, Sequence, Any


class FiniteHorizonOracle:
    """Exact finite-horizon cost-to-go for small calibration action spaces."""

    def __init__(
        self,
        actions: Sequence[str],
        transition: Callable[[Hashable, str], Hashable],
        terminal_error: Callable[[Hashable], float],
        action_cost: Mapping[str, float],
        horizon: int,
    ) -> None:
        self.actions = tuple(actions)
        self.transition = transition
        self.terminal_error = terminal_error
        self.action_cost = dict(action_cost)
        self.horizon = int(horizon)

    def solve(self, state: Hashable, cost_weight: float) -> tuple[float, str]:
        @lru_cache(maxsize=None)
        def recurse(s: Hashable, remaining: int) -> tuple[float, str]:
            stop_cost = float(self.terminal_error(s))
            best_value = stop_cost
            best_action = "STOP"
            if remaining <= 0:
                return best_value, best_action
            for action in self.actions:
                nxt = self.transition(s, action)
                future, _ = recurse(nxt, remaining - 1)
                value = cost_weight * float(self.action_cost[action]) + future
                if value < best_value:
                    best_value, best_action = value, action
            return best_value, best_action

        return recurse(state, self.horizon)
