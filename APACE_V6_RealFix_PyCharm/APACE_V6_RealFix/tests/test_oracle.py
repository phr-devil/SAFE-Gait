from apace.oracle.dp import FiniteHorizonOracle


def test_dynamic_programming_prefers_useful_calibration_then_stop():
    # State is number of references. One action "normal" reduces error once.
    def terminal_error(state):
        return {0: 10.0, 1: 4.0, 2: 3.9}[state]

    def transition(state, action):
        return min(2, state + 1)

    oracle = FiniteHorizonOracle(
        actions=("normal",), transition=transition, terminal_error=terminal_error,
        action_cost={"normal": 1.0}, horizon=2,
    )
    value, action = oracle.solve(0, cost_weight=1.0)
    assert action == "normal"
    next_value, next_action = oracle.solve(1, cost_weight=1.0)
    assert next_action == "STOP"
    assert value < 10.0
    assert next_value == 4.0
