from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
from collections.abc import Iterable, Sequence
import numpy as np
import torch

from apace.models.calibration import CalibrationBeliefEncoder, PersonalizedPredictor


@dataclass(frozen=True)
class EncodedTrial:
    subject_id: str
    trial_id: str
    condition: str
    embedding: np.ndarray
    base_mean: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class ValueRecord:
    subject_id: str
    state: np.ndarray
    action: str
    cost_weight: float
    net_value: float
    stop_error: float
    action_cost_to_go: float


def planning_state(belief, remaining, action_to_index, budget):
    """Only acquired-reference belief, remaining protocol counts and budget; no future labels."""
    counts = np.zeros(len(action_to_index), dtype=np.float32)
    for row in remaining:
        counts[action_to_index[row.condition]] += 1
    return np.concatenate([belief, np.log1p(np.asarray([budget], dtype=np.float32)),
                           np.log1p(counts)]).astype(np.float32)


class CounterfactualEvaluator:
    def __init__(
        self,
        belief_encoder: CalibrationBeliefEncoder,
        personalized_predictor: PersonalizedPredictor,
        action_to_index: dict[str, int],
        device: torch.device | str = "cpu",
    ) -> None:
        self.belief_encoder = belief_encoder
        self.personalized_predictor = personalized_predictor
        self.action_to_index = action_to_index
        self.device = torch.device(device)
        self.belief_encoder.eval()
        self.personalized_predictor.eval()

    @property
    def belief_dim(self) -> int:
        return self.belief_encoder.belief_dim

    @torch.no_grad()
    def state_vector(self, context: Sequence[EncodedTrial]) -> np.ndarray:
        if not context:
            return np.zeros(self.belief_dim, dtype=np.float32)
        emb = torch.tensor(np.stack([t.embedding for t in context]), dtype=torch.float32, device=self.device).unsqueeze(0)
        residual = torch.tensor(np.stack([t.target - t.base_mean for t in context]), dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = torch.tensor([[self.action_to_index[t.condition] for t in context]], dtype=torch.long, device=self.device)
        mask = torch.ones((1, len(context)), dtype=torch.bool, device=self.device)
        return self.belief_encoder(emb, residual, actions, mask).squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def query_error(self, context: Sequence[EncodedTrial], query: Sequence[EncodedTrial]) -> float:
        return float(self.query_errors(context, query).mean())

    @torch.no_grad()
    def query_errors(self, context: Sequence[EncodedTrial], query: Sequence[EncodedTrial]) -> np.ndarray:
        """Absolute error per trial and target, in the model's normalized units."""
        if not query:
            raise ValueError("query cannot be empty")
        target = np.stack([t.target for t in query])
        if not context:
            pred = np.stack([t.base_mean for t in query])
        else:
            state = torch.tensor(self.state_vector(context), dtype=torch.float32, device=self.device).unsqueeze(0)
            emb = torch.tensor(np.stack([t.embedding for t in query]), dtype=torch.float32, device=self.device)
            belief = state.expand(len(query), -1)
            base = torch.tensor(np.stack([t.base_mean for t in query]), dtype=torch.float32, device=self.device)
            pred, _ = self.personalized_predictor.predict(emb, belief, base)
            pred = pred.cpu().numpy()
        return np.abs(target - pred)


def build_protocol_value_records(
    trials: Sequence[EncodedTrial],
    evaluator: CounterfactualEvaluator,
    calibration_ids: set[str],
    query_ids: set[str],
    lambda_values: Sequence[float],
    max_depth: int = 2,
    action_cost: float = 1.0,
) -> list[ValueRecord]:
    """Build long-horizon protocol values with exact stochastic DP.

    An action is a protocol/condition. If multiple unused calibration trials
    exist for that condition, the next outcome is treated as uniformly random.
    Query trials are immutable and disjoint from calibration candidates.
    """
    if calibration_ids & query_ids:
        raise ValueError("calibration and query ids must be disjoint")
    lookup = {t.trial_id: t for t in trials}
    if not calibration_ids <= lookup.keys() or not query_ids <= lookup.keys():
        raise ValueError("unknown calibration/query trial id")
    candidates = [lookup[i] for i in sorted(calibration_ids)]
    query = [lookup[i] for i in sorted(query_ids)]
    by_action: dict[str, list[str]] = defaultdict(list)
    for t in candidates:
        by_action[t.condition].append(t.trial_id)
    actions = sorted(by_action)
    action_to_index = getattr(evaluator, 'action_to_index', {a:i for i,a in enumerate(actions)})
    subject_id = trials[0].subject_id if trials else "unknown"
    records: list[ValueRecord] = []

    def context_from_state(state: tuple[str, ...]) -> list[EncodedTrial]:
        return [lookup[i] for i in state]

    @lru_cache(maxsize=None)
    def error(state):
        return evaluator.query_error(context_from_state(state), query)

    @lru_cache(maxsize=None)
    def vector(state):
        return evaluator.state_vector(context_from_state(state))

    for lam in lambda_values:
        @lru_cache(maxsize=None)
        def optimal_cost(state: tuple[str, ...], remaining: int) -> float:
            best = error(state)
            if remaining <= 0:
                return best
            used = set(state)
            for action in actions:
                next_ids = [i for i in by_action[action] if i not in used]
                if not next_ids:
                    continue
                future = np.mean([
                    optimal_cost(tuple(sorted((*state, i))), remaining - 1) for i in next_ids
                ])
                best = min(best, float(lam) * action_cost + float(future))
            return float(best)

        visited: set[tuple[str, ...]] = set()

        def visit(state: tuple[str, ...], depth: int) -> None:
            if state in visited or depth > max_depth:
                return
            visited.add(state)
            stop_error = error(state)
            used = set(state)
            state_vec = planning_state(vector(state), [t for t in candidates if t.trial_id not in used],
                                       action_to_index, max_depth-depth)
            if depth >= max_depth:
                return
            for action in actions:
                next_ids = [i for i in by_action[action] if i not in used]
                if not next_ids:
                    continue
                q_cost = float(lam) * action_cost + float(np.mean([
                    optimal_cost(tuple(sorted((*state, i))), max_depth - depth - 1) for i in next_ids
                ]))
                records.append(ValueRecord(
                    subject_id=subject_id,
                    state=state_vec.copy(),
                    action=action,
                    cost_weight=float(lam),
                    net_value=float(stop_error - q_cost),
                    stop_error=float(stop_error),
                    action_cost_to_go=q_cost,
                ))
                for i in next_ids:
                    visit(tuple(sorted((*state, i))), depth + 1)

        visit(tuple(), 0)
    return records
