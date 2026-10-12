import torch
from apace.models.value import DistributionalValueModel, quantile_loss
from apace.evaluation.policy import choose_action


def test_value_model_returns_ordered_quantiles_after_projection():
    model = DistributionalValueModel(state_dim=8, n_actions=3, hidden_dim=16)
    q = model(torch.randn(4, 8), torch.tensor([0,1,2,1]), torch.ones(4,1))
    assert q.shape == (4, 3)
    assert torch.all(q[:, 0] <= q[:, 1])
    assert torch.all(q[:, 1] <= q[:, 2])


def test_quantile_loss_zero_for_exact_target():
    pred = torch.tensor([[2.,2.,2.]])
    target = torch.tensor([2.])
    loss = quantile_loss(pred, target)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_policy_stops_when_all_net_values_are_nonpositive():
    action, score = choose_action({"slow": -0.2, "normal": 0.0, "fast": -1.0})
    assert action == "STOP"
    assert score == 0.0


def test_policy_chooses_highest_positive_action():
    action, score = choose_action({"slow": 0.2, "normal": 0.8, "fast": 0.3})
    assert action == "normal"
    assert score == 0.8
