import torch
from apace.models.encoder import PressureEncoder
from apace.models.predictor import ProbabilisticPredictor
from apace.models.calibration import CalibrationBeliefEncoder, PersonalizedPredictor


def test_pressure_encoder_output_shape():
    enc = PressureEncoder(n_sensors=4, hidden_dim=16, out_dim=12)
    x = torch.randn(3, 4, 32)
    coords = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    out = enc(x, coords)
    assert out.shape == (3, 12)


def test_probabilistic_predictor_returns_mean_and_scale():
    head = ProbabilisticPredictor(12, 2)
    mean, scale = head(torch.randn(5, 12))
    assert mean.shape == scale.shape == (5, 2)
    assert torch.all(scale > 0)


def test_calibration_belief_handles_empty_and_nonempty_context():
    belief = CalibrationBeliefEncoder(embedding_dim=12, target_dim=2, n_actions=3, belief_dim=8)
    empty = belief.empty(batch_size=2, device=torch.device("cpu"))
    assert empty.shape == (2, 8)
    emb = torch.randn(2, 3, 12)
    residual = torch.randn(2, 3, 2)
    actions = torch.tensor([[0, 1, 2], [2, 1, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    out = belief(emb, residual, actions, mask)
    assert out.shape == (2, 8)


def test_personalized_predictor_changes_with_belief():
    model = PersonalizedPredictor(embedding_dim=12, belief_dim=8, target_dim=2)
    h = torch.randn(4, 12)
    z0 = torch.zeros(4, 8)
    z1 = torch.ones(4, 8)
    m0, _ = model(h, z0)
    m1, _ = model(h, z1)
    assert not torch.allclose(m0, m1)


def test_personalized_predictor_can_apply_residual_correction_to_base_mean():
    model = PersonalizedPredictor(embedding_dim=12, belief_dim=8, target_dim=2)
    for param in model.parameters():
        torch.nn.init.zeros_(param)
    h = torch.randn(3, 12)
    belief = torch.zeros(3, 8)
    base = torch.tensor([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])
    mean, scale = model.predict(h, belief, base)
    assert torch.allclose(mean, base)
    assert torch.all(scale > 0)
