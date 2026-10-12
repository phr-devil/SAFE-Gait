from apace.pipeline import run_synthetic_pipeline
import math


def test_quick_pipeline_runs_and_emits_core_metrics(tmp_path):
    metrics = run_synthetic_pipeline(output_dir=tmp_path, seed=7, quick=True)
    assert "base_mae" in metrics
    assert "calibrated_mae" in metrics
    assert "mean_references" in metrics
    assert "guard_status" in metrics
    assert 0 <= metrics["mean_references"] <= 2
    assert math.isfinite(metrics['base_mae']) and metrics['base_mae'] >= 0
    assert math.isfinite(metrics['calibrated_mae']) and metrics['calibrated_mae'] >= 0
    # This is an execution test, not a guarantee that short training must succeed scientifically.
    # In particular, STOP is valid and worsening errors must not be declared a research GO.
    if metrics['relative_improvement'] <= 0:
        assert metrics['guard_status'] != 'GO'
    assert (tmp_path / "metrics.json").exists()
