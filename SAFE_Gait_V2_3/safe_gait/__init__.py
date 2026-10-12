"""SAFE-Gait V2.3: coherent hierarchical one-shot residual personalization."""
from .models.hierarchical_residual import HierarchicalResidualModel, PosteriorResult
from .models.damping import GlobalDampingCalibrator

__all__ = ["HierarchicalResidualModel", "PosteriorResult", "GlobalDampingCalibrator"]
__version__ = "2.3.0"
