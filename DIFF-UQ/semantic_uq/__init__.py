"""
Semantic Uncertainty Quantification Module

This module unifies generating uncertainty (epistemic/generative) from either:
1) Precomputed directories of model variations (`{path}/0/imgs`, `{path}/1/imgs`, etc)
2) On-the-fly spatial+pixel perturbations of base images inline.

Examples:
---------
from semantic_uq.api import compute_uncertainty_precomputed, compute_uncertainty_onthefly

# 1. From M directories (precomputed samples)
uncertainty_scores = compute_uncertainty_precomputed(
    path="/path/to/samples", 
    m_samples=6, 
    encoder_name="clip"
)

# 2. On-the-fly perturbations from a single dataset directory
uncertainty_scores_fly = compute_uncertainty_onthefly(
    base_dir="/path/to/base/imgs",
    m_samples=10,
    noise_scale=0.02,
    use_transforms=True,
    encoder_name="clip"
)
"""

from .api import compute_uncertainty_precomputed, compute_uncertainty_onthefly
from .perturbations import ImagePerturber
from .entropy import exact_gaussian_entropy, gaussian_entropy
from .encoders import build_encoder

__all__ = [
    "compute_uncertainty_precomputed",
    "compute_uncertainty_onthefly",
    "ImagePerturber",
    "exact_gaussian_entropy",
    "gaussian_entropy",
    "build_encoder"
]
