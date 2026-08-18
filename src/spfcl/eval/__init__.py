"""Evaluation primitives for encoding forgetting and scientific-probe fidelity."""

from .causal import (
    ConditionalProbeResult,
    DoseResponseResult,
    PairedSwapResult,
    conditional_nuisance_probe,
    dose_response_slope,
    nuisance_projection_strength,
    orthogonal_subspace_ablation,
    paired_nuisance_swap_effect,
)
from .metrics import encoding_metrics, forgetting, localizer_contrast_metrics

__all__ = [
    "ConditionalProbeResult",
    "DoseResponseResult",
    "PairedSwapResult",
    "conditional_nuisance_probe",
    "dose_response_slope",
    "encoding_metrics",
    "forgetting",
    "localizer_contrast_metrics",
    "nuisance_projection_strength",
    "orthogonal_subspace_ablation",
    "paired_nuisance_swap_effect",
]
