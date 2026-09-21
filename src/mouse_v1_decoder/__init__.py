"""Utilities for decoding drifting-grating labels from VISp spike counts."""

from .evaluation import (
    direction_to_orientation,
    evaluate_decoder,
    permutation_baseline,
    population_size_curve,
)

__all__ = [
    "direction_to_orientation",
    "evaluate_decoder",
    "permutation_baseline",
    "population_size_curve",
]
