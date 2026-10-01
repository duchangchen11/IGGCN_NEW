"""Individual-guided Gaussian similarity used by fixed-sigma experiments."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor


def gaussian_similarity(interaction_features: Tensor, sigma: float) -> Tensor:
    """Compare every neighbor/time feature with its target's diagonal feature.

    Args:
        interaction_features: Tensor shaped ``[..., target, source, feature]``.
            The two node axes must have the same length. For the temporal branch,
            the node axes represent time indices.
        sigma: Positive, finite scalar bandwidth. This is a fixed Python scalar;
            it is not a parameter and receives no optimizer state.

    Returns:
        Tensor shaped ``[..., target, source]`` containing
        ``exp(-||R_ij - R_ii||^2 / (2 * sigma^2))``.
    """
    if not isinstance(sigma, Real) or not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError(f"sigma must be a positive finite scalar, got {sigma!r}")
    if interaction_features.ndim < 3:
        raise ValueError(
            "interaction_features must have [..., target, source, feature] dimensions"
        )
    if interaction_features.shape[-3] != interaction_features.shape[-2]:
        raise ValueError("target and source axes must have the same length")
    if interaction_features.shape[-1] == 0:
        raise ValueError("feature dimension must be non-empty")

    count = interaction_features.shape[-2]
    diagonal_indices = torch.arange(count, device=interaction_features.device)
    self_features = interaction_features[
        ..., diagonal_indices, diagonal_indices, :
    ]
    squared_distance = (interaction_features - self_features.unsqueeze(-2)).square().sum(
        dim=-1
    )
    return torch.exp(squared_distance * (-0.5 / (float(sigma) ** 2)))
