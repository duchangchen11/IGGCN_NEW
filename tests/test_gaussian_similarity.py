import math

import pytest
import torch

from research.iggcn.interaction import gaussian_similarity


def test_identical_self_features_have_unit_similarity() -> None:
    features = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
    )

    weights = gaussian_similarity(features, sigma=3.0)

    torch.testing.assert_close(
        torch.diagonal(weights, dim1=-2, dim2=-1),
        torch.ones((1, 2)),
    )


def test_larger_sigma_decays_more_slowly_at_fixed_distance() -> None:
    features = torch.zeros((1, 2, 2, 1), dtype=torch.float64)
    features[0, 0, 1, 0] = 2.0

    weights = [gaussian_similarity(features, sigma=value)[0, 0, 1] for value in range(1, 6)]

    assert all(left < right for left, right in zip(weights, weights[1:]))
    expected = [math.exp(-(2.0**2) / (2.0 * value**2)) for value in range(1, 6)]
    torch.testing.assert_close(torch.stack(weights), torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("sigma", [0.0, -1.0, math.inf, math.nan])
def test_invalid_bandwidth_is_rejected(sigma: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        gaussian_similarity(torch.zeros((1, 2, 2, 1)), sigma=sigma)
