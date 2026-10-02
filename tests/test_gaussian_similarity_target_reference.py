import math

import torch

from research.iggcn.interaction import gaussian_similarity


def test_similarity_uses_each_targets_diagonal_scalar_reference() -> None:
    # [batch, target, source, scalar influence feature]
    refined_qk = torch.tensor(
        [[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=torch.float64
    )

    similarity = gaussian_similarity(refined_qk, sigma=3.0)
    expected_off_diagonal = math.exp(-1.0 / (2.0 * 3.0**2))

    torch.testing.assert_close(
        similarity,
        torch.tensor(
            [[[1.0, expected_off_diagonal], [expected_off_diagonal, 1.0]]],
            dtype=torch.float64,
        ),
    )
