import pytest
import torch

import research.iggcn.model as model_module
from research.iggcn.model import IGGCN


@pytest.mark.parametrize("sigma", [1, 2, 3, 4, 5])
def test_sigma_is_fixed_and_not_an_optimizer_parameter(sigma: int) -> None:
    model = IGGCN(sigma=sigma)

    assert model.sigma == float(sigma)
    assert all("sigma" not in name for name, _ in model.named_parameters())


def test_same_fixed_sigma_reaches_spatial_and_temporal_branches(monkeypatch) -> None:
    recorded_sigmas = []
    original = model_module.gaussian_similarity

    def record_sigma(features, sigma):
        recorded_sigmas.append(sigma)
        return original(features, sigma)

    monkeypatch.setattr(model_module, "gaussian_similarity", record_sigma)
    model = IGGCN(sigma=2)
    observed = torch.randn(1, 8, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    output = model(observed, mask)

    assert recorded_sigmas == [2.0, 2.0]
    assert output.shape == (1, 2, 12, 5)
    assert torch.isfinite(output).all()
