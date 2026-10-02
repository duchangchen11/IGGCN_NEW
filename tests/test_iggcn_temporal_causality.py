import torch
import pytest

from research.iggcn.model import IGGCN


def test_temporal_graph_uses_only_current_and_history() -> None:
    torch.manual_seed(19)
    model = IGGCN(sigma=3).eval()
    observed = torch.randn(1, 8, 2, 2)
    changed_future = observed.clone()
    target_time = 3
    changed_future[:, target_time + 1 :] += torch.randn_like(
        changed_future[:, target_time + 1 :]
    ) * 5.0
    pedestrian_mask = torch.ones((1, 2), dtype=torch.bool)

    with torch.no_grad():
        features_a, _ = model._temporal_graph_branch(
            model._displacements(observed).transpose(1, 2), pedestrian_mask
        )
        features_b, _ = model._temporal_graph_branch(
            model._displacements(changed_future).transpose(1, 2), pedestrian_mask
        )

    torch.testing.assert_close(
        features_a[:, : target_time + 1],
        features_b[:, : target_time + 1],
        atol=1e-6,
        rtol=1e-6,
    )


def test_temporal_adjacency_rows_allow_only_source_at_or_before_target() -> None:
    torch.manual_seed(29)
    model = IGGCN(sigma=3).eval()
    observed = torch.randn(1, 8, 2, 2)
    pedestrian_mask = torch.ones((1, 2), dtype=torch.bool)

    with torch.no_grad():
        _, diagnostics = model._temporal_graph_branch(
            model._displacements(observed).transpose(1, 2),
            pedestrian_mask,
            return_diagnostics=True,
        )

    mask = diagnostics["temporal_causal_mask"]
    adjacency = diagnostics["temporal_adjacency"]
    assert torch.equal(mask, torch.tril(torch.ones_like(mask)))
    assert torch.count_nonzero(adjacency[..., 2, 3:]) == 0
    assert torch.count_nonzero(adjacency[..., 2, :3]) > 0


@pytest.mark.parametrize("direction", ["tril", "triu"])
def test_temporal_mask_direction_is_configurable_without_parameter_changes(direction):
    model = IGGCN(sigma=3, temporal_mask_direction=direction).eval()
    observed = torch.randn(1, 8, 2, 2)
    pedestrian_mask = torch.ones((1, 2), dtype=torch.bool)

    with torch.no_grad():
        _, diagnostics = model._temporal_graph_branch(
            model._displacements(observed).transpose(1, 2),
            pedestrian_mask,
            return_diagnostics=True,
        )

    expected = getattr(torch, direction)(torch.ones_like(diagnostics["temporal_causal_mask"]))
    assert torch.equal(diagnostics["temporal_causal_mask"], expected)
    assert all(
        torch.isfinite(parameter).all() for parameter in model.parameters()
    )


def test_invalid_temporal_mask_direction_is_rejected():
    with pytest.raises(ValueError, match="temporal_mask_direction"):
        IGGCN(temporal_mask_direction="diagonal")
