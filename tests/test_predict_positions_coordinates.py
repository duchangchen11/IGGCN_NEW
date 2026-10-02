import torch

from research.iggcn.model import IGGCN


def test_predict_positions_adds_last_observed_position_once() -> None:
    model = IGGCN(sigma=3).eval()
    parameters = torch.zeros((1, 1, 12, 5), dtype=torch.float32)
    parameters[0, 0, 0, :2] = torch.tensor([0.12, -0.08])
    parameters[0, 0, 1, :2] = torch.tensor([-0.15, 0.06])
    last_observation = torch.tensor([[[1.25, -0.50]]])

    predicted = model.predict_positions(parameters, last_observation)
    expected = parameters[..., :2] + last_observation[:, :, None]

    torch.testing.assert_close(predicted, expected)
    torch.testing.assert_close(predicted[0, 0, 0], torch.tensor([1.37, -0.58]))
    torch.testing.assert_close(predicted[0, 0, 1], torch.tensor([1.10, -0.44]))
    assert not torch.allclose(
        predicted[0, 0, 1], last_observation[0, 0] + parameters[0, 0, :2, :2].sum(0)
    )
