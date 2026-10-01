"""Paper-based IGGCN implementation with an explicitly fixed Gaussian sigma."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import deform_conv2d

from .interaction import gaussian_similarity


class ResidualDeformableLayer(nn.Module):
    """Single-channel 3x3 deformable convolution with a residual connection."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Conv2d(1, 18, kernel_size=3, padding=1)
        self.weight = nn.Parameter(torch.empty(1, 1, 3, 3))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        offsets = self.offset(inputs)
        update = deform_conv2d(
            inputs,
            offsets,
            self.weight,
            self.bias,
            padding=(1, 1),
        )
        return inputs + update


class DeformableInteractionBranch(nn.Module):
    def __init__(self, layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualDeformableLayer() for _ in range(layers)]
        )

    def forward(self, maps: Tensor, valid_map: Tensor | None = None) -> Tensor:
        output = maps
        for layer in self.layers:
            output = layer(output)
            if valid_map is not None:
                output = output * valid_map
        return output


class CausalTemporalLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=0,
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(
            self.convolution(torch.nn.functional.pad(inputs, (self.left_padding, 0)))
        )


class TrajectoryTemporalConv(nn.Module):
    """Five-layer TCN followed by the paper's bivariate-Gaussian parameters."""

    def __init__(
        self,
        input_dimension: int = 64,
        channels: int = 32,
        layers: int = 5,
        prediction_steps: int = 12,
    ) -> None:
        super().__init__()
        dilations = [2**index for index in range(layers)]
        blocks = []
        in_channels = input_dimension
        for dilation in dilations:
            blocks.append(CausalTemporalLayer(in_channels, channels, dilation))
            in_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(channels, prediction_steps * 5)
        self.prediction_steps = prediction_steps

    def forward(self, sequence: Tensor) -> Tensor:
        # sequence: [batch, pedestrians, observed time, graph dimension]
        batch, pedestrians, steps, dimension = sequence.shape
        temporal = sequence.reshape(batch * pedestrians, steps, dimension).transpose(1, 2)
        encoded = self.blocks(temporal)[:, :, -1]
        parameters = self.output(encoded).reshape(
            batch, pedestrians, self.prediction_steps, 5
        )
        return parameters


class IGGCN(nn.Module):
    """Spatial/temporal IGGCN, using only a fixed scalar Gaussian bandwidth."""

    def __init__(
        self,
        sigma: float = 3.0,
        hidden_dimension: int = 64,
        tcn_channels: int = 24,
        deformable_layers: int = 4,
        tcn_layers: int = 5,
        prediction_steps: int = 12,
    ) -> None:
        super().__init__()
        if sigma not in (1, 2, 3, 4, 5):
            raise ValueError("the diagnostic accepts only fixed sigma in {1,2,3,4,5}")
        self.sigma = float(sigma)
        self.prediction_steps = prediction_steps

        self.spatial_embedding = nn.Linear(2, hidden_dimension)
        self.temporal_embedding = nn.Linear(2, hidden_dimension)
        self.spatial_query = nn.Linear(hidden_dimension, hidden_dimension)
        self.spatial_key = nn.Linear(hidden_dimension, hidden_dimension)
        self.temporal_query = nn.Linear(hidden_dimension, hidden_dimension)
        self.temporal_key = nn.Linear(hidden_dimension, hidden_dimension)
        self.spatial_deformable = DeformableInteractionBranch(deformable_layers)
        self.temporal_deformable = DeformableInteractionBranch(deformable_layers)
        self.spatial_graph_weight = nn.Linear(2, hidden_dimension, bias=False)
        self.temporal_graph_weight = nn.Linear(2, hidden_dimension, bias=False)
        self.temporal_conv = TrajectoryTemporalConv(
            input_dimension=hidden_dimension,
            channels=tcn_channels,
            layers=tcn_layers,
            prediction_steps=prediction_steps,
        )

    @staticmethod
    def _sinusoidal_encoding(steps: int, dimension: int, device, dtype) -> Tensor:
        positions = torch.arange(steps, device=device, dtype=dtype).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dimension)
        )
        encoding = torch.zeros(steps, dimension, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor[: encoding[:, 1::2].shape[1]])
        return encoding

    @staticmethod
    def _displacements(observed_xy: Tensor) -> Tensor:
        first_step = torch.zeros_like(observed_xy[:, :1])
        return torch.cat((first_step, observed_xy[:, 1:] - observed_xy[:, :-1]), dim=1)

    def _temporal_graph_branch(
        self,
        temporal_inputs: Tensor,
        pedestrian_mask: Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Build a causal temporal graph with target rows and source columns.

        For row ``t``, only the prefix through ``t`` is passed through the
        deformable layers. This is needed because a full 2-D convolution over
        the target/source time matrix can otherwise move a later query's
        features into an earlier row before the adjacency mask is applied.
        """
        batch_size, pedestrians, observed_steps, _ = temporal_inputs.shape
        temporal_embedding = self.temporal_embedding(temporal_inputs)
        temporal_embedding = temporal_embedding + self._sinusoidal_encoding(
            observed_steps,
            temporal_embedding.shape[-1],
            temporal_embedding.device,
            temporal_embedding.dtype,
        )[None, None]
        temporal_embedding = temporal_embedding * pedestrian_mask[:, :, None, None]
        temporal_query = self.temporal_query(temporal_embedding)
        temporal_key = self.temporal_key(temporal_embedding)
        # Explicit convention: row t is the target time, column q is a source.
        temporal_qk = torch.einsum("bntd,bnqd->bntq", temporal_query, temporal_key)

        valid_map = pedestrian_mask.reshape(batch_size * pedestrians, 1, 1, 1).to(
            temporal_inputs.dtype
        )
        causal_rows = []
        for target_time in range(observed_steps):
            prefix_length = target_time + 1
            prefix_qk = temporal_qk[:, :, :prefix_length, :prefix_length].reshape(
                batch_size * pedestrians,
                1,
                prefix_length,
                prefix_length,
            )
            prefix_maps = self.temporal_deformable(prefix_qk, valid_map)
            target_row = prefix_maps.reshape(
                batch_size, pedestrians, prefix_length, prefix_length
            )[:, :, -1, :]
            causal_rows.append(
                torch.nn.functional.pad(
                    target_row,
                    (0, observed_steps - prefix_length),
                )
            )
        temporal_maps = torch.stack(causal_rows, dim=2)

        eye_time = torch.eye(
            observed_steps,
            device=temporal_inputs.device,
            dtype=temporal_inputs.dtype,
        )
        # A[target, source] @ X[source] requires source <= target for causality.
        causal_mask = torch.tril(torch.ones_like(eye_time))
        temporal_similarity = gaussian_similarity(temporal_maps.unsqueeze(-1), self.sigma)
        temporal_adjacency = temporal_similarity * causal_mask[None, None]
        temporal_adjacency = temporal_adjacency + eye_time[None, None]

        temporal_features = torch.matmul(temporal_adjacency, temporal_inputs)
        temporal_features = self.temporal_graph_weight(temporal_features).transpose(1, 2)
        if return_diagnostics:
            return temporal_features, {
                "temporal_qk": temporal_qk,
                "temporal_maps": temporal_maps,
                "temporal_similarity": temporal_similarity,
                "temporal_adjacency": temporal_adjacency,
                "temporal_causal_mask": causal_mask,
            }
        return temporal_features, None

    def forward(
        self,
        observed_xy: Tensor,
        pedestrian_mask: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Return Gaussian parameters; optionally also expose graph diagnostics."""
        batch_size, observed_steps, pedestrians, _ = observed_xy.shape
        valid_nodes = pedestrian_mask[:, None, :, None].to(observed_xy.dtype)
        displacements = self._displacements(observed_xy) * valid_nodes

        # Spatial QK: [batch, observed time, target pedestrian, source pedestrian].
        spatial_embedding = self.spatial_embedding(displacements) * valid_nodes
        spatial_q = self.spatial_query(spatial_embedding)
        spatial_k = self.spatial_key(spatial_embedding)
        spatial_qk = torch.einsum("btnd,btmd->btnm", spatial_q, spatial_k)
        spatial_pair_mask = (
            pedestrian_mask[:, None, :, None]
            & pedestrian_mask[:, None, None, :]
        ).to(observed_xy.dtype)
        spatial_qk = spatial_qk * spatial_pair_mask
        spatial_maps = self.spatial_deformable(
            spatial_qk.reshape(batch_size * observed_steps, 1, pedestrians, pedestrians),
            spatial_pair_mask.expand(-1, observed_steps, -1, -1).reshape(
                batch_size * observed_steps, 1, pedestrians, pedestrians
            ),
        ).reshape(batch_size, observed_steps, pedestrians, pedestrians)
        spatial_similarity = gaussian_similarity(spatial_maps.unsqueeze(-1), self.sigma)

        # Temporal QK uses target time in rows and source time in columns.
        temporal_inputs = displacements.transpose(1, 2)
        temporal_features, temporal_diagnostics = self._temporal_graph_branch(
            temporal_inputs,
            pedestrian_mask,
            return_diagnostics=return_diagnostics,
        )

        # Individual-guided Gaussian adjacency. Sigma remains an immutable scalar.
        eye_nodes = torch.eye(pedestrians, device=observed_xy.device, dtype=observed_xy.dtype)
        spatial_adjacency = spatial_similarity
        spatial_adjacency = spatial_adjacency + eye_nodes[None, None]

        spatial_features = torch.matmul(spatial_adjacency, displacements)
        spatial_features = self.spatial_graph_weight(spatial_features)
        graph_features = torch.relu(spatial_features + temporal_features)
        graph_features = graph_features * valid_nodes

        parameters = self.temporal_conv(graph_features.transpose(1, 2))
        if return_diagnostics:
            diagnostics = {
                "spatial_qk": spatial_qk.reshape(
                    batch_size, observed_steps, pedestrians, pedestrians
                ),
                "spatial_maps": spatial_maps,
                "spatial_similarity": spatial_similarity,
                "spatial_adjacency": spatial_adjacency,
                **(temporal_diagnostics or {}),
            }
            return parameters, diagnostics
        return parameters

    def predict_positions(self, parameters: Tensor, last_observed_xy: Tensor) -> Tensor:
        means = parameters[..., :2]
        return means + last_observed_xy[:, :, None, :]


def bivariate_gaussian_nll(parameters: Tensor, targets: Tensor) -> Tensor:
    """Per-agent summed bivariate Gaussian NLL, as in the paper's Eq. (7)."""
    mean_x, mean_y = parameters[..., 0], parameters[..., 1]
    sigma_x = parameters[..., 2].clamp(-7.0, 7.0).exp()
    sigma_y = parameters[..., 3].clamp(-7.0, 7.0).exp()
    rho = parameters[..., 4].tanh().clamp(-0.999, 0.999)
    target_x, target_y = targets[..., 0], targets[..., 1]
    normalized_x = (target_x - mean_x) / sigma_x
    normalized_y = (target_y - mean_y) / sigma_y
    one_minus_rho_sq = (1.0 - rho.square()).clamp_min(1e-6)
    quadratic = (
        normalized_x.square()
        + normalized_y.square()
        - 2.0 * rho * normalized_x * normalized_y
    ) / (2.0 * one_minus_rho_sq)
    log_density = (
        -torch.log(torch.tensor(2.0 * torch.pi, device=targets.device, dtype=targets.dtype))
        - sigma_x.log()
        - sigma_y.log()
        - 0.5 * one_minus_rho_sq.log()
        - quadratic
    )
    return -log_density.sum(dim=-1)
