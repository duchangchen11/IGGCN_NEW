"""Scene-level trajectory graphs for paper-protocol IGGCN training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .trajectory_data import iter_target_windows


@dataclass(frozen=True)
class SceneGraphWindow:
    scene: str
    component: str
    start_frame: int
    pedestrian_ids: tuple[str, ...]
    observed_xy: np.ndarray  # [pedestrians, observed frames, 2]
    future_xy: np.ndarray  # [pedestrians, prediction frames, 2]
    state_features: dict[str, np.ndarray]  # one observed-only value per pedestrian


STATE_NAMES = ("v_last", "v_mean", "a_mean", "turning", "density")


def load_scene_graph_windows(
    component_paths: Iterable[tuple[str, str, Path]],
    observed_length: int = 8,
    predicted_length: int = 12,
    raw_frame_stride: int = 10,
    sample_period_seconds: float = 0.4,
    density_radius_meters: float = 2.0,
) -> list[SceneGraphWindow]:
    """Join target windows with the same scene/component/start into one graph."""
    grouped: dict[tuple[str, str, int], list] = defaultdict(list)
    for scene, component, path in component_paths:
        for window in iter_target_windows(
            path,
            scene,
            component,
            observed_length=observed_length,
            predicted_length=predicted_length,
            raw_frame_stride=raw_frame_stride,
            sample_period_seconds=sample_period_seconds,
            density_radius_meters=density_radius_meters,
        ):
            grouped[(scene, component, window.start_frame)].append(window)

    result = []
    for (scene, component, start_frame), windows in sorted(grouped.items()):
        windows.sort(key=lambda window: window.target_pedestrian_id)
        result.append(
            SceneGraphWindow(
                scene=scene,
                component=component,
                start_frame=start_frame,
                pedestrian_ids=tuple(
                    window.target_pedestrian_id for window in windows
                ),
                observed_xy=np.stack(
                    [window.observed_xy for window in windows]
                ).astype(np.float32),
                future_xy=np.stack(
                    [window.future_xy for window in windows]
                ).astype(np.float32),
                state_features={
                    name: np.asarray(
                        [window.state_features[name] for window in windows],
                        dtype=np.float32,
                    )
                    for name in STATE_NAMES
                },
            )
        )
    return result


def chronological_train_validation_split(
    windows: list[SceneGraphWindow],
    validation_fraction: float = 0.1,
    purge_windows: int = 19,
) -> tuple[list[int], list[int]]:
    """Split each clip chronologically and purge overlapping training windows."""
    by_component: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        by_component[(window.scene, window.component)].append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in by_component.values():
        indices.sort(key=lambda index: windows[index].start_frame)
        if len(indices) < 3:
            train_indices.extend(indices)
            continue
        validation_count = max(1, int(np.ceil(len(indices) * validation_fraction)))
        split_at = len(indices) - validation_count
        train_end = max(0, split_at - purge_windows)
        train_indices.extend(indices[:train_end])
        validation_indices.extend(indices[split_at:])

    if not validation_indices:
        raise ValueError("chronological split produced no validation windows")
    if not train_indices:
        raise ValueError("chronological split produced no training windows")
    return train_indices, validation_indices


class SceneGraphDataset(Dataset):
    def __init__(self, windows: list[SceneGraphWindow], indices: list[int]):
        self.windows = windows
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> SceneGraphWindow:
        return self.windows[self.indices[item]]


def collate_scene_graphs(items: list[SceneGraphWindow]) -> dict:
    batch_size = len(items)
    observed_steps = items[0].observed_xy.shape[1]
    future_steps = items[0].future_xy.shape[1]
    max_pedestrians = max(item.observed_xy.shape[0] for item in items)
    observed = torch.zeros(
        batch_size, observed_steps, max_pedestrians, 2, dtype=torch.float32
    )
    future = torch.zeros(
        batch_size, future_steps, max_pedestrians, 2, dtype=torch.float32
    )
    pedestrian_mask = torch.zeros(
        batch_size, max_pedestrians, dtype=torch.bool
    )
    states = {
        name: torch.zeros(batch_size, max_pedestrians, dtype=torch.float32)
        for name in STATE_NAMES
    }
    for batch_index, item in enumerate(items):
        count = item.observed_xy.shape[0]
        observed[batch_index, :, :count] = torch.from_numpy(
            item.observed_xy.transpose(1, 0, 2).copy()
        )
        future[batch_index, :, :count] = torch.from_numpy(
            item.future_xy.transpose(1, 0, 2).copy()
        )
        pedestrian_mask[batch_index, :count] = True
        for name in STATE_NAMES:
            states[name][batch_index, :count] = torch.from_numpy(
                item.state_features[name]
            )
    return {
        "observed_xy": observed,
        "future_xy": future,
        "pedestrian_mask": pedestrian_mask,
        "state_features": states,
        "metadata": items,
    }
