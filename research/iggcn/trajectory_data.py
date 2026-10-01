"""ETH/UCY window extraction and observed-only motion-state features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class TargetWindow:
    sample_id: str
    scene: str
    component: str
    target_pedestrian_id: str
    start_frame: int
    observed_frames: np.ndarray
    observed_xy: np.ndarray
    future_xy: np.ndarray
    # Relative neighbor offsets are grouped by observed frame; they are context only.
    observed_neighbor_offsets: tuple[np.ndarray, ...]
    state_features: dict[str, float]


def _format_id(value: float) -> str:
    return f"{value:.12g}"


def _load_rows(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError(f"{path}: expected frame, pedestrian_id, x, y columns")
    if not np.isfinite(rows).all():
        raise ValueError(f"{path}: found NaN or Inf")
    pairs = rows[:, :2]
    if np.unique(pairs, axis=0).shape[0] != rows.shape[0]:
        raise ValueError(f"{path}: duplicate frame/pedestrian rows")
    return rows


def observed_state_features(
    observed_xy: np.ndarray,
    observed_neighbor_offsets: tuple[np.ndarray, ...],
    sample_period_seconds: float = 0.4,
    density_radius_meters: float = 2.0,
) -> dict[str, float]:
    """Compute speed, acceleration, turning, and density from observed frames only."""
    positions = np.asarray(observed_xy, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
        raise ValueError("observed_xy must have shape [observed_frames>=2, 2]")
    if len(observed_neighbor_offsets) != positions.shape[0]:
        raise ValueError("one neighbor-coordinate array is required per observed frame")

    velocities = np.diff(positions, axis=0) / sample_period_seconds
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(velocities, axis=0) / sample_period_seconds

    moving = speeds > 1e-8
    headings = np.arctan2(velocities[moving, 1], velocities[moving, 0])
    if headings.size > 1:
        heading_steps = np.diff(headings)
        wrapped_steps = np.arctan2(np.sin(heading_steps), np.cos(heading_steps))
        turning = float(np.abs(wrapped_steps).sum())
    else:
        turning = 0.0

    neighbor_counts = []
    for frame_neighbors in observed_neighbor_offsets:
        other_positions = np.asarray(frame_neighbors, dtype=np.float64).reshape(-1, 2)
        if other_positions.size == 0:
            neighbor_counts.append(0.0)
            continue
        distances = np.linalg.norm(other_positions, axis=1)
        neighbor_counts.append(float(np.count_nonzero(distances <= density_radius_meters)))

    return {
        "v_last": float(speeds[-1]),
        "v_mean": float(speeds.mean()),
        "a_mean": float(np.linalg.norm(accelerations, axis=1).mean())
        if accelerations.size
        else 0.0,
        "turning": turning,
        "density": float(np.mean(neighbor_counts)),
    }


def iter_target_windows(
    path: Path,
    scene: str,
    component: str,
    observed_length: int = 8,
    predicted_length: int = 12,
    raw_frame_stride: int = 10,
    sample_period_seconds: float = 0.4,
    density_radius_meters: float = 2.0,
) -> Iterator[TargetWindow]:
    """Yield contiguous 8+12 trajectory windows for each target pedestrian.

    Files are treated as independent video components. This is important for
    UNIV, whose two source clips have independent frame counters and IDs.
    """
    rows = _load_rows(path)
    total_length = observed_length + predicted_length
    all_people_by_frame: dict[int, dict[float, np.ndarray]] = {}
    for frame, pedestrian_id, x, y in rows:
        all_people_by_frame.setdefault(int(frame), {})[pedestrian_id] = np.array(
            [x, y], dtype=np.float64
        )

    for pedestrian_id in np.unique(rows[:, 1]):
        track = rows[rows[:, 1] == pedestrian_id]
        track = track[np.argsort(track[:, 0])]
        frames = track[:, 0].astype(np.int64)
        positions = track[:, 2:4]
        for start in range(0, len(track) - total_length + 1):
            end = start + total_length
            window_frames = frames[start:end]
            if not np.all(np.diff(window_frames) == raw_frame_stride):
                continue

            observed = positions[start : start + observed_length].copy()
            observed_frame_ids = window_frames[:observed_length].copy()
            neighbors_by_frame = []
            for frame, target_xy in zip(observed_frame_ids, observed):
                frame_people = all_people_by_frame.get(int(frame), {})
                neighbors = [
                    person_xy - target_xy
                    for other_id, person_xy in frame_people.items()
                    if other_id != pedestrian_id
                ]
                neighbors_by_frame.append(
                    np.asarray(neighbors, dtype=np.float64).reshape(-1, 2)
                )
            neighbor_tuple = tuple(neighbors_by_frame)
            state = observed_state_features(
                observed,
                neighbor_tuple,
                sample_period_seconds=sample_period_seconds,
                density_radius_meters=density_radius_meters,
            )
            start_frame = int(window_frames[0])
            target_id = f"{component}:{_format_id(float(pedestrian_id))}"
            yield TargetWindow(
                sample_id=f"{scene}:{target_id}:{start_frame}",
                scene=scene,
                component=component,
                target_pedestrian_id=target_id,
                start_frame=start_frame,
                observed_frames=observed_frame_ids,
                observed_xy=observed,
                future_xy=positions[start + observed_length : end].copy(),
                observed_neighbor_offsets=neighbor_tuple,
                state_features=state,
            )
