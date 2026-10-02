"""Rebuild only complete graph-node tracks from saved per-target trajectories.

This is a fallback for controlled model ablations when the raw ETH/UCY volume is
unmounted. The output is rounded serialized trajectory data, not a replacement
for the original dataset. It is sufficient to reproduce this implementation's
complete-8+12 graph nodes because partial tracks are not used as model nodes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "results/iggcn_sigma_diagnostic/per_sample_errors.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "results/baseline_debug/reconstructed_inputs_manifest.json"
SCENE_FILES = {
    "ETH": {"biwi_eth": "biwi_eth.txt"},
    "HOTEL": {"biwi_hotel": "biwi_hotel.txt"},
    "UNIV": {
        "students001": "students001.txt",
        "students003": "students003.txt",
    },
    "ZARA1": {"crowds_zara01": "crowds_zara01.txt"},
    "ZARA2": {"crowds_zara02": "crowds_zara02.txt"},
}


def parse_trajectory(value: str, expected_steps: int) -> np.ndarray:
    result = np.asarray(
        [[float(axis) for axis in point.split(":")] for point in value.split(";")],
        dtype=np.float64,
    )
    if result.shape != (expected_steps, 2):
        raise ValueError(f"expected {expected_steps} xy positions; got {result.shape}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-frame-stride", type=int, default=10)
    args = parser.parse_args()

    tracks: dict[tuple[str, str, str], dict[int, np.ndarray]] = defaultdict(dict)
    sample_counts = defaultdict(int)
    max_overlap_delta = 0.0
    with args.source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            component = row["component"]
            scene = row["scene"]
            if component not in SCENE_FILES[scene]:
                raise ValueError(f"unexpected component {scene}/{component}")
            pedestrian_id = row["target_pedestrian_id"].split(":", 1)[1]
            start = int(row["start_frame"])
            sequence = np.concatenate(
                (
                    parse_trajectory(row["observed_trajectory"], 8),
                    parse_trajectory(row["ground_truth_future"], 12),
                ),
                axis=0,
            )
            sample_counts[scene] += 1
            track = tracks[(scene, component, pedestrian_id)]
            for offset, xy in enumerate(sequence):
                frame = start + offset * args.raw_frame_stride
                if frame in track:
                    max_overlap_delta = max(
                        max_overlap_delta,
                        float(np.max(np.abs(track[frame] - xy))),
                    )
                else:
                    track[frame] = xy

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows_per_component = {}
    tracks_per_component = {}
    for scene, components in SCENE_FILES.items():
        for component, filename in components.items():
            subset = {
                pedestrian: frames
                for (track_scene, track_component, pedestrian), frames in tracks.items()
                if track_scene == scene and track_component == component
            }
            destination = args.output_root / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                (frame, float(pedestrian), xy)
                for pedestrian, frame_map in subset.items()
                for frame, xy in frame_map.items()
            ]
            rows.sort(key=lambda item: (item[0], item[1]))
            with destination.open("w", encoding="utf-8") as stream:
                for frame, pedestrian, xy in rows:
                    stream.write(
                        f"{frame:d}\t{pedestrian:.12g}\t{xy[0]:.12g}\t{xy[1]:.12g}\n"
                    )
            rows_per_component[f"{scene}/{component}"] = len(rows)
            tracks_per_component[f"{scene}/{component}"] = len(subset)

    manifest = {
        "source_csv": str(args.source),
        "derived_data_root": str(args.output_root),
        "purpose": "controlled mask ablation only; complete target tracks reconstructed from saved per-target windows",
        "source_coordinates_were_serialized_to_significant_digits": 7,
        "raw_frame_stride": args.raw_frame_stride,
        "source_target_windows_by_scene": dict(sample_counts),
        "reconstructed_tracks_by_component": tracks_per_component,
        "reconstructed_frame_rows_by_component": rows_per_component,
        "max_overlap_position_disagreement_m": max_overlap_delta,
        "limits": [
            "Not the raw Social-STGCNN text data and must not be described as raw-data validation.",
            "Contains only pedestrians with at least one complete contiguous 8+12 target window.",
            "Coordinates inherit the precision of saved trajectory strings.",
            "The HOTEL test scene is held out by the training driver; only the other four scenes enter training.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
