"""Create a reproducible index of contiguous ETH/UCY 8+12 sample windows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.iggcn.trajectory_data import iter_target_windows


DEFAULT_DATA_ROOT = Path(
    "/media/lrj/54926A1D926A0438/datasets/iggcn_eth_ucy/raw"
)
DEFAULT_OUTPUT = Path("results/iggcn_sigma_diagnostic/dataset_windows.csv")
SCENE_COMPONENTS = {
    "ETH": {"biwi_eth": "biwi_eth.txt"},
    "HOTEL": {"biwi_hotel": "biwi_hotel.txt"},
    "UNIV": {
        "students001": "students001.txt",
        "students003": "students003.txt",
    },
    "ZARA1": {"crowds_zara01": "crowds_zara01.txt"},
    "ZARA2": {"crowds_zara02": "crowds_zara02.txt"},
}
FIELDNAMES = [
    "sample_id",
    "scene",
    "component",
    "target_pedestrian_id",
    "start_frame",
    "observed_length",
    "predicted_length",
    "observed_frames",
    "observed_xy",
    "v_last",
    "v_mean",
    "a_mean",
    "turning",
    "density",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        counts: dict[str, int] = {scene: 0 for scene in SCENE_COMPONENTS}
        for scene, components in SCENE_COMPONENTS.items():
            for component, filename in components.items():
                path = args.data_root / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                for window in iter_target_windows(
                    path=path,
                    scene=scene,
                    component=component,
                ):
                    counts[scene] += 1
                    features = window.state_features
                    writer.writerow(
                        {
                            "sample_id": window.sample_id,
                            "scene": scene,
                            "component": component,
                            "target_pedestrian_id": window.target_pedestrian_id,
                            "start_frame": window.start_frame,
                            "observed_length": len(window.observed_xy),
                            "predicted_length": len(window.future_xy),
                            "observed_frames": ";".join(
                                str(int(frame)) for frame in window.observed_frames
                            ),
                            "observed_xy": ";".join(
                                f"{x:.8g}:{y:.8g}" for x, y in window.observed_xy
                            ),
                            **features,
                        }
                    )
    total = sum(counts.values())
    print(f"Wrote {total} target windows to {args.output}")
    for scene, count in counts.items():
        print(f"{scene}: {count}")


if __name__ == "__main__":
    main()
