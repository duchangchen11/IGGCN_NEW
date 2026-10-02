"""Audit the saved Gaussian-mean trajectories and relative-to-world decoder."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.iggcn.model import IGGCN


SAMPLES = ROOT / "results/iggcn_sigma_diagnostic/per_sample_errors.csv"
OUTPUT = ROOT / "results/baseline_debug/decoder_coordinate_probe.json"


def parse_trajectory(value: str) -> np.ndarray:
    return np.asarray(
        [[float(axis) for axis in point.split(":")] for point in value.split(";")],
        dtype=np.float64,
    )


def main() -> None:
    model = IGGCN(sigma=3).eval()
    parameters = torch.zeros((1, 1, 12, 5), dtype=torch.float32)
    parameters[0, 0, 0, :2] = torch.tensor([0.12, -0.08])
    parameters[0, 0, 1, :2] = torch.tensor([-0.15, 0.06])
    last_observation = torch.tensor([[[1.25, -0.50]]])
    with torch.no_grad():
        decoded = model.predict_positions(parameters, last_observation)
    expected = parameters[..., :2] + last_observation[:, :, None]
    torch.testing.assert_close(decoded, expected)

    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    first_example = None
    max_ade_error = 0.0
    max_fde_error = 0.0
    first_step_gt_gaps = []
    first_step_prediction_gaps = []
    with SAMPLES.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            observed = parse_trajectory(row["observed_trajectory"])
            future = parse_trajectory(row["ground_truth_future"])
            predicted = parse_trajectory(row["predicted_future_mean"])
            distances = np.linalg.norm(predicted - future, axis=1)
            ade = float(distances.mean())
            fde = float(distances[-1])
            max_ade_error = max(max_ade_error, abs(ade - float(row["ADE"])))
            max_fde_error = max(max_fde_error, abs(fde - float(row["FDE"])))
            obs_last = observed[-1]
            gt_first = future[0]
            pred_first = predicted[0]
            first_step_gt_gaps.append(float(np.linalg.norm(gt_first - obs_last)))
            first_step_prediction_gaps.append(
                float(np.linalg.norm(pred_first - obs_last))
            )
            scene_rows[row["scene"]].append({"ADE": ade, "FDE": fde})
            if first_example is None:
                first_example = {
                    "sample_id": row["sample_id"],
                    "observed_last_xy": obs_last.tolist(),
                    "future_first_xy": gt_first.tolist(),
                    "prediction_first_mean_xy": pred_first.tolist(),
                    "predicted_first_offset_from_observed_last": (
                        pred_first - obs_last
                    ).tolist(),
                    "ground_truth_first_offset_from_observed_last": (
                        gt_first - obs_last
                    ).tolist(),
                }

    per_scene = {
        scene: {
            "targets": len(rows),
            "ADE": float(np.mean([row["ADE"] for row in rows])),
            "FDE": float(np.mean([row["FDE"] for row in rows])),
        }
        for scene, rows in scene_rows.items()
    }
    result = {
        "source": str(SAMPLES.relative_to(ROOT)),
        "prediction_parameterization": "future offsets from last observed position; add last observed xy exactly once; no cumulative integration",
        "synthetic_decoder_probe": {
            "last_observed_xy": last_observation[0, 0].tolist(),
            "first_two_mean_offsets": parameters[0, 0, :2, :2].tolist(),
            "decoded_first_two_positions": decoded[0, 0, :2].tolist(),
            "passed": True,
        },
        "saved_prediction_example": first_example,
        "first_step_displacement_from_observed_last_m": {
            "ground_truth_median": float(np.median(first_step_gt_gaps)),
            "prediction_median": float(np.median(first_step_prediction_gaps)),
            "ground_truth_p95": float(np.quantile(first_step_gt_gaps, 0.95)),
            "prediction_p95": float(
                np.quantile(first_step_prediction_gaps, 0.95)
            ),
        },
        "recomputed_from_saved_trajectories": {
            "per_scene": per_scene,
            "max_abs_per_target_ADE_difference": max_ade_error,
            "max_abs_per_target_FDE_difference": max_fde_error,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
