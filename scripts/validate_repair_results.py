"""Check that repaired fold CSVs cover the same test targets as the baseline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = PROJECT_ROOT / "results/iggcn_sigma_diagnostic"
NEW_ROOT = PROJECT_ROOT / "results/iggcn_baseline_repair/full_sigma3_baseline"
CV_PATH = PROJECT_ROOT / "results/iggcn_baseline_repair/pre_fix_audit/protocol_and_constant_velocity.csv"
OUTPUT = NEW_ROOT / "validation_summary.json"
EXPECTED_SCENES = ("ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    old_samples = read_csv(OLD_ROOT / "per_sample_errors.csv")
    new_samples = read_csv(NEW_ROOT / "per_sample_errors.csv")
    fold_rows = read_csv(NEW_ROOT / "baseline_reproduction.csv")
    cv_rows = read_csv(CV_PATH)
    old_ids = {(row["scene"], row["sample_id"]) for row in old_samples}
    new_ids = {(row["scene"], row["sample_id"]) for row in new_samples}
    if len(old_ids) != len(old_samples) or len(new_ids) != len(new_samples):
        raise ValueError("Duplicate (scene, sample_id) values found")
    if old_ids != new_ids:
        missing = sorted(old_ids - new_ids)[:10]
        extra = sorted(new_ids - old_ids)[:10]
        raise ValueError(f"Test target IDs changed. missing={missing}; extra={extra}")
    if any(not math.isfinite(float(row[k])) for row in new_samples for k in ("ADE", "FDE")):
        raise ValueError("Non-finite repaired per-sample ADE/FDE")

    if {row["test_scene"] for row in fold_rows} != set(EXPECTED_SCENES):
        raise ValueError("Repaired fold CSV does not contain exactly the five scenes")
    if any(row["sigma"] != "3" or row["seed"] != "42" for row in fold_rows):
        raise ValueError("Unexpected sigma or seed in repaired baseline")

    per_scene = {}
    for scene in EXPECTED_SCENES:
        samples = [row for row in new_samples if row["scene"] == scene]
        fold = next(row for row in fold_rows if row["test_scene"] == scene)
        ade = sum(float(row["ADE"]) for row in samples) / len(samples)
        fde = sum(float(row["FDE"]) for row in samples) / len(samples)
        if len(samples) != int(fold["test_targets"]):
            raise ValueError(f"Target count mismatch for {scene}")
        if abs(ade - float(fold["ADE"])) > 1e-6 or abs(fde - float(fold["FDE"])) > 1e-6:
            raise ValueError(f"Per-sample metrics do not recreate fold CSV for {scene}")
        per_scene[scene] = {
            "targets": len(samples),
            "ADE": ade,
            "FDE": fde,
            "best_epoch": int(fold["best_epoch"]),
        }

    macro_ade = sum(row["ADE"] for row in per_scene.values()) / len(per_scene)
    macro_fde = sum(row["FDE"] for row in per_scene.values()) / len(per_scene)
    old_folds = read_csv(OLD_ROOT / "baseline_reproduction.csv")
    old_macro_ade = sum(float(row["ADE"]) for row in old_folds) / len(old_folds)
    old_macro_fde = sum(float(row["FDE"]) for row in old_folds) / len(old_folds)
    cv_macro = next(row for row in cv_rows if row["scene"] == "UNWEIGHTED_SCENE_MEAN")
    ade_ratio = macro_ade / 0.34
    fde_ratio = macro_fde / 0.59
    result = {
        "same_test_sample_ids_as_old_baseline": True,
        "old_unique_sample_count": len(old_ids),
        "repaired_unique_sample_count": len(new_ids),
        "all_repaired_metrics_finite": True,
        "fold_csv_recreated_from_per_sample_values": True,
        "sigma": 3,
        "seed": 42,
        "per_scene": per_scene,
        "macro_scene_mean": {
            "ADE": macro_ade,
            "FDE": macro_fde,
            "ADE_to_paper_ratio": ade_ratio,
            "FDE_to_paper_ratio": fde_ratio,
            "old_ADE": old_macro_ade,
            "old_FDE": old_macro_fde,
            "constant_velocity_ADE": float(cv_macro["constant_velocity_ADE"]),
            "constant_velocity_FDE": float(cv_macro["constant_velocity_FDE"]),
            "paper_ADE": 0.34,
            "paper_FDE": 0.59,
        },
        "gate": {
            "task_condition": "protocol confirmed, mean ADE/FDE near the paper magnitude, and no obvious per-scene anomaly",
            "protocol_and_sampling": "frame period and direct-mean metric align; context-pedestrian inclusion remains unspecified",
            "metrics_near_paper_magnitude": False,
            "scene_anomaly": "HOTEL and UNIV remain worse than the same-split constant-velocity baseline",
            "passed": False,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
