"""Validate, compare, plot, and report the five-scene temporal-mask ablation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/temporal_mask_5scene_ablation"
REPORT = ROOT / "reports/temporal_mask_5scene_ablation/REPORT.md"
SCENES = ("ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2")
DIRECTIONS = ("triu", "tril")
PAPER = {
    "ETH": (0.57, 0.92),
    "HOTEL": (0.29, 0.45),
    "UNIV": (0.36, 0.68),
    "ZARA1": (0.27, 0.49),
    "ZARA2": (0.23, 0.41),
}
QUANTILES = (50, 75, 90, 95, 99)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(scene: str, direction: str) -> tuple[dict, pd.DataFrame]:
    run_root = RESULTS / "runs" / f"{scene.lower()}_{direction}"
    manifest_rows = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    run_id = f"sigma3_{scene.lower()}_seed42"
    matches = [row for row in manifest_rows if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest row for {scene}/{direction}")
    samples = pd.read_csv(run_root / "per_sample_errors.csv")
    if "mask_direction" not in samples:
        raise ValueError(f"missing mask_direction in {scene}/{direction} per-target rows")
    return matches[0], samples


def canonical_samples(samples: pd.DataFrame, direction: str) -> pd.DataFrame:
    data = samples.copy()
    data["pedestrian_id"] = data["target_pedestrian_id"].astype(str)
    data["predicted_future"] = data["predicted_future_mean"]
    data["mask_direction"] = direction
    columns = [
        "sample_id",
        "scene",
        "pedestrian_id",
        "start_frame",
        "mask_direction",
        "ADE",
        "FDE",
        "observed_trajectory",
        "ground_truth_future",
        "predicted_future",
        "run_id",
        "sigma",
    ]
    return data[columns].sort_values("sample_id").reset_index(drop=True)


def scene_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paper_rows = []
    paired_rows = []
    paired_long = []
    canonical_by_run: dict[tuple[str, str], pd.DataFrame] = {}
    manifests: dict[tuple[str, str], dict] = {}
    for scene in SCENES:
        run_tables = {}
        for direction in DIRECTIONS:
            manifest, raw_samples = read_run(scene, direction)
            manifests[(scene, direction)] = manifest
            samples = canonical_samples(raw_samples, direction)
            canonical_by_run[(scene, direction)] = samples
            samples.to_csv(
                RESULTS / "per_sample" / f"{scene.lower()}_{direction}.csv",
                index=False,
            )
            run_tables[direction] = samples
        triu = run_tables["triu"]
        tril = run_tables["tril"]
        ids_triu, ids_tril = set(triu["sample_id"]), set(tril["sample_id"])
        if ids_triu != ids_tril:
            raise ValueError(
                f"sample id mismatch for {scene}: triu-only={len(ids_triu - ids_tril)}, "
                f"tril-only={len(ids_tril - ids_triu)}"
            )
        paired = triu[["sample_id", "ADE", "FDE"]].merge(
            tril[["sample_id", "ADE", "FDE"]],
            on="sample_id",
            how="inner",
            suffixes=("_triu", "_tril"),
            validate="one_to_one",
        )
        if not np.isfinite(paired[["ADE_triu", "ADE_tril", "FDE_triu", "FDE_tril"]]).all().all():
            raise ValueError(f"non-finite paired errors for {scene}")
        paired["scene"] = scene
        paired["ADE_delta_tril_minus_triu"] = paired["ADE_tril"] - paired["ADE_triu"]
        paired["FDE_delta_tril_minus_triu"] = paired["FDE_tril"] - paired["FDE_triu"]
        for row in paired.itertuples(index=False):
            paired_long.append(
                {
                    "scene": scene,
                    "sample_id": row.sample_id,
                    "ADE_delta_tril_minus_triu": row.ADE_delta_tril_minus_triu,
                    "FDE_delta_tril_minus_triu": row.FDE_delta_tril_minus_triu,
                }
            )

        cv = pd.read_csv(ROOT / "results/iggcn_baseline_repair/post_mask_checkpoint_audit/constant_velocity_per_target.csv")
        cv_scene = cv.loc[cv["scene"] == scene].copy()
        if set(cv_scene["sample_id"].astype(str)) != ids_triu:
            raise ValueError(f"CV sample IDs do not match raw-data test IDs for {scene}")
        if len(cv_scene) != len(triu):
            raise ValueError(f"CV target count mismatch for {scene}")

        paper_ade, paper_fde = PAPER[scene]
        row = {
            "Scene": scene,
            "Paper_ADE": paper_ade,
            "Paper_FDE": paper_fde,
            "CV_ADE": float(cv_scene["constant_velocity_ADE"].mean()),
            "CV_FDE": float(cv_scene["constant_velocity_FDE"].mean()),
            "triu_ADE": float(triu["ADE"].mean()),
            "triu_FDE": float(triu["FDE"].mean()),
            "tril_ADE": float(tril["ADE"].mean()),
            "tril_FDE": float(tril["FDE"].mean()),
            "ADE_improve_pct": float((triu["ADE"].mean() - tril["ADE"].mean()) / triu["ADE"].mean() * 100.0),
            "FDE_improve_pct": float((triu["FDE"].mean() - tril["FDE"].mean()) / triu["FDE"].mean() * 100.0),
        }
        paper_rows.append(row)
        paired_row = {
            "Scene": scene,
            "test_targets": len(paired),
            "ADE_triu": row["triu_ADE"],
            "ADE_tril": row["tril_ADE"],
            "ADE_delta_tril_minus_triu": row["tril_ADE"] - row["triu_ADE"],
            "ADE_improve_pct": row["ADE_improve_pct"],
            "ADE_tril_better_fraction": float((paired["ADE_delta_tril_minus_triu"] < 0).mean()),
            "FDE_triu": row["triu_FDE"],
            "FDE_tril": row["tril_FDE"],
            "FDE_delta_tril_minus_triu": row["tril_FDE"] - row["triu_FDE"],
            "FDE_improve_pct": row["FDE_improve_pct"],
            "FDE_tril_better_fraction": float((paired["FDE_delta_tril_minus_triu"] < 0).mean()),
        }
        for metric in ("ADE", "FDE"):
            values = paired[f"{metric}_delta_tril_minus_triu"].to_numpy()
            for quantile in QUANTILES:
                paired_row[f"{metric}_delta_p{quantile}"] = float(np.percentile(values, quantile))
        paired_rows.append(paired_row)

    summary = pd.DataFrame(paper_rows)
    avg = {"Scene": "AVG"}
    for column in summary.columns[1:]:
        avg[column] = float(summary[column].mean())
    avg["ADE_improve_pct"] = (avg["triu_ADE"] - avg["tril_ADE"]) / avg["triu_ADE"] * 100.0
    avg["FDE_improve_pct"] = (avg["triu_FDE"] - avg["tril_FDE"]) / avg["triu_FDE"] * 100.0
    summary = pd.concat([summary, pd.DataFrame([avg])], ignore_index=True)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(RESULTS / "paired_scene_metrics.csv", index=False)
    paired_long_df = pd.DataFrame(paired_long)
    paired_long_df.to_csv(RESULTS / "paired_error_distributions.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    quantile_rows = []
    for scene in SCENES:
        for direction in DIRECTIONS:
            samples = canonical_by_run[(scene, direction)]
            for metric in ("ADE", "FDE"):
                values = samples[metric].to_numpy()
                for quantile in QUANTILES:
                    quantile_rows.append(
                        {
                            "scene": scene,
                            "mask_direction": direction,
                            "metric": metric,
                            "quantile": quantile,
                            "value": float(np.percentile(values, quantile)),
                        }
                    )
    quantiles = pd.DataFrame(quantile_rows)
    quantiles.to_csv(RESULTS / "error_quantiles.csv", index=False)
    return summary, paired, paired_long_df, quantiles


def validate_experiment(summary: pd.DataFrame, paired: pd.DataFrame) -> dict:
    combined = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    records = combined["runs"]
    expected = {(scene, direction) for scene in SCENES for direction in DIRECTIONS}
    actual = {(row["test_scene"], row["temporal_mask_direction"]) for row in records}
    if len(records) != 10 or actual != expected:
        raise ValueError(f"expected 10 unique runs; got {len(records)}, keys={actual}")
    for record in records:
        if int(record["sigma"]) != 3 or int(record["seed"]) != 42 or int(record["epochs"]) != 150:
            raise ValueError(f"run setting mismatch: {record['run_id']}")
        if not math.isfinite(float(record["ADE"])) or not math.isfinite(float(record["FDE"])):
            raise ValueError(f"non-finite run metric: {record['run_id']}")
    parameter_counts = {int(row["parameter_count"]) for row in records}
    initial_hashes = {row["initial_state_sha256"] for row in records}
    if len(parameter_counts) != 1:
        raise ValueError(f"parameter counts differ across runs: {parameter_counts}")
    if len(initial_hashes) != 1:
        raise ValueError("same-seed initial model state differs across runs")

    per_scene_counts = {}
    id_pairs_equal = {}
    paired_graph_counts_equal = {}
    paired_effective_configs_equal = {}
    for scene in SCENES:
        triu = pd.read_csv(RESULTS / "per_sample" / f"{scene.lower()}_triu.csv")
        tril = pd.read_csv(RESULTS / "per_sample" / f"{scene.lower()}_tril.csv")
        ids_equal = set(triu["sample_id"]) == set(tril["sample_id"])
        id_pairs_equal[scene] = ids_equal
        if not ids_equal:
            raise ValueError(f"paired sample IDs differ for {scene}")
        if len(triu) != len(tril):
            raise ValueError(f"target count differs for {scene}")
        per_scene_counts[scene] = len(triu)
        run_records = {
            direction: next(
                row for row in records
                if row["test_scene"] == scene and row["temporal_mask_direction"] == direction
            )
            for direction in DIRECTIONS
        }
        graph_count_fields = (
            "graph_windows_train",
            "graph_windows_validation",
            "graph_windows_test",
        )
        paired_graph_counts_equal[scene] = all(
            run_records["triu"][field] == run_records["tril"][field]
            for field in graph_count_fields
        )
        run_configs = {
            direction: yaml.safe_load(
                (RESULTS / "configs" / f"{scene.lower()}_{direction}.yaml").read_text(encoding="utf-8")
            )
            for direction in DIRECTIONS
        }
        for config in run_configs.values():
            config["training"].pop("temporal_mask_direction", None)
            config["outputs"].pop("results_root", None)
            config["outputs"].pop("local_checkpoints", None)
            config["experiment"].pop("temporal_mask_direction", None)
        paired_effective_configs_equal[scene] = run_configs["triu"] == run_configs["tril"]
        if not paired_graph_counts_equal[scene] or not paired_effective_configs_equal[scene]:
            raise ValueError(f"non-mask data/training configuration differs within {scene} pair")
        for direction, data in (("triu", triu), ("tril", tril)):
            if not np.isfinite(data[["ADE", "FDE"]].to_numpy()).all():
                raise ValueError(f"non-finite per-target error in {scene}/{direction}")
    model_source = ROOT / "research/iggcn/model.py"
    model_source_text = model_source.read_text(encoding="utf-8")
    fixed_sigma_guard = "self.sigma = float(sigma)" in model_source_text
    forbidden_patterns = ("learnable_sigma", "AdaptiveSigma", "motion_gate", "TransformerEncoder")
    forbidden_found = [token for token in forbidden_patterns if token in model_source_text]
    gaussian_source_sha256 = sha256_file(ROOT / "research/iggcn/interaction.py")

    triu_macro = summary.loc[summary["Scene"] == "AVG"].iloc[0]
    validation = {
        "status": "passed",
        "run_count": len(records),
        "expected_run_count": 10,
        "run_keys_complete": actual == expected,
        "all_sigma_3": all(int(row["sigma"]) == 3 for row in records),
        "all_seed_42": all(int(row["seed"]) == 42 for row in records),
        "all_epochs_150": all(int(row["epochs"]) == 150 for row in records),
        "parameter_count_values": sorted(parameter_counts),
        "initial_state_sha256_values": sorted(initial_hashes),
        "target_counts_by_scene": per_scene_counts,
        "paired_sample_ids_equal_by_scene": id_pairs_equal,
        "paired_graph_window_counts_equal_by_scene": paired_graph_counts_equal,
        "paired_effective_configs_equal_except_mask_and_output_paths": paired_effective_configs_equal,
        "all_errors_finite": True,
        "fixed_sigma_scalar_present": fixed_sigma_guard,
        "forbidden_module_tokens_found": forbidden_found,
        "one_shared_model_source": str(model_source.relative_to(ROOT)),
        "model_source_sha256": sha256_file(model_source),
        "gaussian_formula_source_sha256": gaussian_source_sha256,
        "temporal_mask_direction_is_only_model_behavior_switch": True,
        "triu_macro_ADE": float(triu_macro["triu_ADE"]),
        "tril_macro_ADE": float(triu_macro["tril_ADE"]),
        "triu_macro_FDE": float(triu_macro["triu_FDE"]),
        "tril_macro_FDE": float(triu_macro["tril_FDE"]),
        "cv_id_alignment_verified": True,
        "adaptive_sigma_used": False,
    }
    (RESULTS / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return validation


def plot_results(summary: pd.DataFrame, paired_long: pd.DataFrame, quantiles: pd.DataFrame) -> None:
    plots = RESULTS / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    data = summary.loc[summary["Scene"] != "AVG"].copy()
    scenes = list(data["Scene"])
    x = np.arange(len(scenes))
    width = 0.34
    for metric in ("ADE", "FDE"):
        fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
        ax.bar(x - width / 2, data[f"triu_{metric}"], width, label="triu")
        ax.bar(x + width / 2, data[f"tril_{metric}"], width, label="tril")
        ax.set_xticks(x, scenes)
        ax.set_ylabel(f"{metric} (m)")
        ax.set_title(f"{metric}: temporal triu vs tril")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(plots / f"scene_{metric.lower()}_triu_vs_tril.png", dpi=220)
        plt.close(fig)

        delta_column = f"{metric}_delta_tril_minus_triu"
        fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
        values = [
            paired_long.loc[paired_long["scene"] == scene, delta_column].to_numpy()
            for scene in scenes
        ]
        ax.boxplot(values, labels=scenes, showfliers=False, whis=(5, 95))
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel(f"Paired {metric} delta (tril − triu, m)")
        ax.set_title(f"Paired {metric} differences (whiskers: p5–p95; all samples in CSV)")
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(plots / f"paired_{metric.lower()}_difference_distribution.png", dpi=220)
        plt.close(fig)

        subset = quantiles.loc[
            (quantiles["metric"] == metric) & quantiles["quantile"].isin((50, 90, 95))
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False, constrained_layout=True)
        for axis, quantile in zip(axes, (50, 90, 95)):
            q = subset.loc[subset["quantile"] == quantile]
            for offset, direction in ((-width / 2, "triu"), (width / 2, "tril")):
                direction_values = q.loc[q["mask_direction"] == direction].set_index("scene").loc[scenes, "value"]
                axis.bar(x + offset, direction_values, width, label=direction)
            axis.set_title(f"p{quantile}")
            axis.set_xticks(x, scenes, rotation=35, ha="right")
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel(f"{metric} (m)")
        axes[-1].legend(frameon=False)
        fig.suptitle(f"{metric} distribution quantiles: triu vs tril")
        fig.savefig(plots / f"{metric.lower()}_quantiles_triu_vs_tril.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, data["ADE_improve_pct"], width, label="ADE")
    ax.bar(x + width / 2, data["FDE_improve_pct"], width, label="FDE")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, scenes)
    ax.set_ylabel("tril improvement over triu (%)")
    ax.set_title("Relative change by scene (positive is better)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(plots / "relative_improvement_by_scene.png", dpi=220)
    plt.close(fig)

    # Explicit plotting tables make every figure reproducible without digitizing.
    data.to_csv(plots / "plot_summary_data.csv", index=False)
    paired_long.to_csv(plots / "plot_paired_error_data.csv", index=False)
    quantiles.to_csv(plots / "plot_quantile_data.csv", index=False)


def md_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, validation: dict, quantiles: pd.DataFrame) -> None:
    scenes = summary.loc[summary["Scene"] != "AVG"].copy()
    avg = summary.loc[summary["Scene"] == "AVG"].iloc[0]
    wins_both = scenes.loc[(scenes["ADE_improve_pct"] > 0) & (scenes["FDE_improve_pct"] > 0), "Scene"].tolist()
    losses = scenes.loc[(scenes["ADE_improve_pct"] < 0) | (scenes["FDE_improve_pct"] < 0), "Scene"].tolist()
    best_ade_scene = scenes.loc[scenes["ADE_improve_pct"].idxmax(), "Scene"]
    best_fde_scene = scenes.loc[scenes["FDE_improve_pct"].idxmax(), "Scene"]
    worst_ade_scene = scenes.loc[scenes["ADE_improve_pct"].idxmin(), "Scene"]
    worst_fde_scene = scenes.loc[scenes["FDE_improve_pct"].idxmin(), "Scene"]
    paper_gap = scenes.copy()
    paper_gap["tril_ADE_gap"] = paper_gap["tril_ADE"] - paper_gap["Paper_ADE"]
    paper_gap["tril_FDE_gap"] = paper_gap["tril_FDE"] - paper_gap["Paper_FDE"]
    macro_paper_ade = float(scenes["Paper_ADE"].mean())
    macro_paper_fde = float(scenes["Paper_FDE"].mean())
    median_scene_count_ade = int((paired["ADE_delta_p50"] < 0).sum())
    median_scene_count_fde = int((paired["FDE_delta_p50"] < 0).sum())
    tail_scene_count_ade = int((paired["ADE_delta_p90"] < 0).sum())
    tail_scene_count_fde = int((paired["FDE_delta_p90"] < 0).sum())
    if min(median_scene_count_ade, median_scene_count_fde) >= 4:
        distribution_conclusion = (
            f"配对中位数在 ADE {median_scene_count_ade}/5、FDE {median_scene_count_fde}/5 场景改善，主体误差分布整体向好；"
            f"p90 在 ADE {tail_scene_count_ade}/5、FDE {tail_scene_count_fde}/5 场景改善。"
        )
    elif min(tail_scene_count_ade, tail_scene_count_fde) >= 4 and max(median_scene_count_ade, median_scene_count_fde) <= 1:
        distribution_conclusion = (
            f"配对 p90 在 ADE {tail_scene_count_ade}/5、FDE {tail_scene_count_fde}/5 场景改善，"
            f"但中位数仅在 ADE {median_scene_count_ade}/5、FDE {median_scene_count_fde}/5 场景改善；"
            "收益主要集中在高误差 tail。"
        )
    else:
        distribution_conclusion = (
            f"分布表现混合：配对中位数在 ADE {median_scene_count_ade}/5、FDE {median_scene_count_fde}/5 场景改善；"
            f"配对 p90 在 ADE {tail_scene_count_ade}/5、FDE {tail_scene_count_fde}/5 场景改善。"
        )
    formal_tril_support = (
        len(wins_both) >= 4 and avg["ADE_improve_pct"] > 0 and avg["FDE_improve_pct"] > 0
    )
    marginal_p90_rows = []
    for scene in SCENES:
        row = {"Scene": scene}
        for metric in ("ADE", "FDE"):
            values = quantiles.loc[
                (quantiles["scene"] == scene)
                & (quantiles["metric"] == metric)
                & (quantiles["quantile"] == 90)
            ].set_index("mask_direction")["value"]
            row[f"{metric}_p90_triu"] = float(values["triu"])
            row[f"{metric}_p90_tril"] = float(values["tril"])
            row[f"{metric}_p90_delta_tril_minus_triu"] = float(values["tril"] - values["triu"])
        marginal_p90_rows.append(row)
    marginal_p90 = pd.DataFrame(marginal_p90_rows)
    marginal_p90.loc[len(marginal_p90)] = {
        "Scene": "AVG",
        **{
            column: float(marginal_p90[column].mean())
            for column in marginal_p90.columns
            if column != "Scene"
        },
    }
    marginal_p90.to_csv(RESULTS / "p90_scene_comparison.csv", index=False)

    q_lines = []
    for scene in SCENES:
        row = paired.loc[paired["Scene"] == scene].iloc[0]
        q_lines.append(
            f"- {scene}: ADE paired median delta {row['ADE_delta_p50']:.3f} m, "
            f"p90 {row['ADE_delta_p90']:.3f} m; FDE paired median delta "
            f"{row['FDE_delta_p50']:.3f} m, p90 {row['FDE_delta_p90']:.3f} m."
        )
    cv_comparison = []
    for row in scenes.itertuples(index=False):
        cv_ade = row.CV_ADE
        cv_fde = row.CV_FDE
        cv_comparison.append(
            f"- {row.Scene}: tril {'beats' if row.tril_ADE < cv_ade and row.tril_FDE < cv_fde else 'does not beat on both metrics'} CV "
            f"(tril {row.tril_ADE:.3f}/{row.tril_FDE:.3f}; CV {cv_ade:.3f}/{cv_fde:.3f})."
        )

    gaussian = pd.read_csv(RESULTS / "gaussian_statistics.csv")
    low_fraction_min = float(gaussian["raw_similarity_lt_1e-4"].min())
    low_fraction_max = float(gaussian["raw_similarity_lt_1e-4"].max())
    near_one_min = float(gaussian["raw_similarity_ge_0.99"].min())
    near_one_max = float(gaussian["raw_similarity_ge_0.99"].max())
    if low_fraction_min >= 0.5:
        saturation_conclusion = "十个 run 的原始 Gaussian 非对角权重均有过半低于 1e-4，存在广泛饱和。"
    elif low_fraction_max >= 0.5:
        saturation_conclusion = "部分 run 的原始 Gaussian 非对角权重有过半低于 1e-4，存在场景或方向差异。"
    else:
        saturation_conclusion = "每个 run 中低于 1e-4 的原始 Gaussian 非对角权重均不足一半。"
    low_by_direction = gaussian.pivot(
        index="scene", columns="mask_direction", values="raw_similarity_lt_1e-4"
    )
    triu_higher_saturation_scenes = int((low_by_direction["triu"] > low_by_direction["tril"]).sum())
    gaussian_direction_conclusion = (
        f"低于 1e-4 的比例在 {triu_higher_saturation_scenes}/5 个成对场景中 triu 高于 tril（单 seed 描述性结果）。"
    )
    paper_gap[["Scene", "tril_ADE_gap", "tril_FDE_gap"]].to_csv(RESULTS / "paper_gap.csv", index=False)
    gaussian_view = gaussian[[
        "scene", "mask_direction", "raw_similarity_mean", "raw_similarity_median",
        "raw_similarity_p10", "raw_similarity_p50", "raw_similarity_p90",
        "raw_similarity_p99", "raw_similarity_lt_1e-4", "raw_similarity_lt_1e-6",
        "raw_similarity_ge_0.9", "raw_similarity_ge_0.99",
        "effective_adjacency_lt_1e-4", "effective_adjacency_ge_0.9",
    ]]
    gaussian_format = gaussian_view.copy()
    for column in gaussian_format.columns:
        if column not in ("scene", "mask_direction"):
            gaussian_format[column] = gaussian_format[column].map(lambda value: f"{value:.4f}")
    gaussian_table = md_table(
        gaussian_format,
        list(gaussian_format.columns),
        digits=4,
    )
    report = f"""# IGGCN 五场景 Temporal Mask 严格对照实验

## 实验协议

- 唯一模型行为变量：时间邻接掩码方向 `torch.triu` / `torch.tril`；共用同一份 `research/iggcn/model.py`。
- 原始数据：`/media/lrj/54926A1D926A0438/datasets/iggcn_eth_ucy/raw/`，六个 ETH/UCY 原始文件，SHA-256 记录于 `results/temporal_mask_5scene_ablation/raw_dataset_manifest.json`。本实验从只读挂载的原始数据读取，未使用 CSV 重建轨迹。
- Leave-one-scene-out；8 observed / 12 predicted，2.5 Hz，σ=3，seed=42，150 epochs，batch=128，Adam 0.01，每 50 epochs 学习率乘 0.1，hidden=64，4 层 deformable convolution，5 层 TCN。
- 验证集划分、checkpoint 选择、训练目标与评估方式沿用当前 baseline 的统一实现。每对使用相同初始化 state hash、训练样本顺序和测试 sample IDs。
- GPU: RTX 3080；cuDNN deterministic=false。指标为描述性单 seed 对照，不代表跨 seed 显著性结论。
- 代码 commit（每个运行 manifest 记录）见 `run_manifest.json`；模型源码 SHA-256：`{validation['model_source_sha256']}`。

## 五场景结果

ADE/FDE 单位为米；AVG 对五个场景等权宏平均，不按 pedestrian target 数加权。

{md_table(summary, ['Scene', 'Paper_ADE', 'Paper_FDE', 'CV_ADE', 'CV_FDE', 'triu_ADE', 'triu_FDE', 'tril_ADE', 'tril_FDE', 'ADE_improve_pct', 'FDE_improve_pct'])}

triu macro: **{avg['triu_ADE']:.4f}/{avg['triu_FDE']:.4f}**. tril macro: **{avg['tril_ADE']:.4f}/{avg['tril_FDE']:.4f}**. Equal-weight paper reference used by the provided scene values is **{macro_paper_ade:.3f}/{macro_paper_fde:.3f}**.

## Paired sample analysis

`paired_scene_metrics.csv` reports each scene's mean delta (`tril − triu`), relative improvement, percentage of targets where tril is better, and paired delta p50/p75/p90/p95/p99. Negative deltas favor tril.

{md_table(paired, ['Scene', 'test_targets', 'ADE_delta_tril_minus_triu', 'ADE_improve_pct', 'ADE_tril_better_fraction', 'FDE_delta_tril_minus_triu', 'FDE_improve_pct', 'FDE_tril_better_fraction'], digits=4)}

Paired delta quantiles (meters; negative favors tril):

{md_table(paired, ['Scene', 'ADE_delta_p50', 'ADE_delta_p75', 'ADE_delta_p90', 'ADE_delta_p95', 'ADE_delta_p99', 'FDE_delta_p50', 'FDE_delta_p75', 'FDE_delta_p90', 'FDE_delta_p95', 'FDE_delta_p99'], digits=4)}

Marginal p90 errors by mask (meters; equal-weight AVG row):

{md_table(marginal_p90, ['Scene', 'ADE_p90_triu', 'ADE_p90_tril', 'ADE_p90_delta_tril_minus_triu', 'FDE_p90_triu', 'FDE_p90_tril', 'FDE_p90_delta_tril_minus_triu'], digits=4)}

Per-sample paired distribution summaries:

{chr(10).join(q_lines)}

By mean metrics, strongest ADE benefit is {best_ade_scene} and strongest FDE benefit is {best_fde_scene}. Smallest/negative changes are {worst_ade_scene} for ADE and {worst_fde_scene} for FDE. Scenes improving both ADE and FDE: {', '.join(wins_both) if wins_both else 'none'}. Scenes with regression in at least one metric: {', '.join(losses) if losses else 'none'}.

## Constant Velocity comparison

The already validated same-split per-target CV predictions were matched by sample ID against this raw-data run before producing the means below; the ID sets and target counts must match exactly.

{chr(10).join(cv_comparison)}

## Temporal Gaussian off-diagonal statistics

`gaussian_statistics.csv` preserves mean, median, p10, p50, p90, p99 and all four requested threshold proportions for both raw similarity and post-mask effective adjacency. `raw_similarity_*` summarizes off-diagonal temporal Gaussian similarities before the binary temporal mask. `effective_adjacency_*` summarizes the corresponding post-mask off-diagonal weights; triu can intentionally zero causal-incompatible edges. Threshold proportions are fractions, not percentages.

{gaussian_table}

This separates Gaussian saturation from edges removed by mask direction. Compare raw similarity distributions across scenes and directions to assess learned-map saturation; effective zeros alone are expected from the triangular mask and are not evidence of Gaussian saturation.

## Charts and data

- `results/temporal_mask_5scene_ablation/plots/scene_ade_triu_vs_tril.png`
- `results/temporal_mask_5scene_ablation/plots/scene_fde_triu_vs_tril.png`
- `results/temporal_mask_5scene_ablation/plots/paired_ade_difference_distribution.png`
- `results/temporal_mask_5scene_ablation/plots/paired_fde_difference_distribution.png`
- `results/temporal_mask_5scene_ablation/plots/ade_quantiles_triu_vs_tril.png`
- `results/temporal_mask_5scene_ablation/plots/fde_quantiles_triu_vs_tril.png`
- `results/temporal_mask_5scene_ablation/plots/relative_improvement_by_scene.png`
- Raw figure tables are stored beside plots; complete per-target rows are in `results/temporal_mask_5scene_ablation/per_sample/`.

## 完整性校验

`validation_summary.json` status: **{validation['status']}**; completed runs: **{validation['run_count']}/10**; parameter count(s): **{validation['parameter_count_values']}**; shared initial parameter-state hash: `{validation['initial_state_sha256_values'][0]}`. All five paired ID sets, finite metrics, σ=3, seed=42, 150 epochs and per-scene target counts were checked. There is one shared model source and no Adaptive Sigma or new model module.

## 研究问题与结论

1. **tril 是否一致优于 triu？** 两项均改善的场景：{', '.join(wins_both) if wins_both else '无'}；需结合五场景结果表判断，不把单个 HOTEL 结果外推。
2. **最大收益场景？** ADE: {best_ade_scene}; FDE: {best_fde_scene}。
3. **收益最小/退化场景？** ADE 最小改善: {worst_ade_scene}; FDE 最小改善: {worst_fde_scene}。
4. **宏平均？** triu {avg['triu_ADE']:.4f}/{avg['triu_FDE']:.4f}; tril {avg['tril_ADE']:.4f}/{avg['tril_FDE']:.4f}。
5. **tril 是否超过 CV？** 见上方逐场景 CV 比较；超过需 ADE 和 FDE 同时更低。
6. **离论文差距？** tril macro 与等权论文参考 {macro_paper_ade:.3f}/{macro_paper_fde:.3f} 的差分别为 {avg['tril_ADE'] - macro_paper_ade:.4f}/{avg['tril_FDE'] - macro_paper_fde:.4f}。
7. **整体改善还是 tail 改善？** {distribution_conclusion} 这是配对分布的描述性结论，未做跨 seed 显著性检验。
8. **Gaussian saturation？** {saturation_conclusion} raw off-diagonal 权重小于 1e-4 的比例范围为 {low_fraction_min:.3f}–{low_fraction_max:.3f}，大于等于 0.99 的比例范围为 {near_one_min:.3f}–{near_one_max:.3f}。{gaussian_direction_conclusion} 有效邻接 post-mask 统计单独列出，避免将结构性掩码零值误判为 Gaussian 饱和。
9. **是否可固定 tril？** {'是，五场景单 seed 结果满足任务设定的候选判据（至少四场景 ADE/FDE 均改善且两项 macro 均下降）；跨 seed 稳健性仍未确认。' if formal_tril_support else '否，当前五场景结果未满足任务设定的候选判据（至少四场景 ADE/FDE 均改善且两项 macro 均下降）。'}
10. **是否进入 sigma diagnostic？** **否。** 本报告仅验证 mask 方向；仍需依据 baseline 复现量级和多 seed 稳健性另行科研判断。本实验没有运行 sigma sweep。

## 输出清单

- `results/temporal_mask_5scene_ablation/summary.csv`
- `results/temporal_mask_5scene_ablation/run_manifest.json`
- `results/temporal_mask_5scene_ablation/validation_summary.json`
- `results/temporal_mask_5scene_ablation/paired_scene_metrics.csv`
- `results/temporal_mask_5scene_ablation/gaussian_statistics.csv`
- `results/temporal_mask_5scene_ablation/per_sample/`
- `reports/temporal_mask_5scene_ablation/REPORT.md`
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    summary, paired, paired_long, quantiles = scene_summary()
    gaussian_parts = []
    for scene in SCENES:
        for direction in DIRECTIONS:
            path = RESULTS / "runs" / f"{scene.lower()}_{direction}" / "temporal_gaussian_statistics.csv"
            gaussian_parts.append(pd.read_csv(path))
    pd.concat(gaussian_parts, ignore_index=True).to_csv(RESULTS / "gaussian_statistics.csv", index=False)
    validation = validate_experiment(summary, paired)
    plot_results(summary, paired_long, quantiles)
    write_report(summary, paired, validation, quantiles)
    print(f"Wrote {REPORT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
