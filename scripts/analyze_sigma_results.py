"""Build fixed-sigma, state-group, oracle, and case analyses from fold CSVs."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENES = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
SIGMAS = [1, 2, 3, 4, 5]
STATE_FEATURES = ["v_mean", "density", "turning", "a_mean"]
STATE_TITLES = {
    "v_mean": "Mean observed speed",
    "density": "Observed local density",
    "turning": "Observed cumulative turning",
    "a_mean": "Mean observed acceleration",
}
PAPER_RESULTS = {
    "ETH": (0.57, 0.92),
    "HOTEL": (0.29, 0.45),
    "UNIV": (0.36, 0.68),
    "ZARA1": (0.27, 0.49),
    "ZARA2": (0.23, 0.41),
    "AVG": (0.34, 0.59),
}


def quantile_group_by_scene(frame: pd.DataFrame, feature: str) -> pd.Series:
    groups = pd.Series(index=frame.index, dtype="object")
    for _, indices in frame.groupby("scene").groups.items():
        percentile = frame.loc[indices, feature].rank(method="average", pct=True)
        groups.loc[indices] = np.select(
            [percentile <= 1 / 3, percentile <= 2 / 3],
            ["low", "medium"],
            default="high",
        )
    return groups


def parse_trajectory(value: str) -> np.ndarray:
    return np.asarray(
        [[float(coord) for coord in point.split(":")] for point in value.split(";")],
        dtype=float,
    )


def choose_typical_cases(oracle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = {
        "sigma1_better": (1, [3, 5]),
        "sigma3_better": (3, [1, 5]),
        "sigma5_better": (5, [1, 3]),
    }
    for label, (candidate_sigma, comparison_sigmas) in comparisons.items():
        candidate_error = oracle[f"ADE_sigma{candidate_sigma}"]
        comparison_error = oracle[
            [f"ADE_sigma{sigma}" for sigma in comparison_sigmas]
        ].min(axis=1)
        candidates = oracle.loc[candidate_error < comparison_error].copy()
        if candidates.empty:
            continue
        candidates["case_margin"] = comparison_error.loc[candidates.index] - candidate_error.loc[
            candidates.index
        ]
        best = candidates.sort_values("case_margin", ascending=False).iloc[0].copy()
        best["case_type"] = label
        rows.append(best)
    return pd.DataFrame(rows)


def draw_case(row: pd.Series, output_path: Path) -> None:
    observed = parse_trajectory(row["observed_trajectory"])
    ground_truth = parse_trajectory(row["ground_truth_future"])
    predictions = {
        sigma: parse_trajectory(row[f"prediction_sigma{sigma}"])
        for sigma in SIGMAS
    }
    last_point = observed[-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    ax.plot(observed[:, 0], observed[:, 1], color="#bd2d28", marker="o", label="Observed")
    ax.plot(
        np.r_[last_point[0], ground_truth[:, 0]],
        np.r_[last_point[1], ground_truth[:, 1]],
        color="#1f4e79",
        marker="o",
        label="Ground truth",
    )
    colors = {1: "#e69f00", 2: "#56b4e9", 3: "#009e73", 4: "#cc79a7", 5: "#7b61a8"}
    for sigma in (1, 3, 5):
        prediction = predictions[sigma]
        ax.plot(
            np.r_[last_point[0], prediction[:, 0]],
            np.r_[last_point[1], prediction[:, 1]],
            color=colors[sigma],
            linestyle="--",
            marker=".",
            label=f"sigma={sigma} (ADE {row[f'ADE_sigma{sigma}']:.3f})",
        )
    ax.set_title(
        f"{row['case_type']}: {row['scene']} / {row['target_pedestrian_id']} / frame {row['start_frame']}"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results/iggcn_sigma_diagnostic",
    )
    args = parser.parse_args()
    root = args.results_root
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fold_metrics = pd.read_csv(root / "sigma_sweep_seed42.csv")
    expected = {(sigma, scene) for sigma in SIGMAS for scene in SCENES}
    actual = set(zip(fold_metrics["sigma"].astype(int), fold_metrics["test_scene"]))
    if actual != expected or len(fold_metrics) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"need exactly 25 unique sigma/fold rows; missing={missing}, extra={extra}")

    scene_summary = fold_metrics.pivot(
        index="sigma", columns="test_scene", values=["ADE", "FDE"]
    )
    scene_summary.columns = [f"{metric}_{scene}" for metric, scene in scene_summary.columns]
    scene_summary = scene_summary.reset_index()
    for metric in ("ADE", "FDE"):
        scene_columns = [f"{metric}_{scene}" for scene in SCENES]
        scene_summary[f"AVG_{metric}"] = scene_summary[scene_columns].mean(axis=1)
    scene_summary.to_csv(root / "sigma_scene_summary.csv", index=False)

    overall = scene_summary[["sigma", "AVG_ADE", "AVG_FDE"]].copy()
    overall.to_csv(root / "sigma_overall_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    ax.plot(overall["sigma"], overall["AVG_ADE"], marker="o", label="AVG ADE")
    ax.set_xticks(SIGMAS)
    ax.set_xlabel("Fixed Gaussian sigma")
    ax.set_ylabel("Mean ADE (m)")
    ax.grid(alpha=0.25)
    fig.savefig(plots / "sigma_vs_avg_ade.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    ax.plot(overall["sigma"], overall["AVG_FDE"], marker="o", color="#d55e00", label="AVG FDE")
    ax.set_xticks(SIGMAS)
    ax.set_xlabel("Fixed Gaussian sigma")
    ax.set_ylabel("Mean FDE (m)")
    ax.grid(alpha=0.25)
    fig.savefig(plots / "sigma_vs_avg_fde.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    heat = scene_summary.set_index("sigma")[[f"ADE_{scene}" for scene in SCENES]].T
    heat.index = SCENES
    heat.to_csv(root / "plot_data_scene_sigma_ade.csv")
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "ADE (m)"})
    ax.set_xlabel("Fixed Gaussian sigma")
    ax.set_ylabel("Test scene")
    fig.savefig(plots / "scene_sigma_ade_heatmap.png", dpi=220)
    plt.close(fig)

    raw = pd.read_csv(root / "per_sample_errors.csv")
    if set(raw["sigma"].astype(int).unique()) != set(SIGMAS):
        raise ValueError("per-sample file must contain sigma 1 through 5")
    identity = [
        "sample_id", "scene", "component", "target_pedestrian_id", "start_frame",
        *STATE_FEATURES, "v_last", "observed_trajectory", "ground_truth_future",
    ]
    prediction_wide = raw.pivot(index=identity, columns="sigma", values="predicted_future_mean")
    prediction_wide.columns = [f"prediction_sigma{int(sigma)}" for sigma in prediction_wide.columns]
    error_wide = raw.pivot(index=identity, columns="sigma", values=["ADE", "FDE"])
    error_wide.columns = [f"{metric}_sigma{int(sigma)}" for metric, sigma in error_wide.columns]
    oracle = error_wide.join(prediction_wide).reset_index()
    if oracle[[f"ADE_sigma{s}" for s in SIGMAS]].isna().any().any():
        raise ValueError("sigma runs do not contain aligned per-target sample ids")
    ade_columns = [f"ADE_sigma{s}" for s in SIGMAS]
    fde_columns = [f"FDE_sigma{s}" for s in SIGMAS]
    oracle["best_sigma"] = np.asarray(SIGMAS)[
        oracle[ade_columns].to_numpy().argmin(axis=1)
    ]
    oracle["best_ADE"] = oracle[ade_columns].min(axis=1)
    oracle["FDE_at_best_ADE_sigma"] = [
        oracle.iloc[index][f"FDE_sigma{int(sigma)}"]
        for index, sigma in enumerate(oracle["best_sigma"])
    ]
    oracle["best_FDE_independent"] = oracle[fde_columns].min(axis=1)
    oracle["ADE_gain_vs_sigma3"] = oracle["ADE_sigma3"] - oracle["best_ADE"]
    oracle["FDE_gain_at_ADE_best_sigma_vs_sigma3"] = (
        oracle["FDE_sigma3"] - oracle["FDE_at_best_ADE_sigma"]
    )
    oracle.to_csv(root / "oracle_sigma_analysis.csv", index=False)

    grouped_parts = []
    distribution_parts = []
    state_input = oracle.copy()
    for feature in STATE_FEATURES:
        group_column = f"{feature}_group"
        state_input[group_column] = quantile_group_by_scene(state_input, feature)
        feature_runs = raw.merge(
            state_input[["sample_id", "scene", group_column]],
            on=["sample_id", "scene"],
            validate="many_to_one",
        )
        for (scene, group, sigma), subset in feature_runs.groupby(
            ["scene", group_column, "sigma"], observed=True
        ):
            grouped_parts.append(
                {
                    "state_feature": feature,
                    "test_scene": scene,
                    "state_group": group,
                    "sigma": int(sigma),
                    "n_samples": len(subset),
                    "ADE": subset["ADE"].mean(),
                    "FDE": subset["FDE"].mean(),
                }
            )
        for (scene, group, sigma), subset in state_input.groupby(
            ["scene", group_column, "best_sigma"], observed=True
        ):
            distribution_parts.append(
                {
                    "state_feature": feature,
                    "test_scene": scene,
                    "state_group": group,
                    "best_sigma": int(sigma),
                    "n_samples": len(subset),
                }
            )
    state_metrics = pd.DataFrame(grouped_parts)
    state_metrics["n_in_group"] = state_metrics.groupby(
        ["state_feature", "test_scene", "state_group"]
    )["n_samples"].transform("max")
    state_metrics.to_csv(root / "state_group_metrics.csv", index=False)
    state_summary = (
        state_metrics.assign(weighted_ADE=state_metrics["ADE"] * state_metrics["n_samples"],
                             weighted_FDE=state_metrics["FDE"] * state_metrics["n_samples"])
        .groupby(["state_feature", "state_group", "sigma"], as_index=False)
        .agg(n_samples=("n_samples", "sum"), weighted_ADE=("weighted_ADE", "sum"),
             weighted_FDE=("weighted_FDE", "sum"))
    )
    state_summary["ADE"] = state_summary["weighted_ADE"] / state_summary["n_samples"]
    state_summary["FDE"] = state_summary["weighted_FDE"] / state_summary["n_samples"]
    state_summary["n_in_group"] = state_summary.groupby(
        ["state_feature", "state_group"]
    )["n_samples"].transform("max")
    state_summary.to_csv(root / "state_group_summary.csv", index=False)
    distribution = pd.DataFrame(distribution_parts)
    distribution["n_in_group"] = distribution.groupby(
        ["state_feature", "test_scene", "state_group"]
    )["n_samples"].transform("sum")
    distribution["proportion"] = distribution["n_samples"] / distribution["n_in_group"]
    distribution.to_csv(root / "oracle_best_sigma_distribution.csv", index=False)
    distribution_summary = (
        distribution.groupby(["state_feature", "state_group", "best_sigma"], as_index=False)
        ["n_samples"].sum()
    )
    distribution_summary["n_in_group"] = distribution_summary.groupby(
        ["state_feature", "state_group"]
    )["n_samples"].transform("sum")
    distribution_summary["proportion"] = (
        distribution_summary["n_samples"] / distribution_summary["n_in_group"]
    )
    distribution_summary.to_csv(root / "oracle_best_sigma_distribution_summary.csv", index=False)

    for feature in ("v_mean", "density", "turning"):
        table = state_summary.loc[state_summary["state_feature"] == feature].pivot(
            index="state_group", columns="sigma", values="ADE"
        ).reindex(["low", "medium", "high"])
        table.to_csv(root / f"plot_data_{feature}_group_sigma_ade.csv")
        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        sns.heatmap(table, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "ADE (m)"})
        ax.set_title(f"{STATE_TITLES[feature]} × sigma")
        ax.set_xlabel("Fixed Gaussian sigma")
        ax.set_ylabel("Within-scene quantile group")
        fig.savefig(plots / f"{feature}_group_sigma_ade_heatmap.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    for axis, feature in zip(axes.flat, STATE_FEATURES):
        subset = distribution_summary.loc[distribution_summary["state_feature"] == feature]
        pivot = subset.pivot(index="state_group", columns="best_sigma", values="proportion")
        pivot = pivot.reindex(index=["low", "medium", "high"], columns=SIGMAS).fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=axis, colormap="viridis", width=0.8)
        axis.set_title(STATE_TITLES[feature])
        axis.set_xlabel("Within-scene quantile group")
        axis.set_ylabel("Oracle best-sigma fraction")
        axis.set_ylim(0, 1)
        axis.legend(title="Best sigma", fontsize=7, title_fontsize=8, frameon=False)
    fig.savefig(plots / "oracle_best_sigma_state_distribution.png", dpi=220)
    plt.close(fig)

    cases = choose_typical_cases(oracle)
    cases.to_csv(root / "typical_case_samples.csv", index=False)
    for _, row in cases.iterrows():
        draw_case(row, plots / f"case_{row['case_type']}.png")

    comparison_rows = []
    baseline = scene_summary.loc[scene_summary["sigma"] == 3].iloc[0]
    for scene in [*SCENES, "AVG"]:
        measured_ade = baseline[f"ADE_{scene}"] if scene != "AVG" else baseline["AVG_ADE"]
        measured_fde = baseline[f"FDE_{scene}"] if scene != "AVG" else baseline["AVG_FDE"]
        reference_ade, reference_fde = PAPER_RESULTS[scene]
        comparison_rows.append(
            {
                "scene": scene,
                "measured_ADE": measured_ade,
                "paper_ADE": reference_ade,
                "ADE_relative_difference_pct": 100 * (measured_ade - reference_ade) / reference_ade,
                "measured_FDE": measured_fde,
                "paper_FDE": reference_fde,
                "FDE_relative_difference_pct": 100 * (measured_fde - reference_fde) / reference_fde,
            }
        )
    pd.DataFrame(comparison_rows).to_csv(root / "baseline_vs_paper.csv", index=False)

    print(f"Wrote analysis tables and plots under {root}")
    print(overall.to_string(index=False))
    print(f"Typical case examples available: {len(cases)}/3")


if __name__ == "__main__":
    main()
