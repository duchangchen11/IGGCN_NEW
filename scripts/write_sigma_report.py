"""Write the final fixed-sigma diagnostic report from generated result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENES = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
SIGMAS = [1, 2, 3, 4, 5]
STATE_FEATURES = ["v_mean", "density", "turning", "a_mean"]
STATE_NAMES = {
    "v_mean": "Speed",
    "density": "Density",
    "turning": "Turning",
    "a_mean": "Acceleration",
}


def markdown_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    columns = list(frame.columns)
    rows = ["| " + " | ".join(str(column) for column in columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, record in frame.iterrows():
        values = []
        for column in columns:
            value = record[column]
            if pd.isna(value):
                values.append("—")
            elif column in formats:
                values.append(format(value, formats[column]))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def best_sigma_by_group(summary: pd.DataFrame, feature: str) -> pd.DataFrame:
    subset = summary.loc[summary["state_feature"] == feature].copy()
    if subset.empty:
        return pd.DataFrame(columns=["state_group", "best_sigma", "best_ADE", "sigma3_ADE", "gain_vs_sigma3"])
    best_indices = subset.groupby("state_group")["ADE"].idxmin()
    best = subset.loc[best_indices, ["state_group", "sigma", "ADE"]].rename(
        columns={"sigma": "best_sigma", "ADE": "best_ADE"}
    )
    baseline = subset.loc[subset["sigma"] == 3, ["state_group", "ADE"]].rename(
        columns={"ADE": "sigma3_ADE"}
    )
    result = best.merge(baseline, on="state_group", how="left")
    result["gain_vs_sigma3"] = result["sigma3_ADE"] - result["best_ADE"]
    order = pd.Categorical(result["state_group"], ["low", "medium", "high"], ordered=True)
    return result.assign(_order=order).sort_values("_order").drop(columns="_order")


def fmt_markdown(frame: pd.DataFrame, decimals: int = 3) -> str:
    local = frame.copy()
    for column in local.select_dtypes(include="number").columns:
        local[column] = local[column].map(lambda value: f"{value:.{decimals}f}")
    return markdown_table(local)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path,
        default=PROJECT_ROOT / "results/iggcn_sigma_diagnostic",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "reports/iggcn_sigma_diagnostic/REPORT.md",
    )
    args = parser.parse_args()
    root = args.results_root

    scene = pd.read_csv(root / "sigma_scene_summary.csv")
    overall = pd.read_csv(root / "sigma_overall_summary.csv")
    baseline = pd.read_csv(root / "baseline_vs_paper.csv")
    state = pd.read_csv(root / "state_group_summary.csv")
    oracle = pd.read_csv(root / "oracle_sigma_analysis.csv")
    oracle_distribution = pd.read_csv(root / "oracle_best_sigma_distribution_summary.csv")
    cases_path = root / "typical_case_samples.csv"
    cases = pd.read_csv(cases_path) if cases_path.exists() else pd.DataFrame()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))

    baseline_three = overall.loc[overall["sigma"] == 3].iloc[0]
    scene_best_rows = []
    for test_scene in SCENES:
        column = f"ADE_{test_scene}"
        row = scene.loc[scene[column].idxmin()]
        scene_best_rows.append(
            {"Scene": test_scene, "best sigma": int(row["sigma"]), "ADE": row[column]}
        )
    scene_best = pd.DataFrame(scene_best_rows)

    state_tables = {}
    for feature in STATE_FEATURES:
        table = best_sigma_by_group(state, feature)
        table.insert(0, "state", table["state_group"])
        table = table[["state", "best_sigma", "best_ADE", "sigma3_ADE", "gain_vs_sigma3"]]
        state_tables[feature] = table

    oracle_ade3 = oracle["ADE_sigma3"].mean()
    oracle_ade = oracle["best_ADE"].mean()
    oracle_fde3 = oracle["FDE_sigma3"].mean()
    oracle_fde_selected = oracle["FDE_at_best_ADE_sigma"].mean()
    oracle_fde_independent = oracle["best_FDE_independent"].mean()
    oracle_rows = [
        {"Metric": "ADE", "Fixed sigma=3": oracle_ade3, "Oracle": oracle_ade,
         "Absolute gain": oracle_ade3 - oracle_ade,
         "Relative gain (%)": 100 * (oracle_ade3 - oracle_ade) / oracle_ade3},
        {"Metric": "FDE (sigma selected by per-sample ADE)", "Fixed sigma=3": oracle_fde3,
         "Oracle": oracle_fde_selected, "Absolute gain": oracle_fde3 - oracle_fde_selected,
         "Relative gain (%)": 100 * (oracle_fde3 - oracle_fde_selected) / oracle_fde3},
        {"Metric": "FDE (independently minimized; lower bound)", "Fixed sigma=3": oracle_fde3,
         "Oracle": oracle_fde_independent, "Absolute gain": oracle_fde3 - oracle_fde_independent,
         "Relative gain (%)": 100 * (oracle_fde3 - oracle_fde_independent) / oracle_fde3},
    ]
    oracle_table = pd.DataFrame(oracle_rows)

    oracle_mean_gain = float(oracle["ADE_gain_vs_sigma3"].mean())
    oracle_relative_gain = 100 * oracle_mean_gain / oracle_ade3
    best_global = overall.loc[overall["AVG_ADE"].idxmin()]
    best_global_fde = overall.loc[overall["AVG_FDE"].idxmin()]
    baseline_ade_gap = float(
        baseline.loc[baseline["scene"] == "AVG", "ADE_relative_difference_pct"].iloc[0]
    )
    baseline_fde_gap = float(
        baseline.loc[baseline["scene"] == "AVG", "FDE_relative_difference_pct"].iloc[0]
    )

    state_signal = False
    signal_lines = []
    for feature, table in state_tables.items():
        if len(table) >= 2 and table["best_sigma"].nunique() > 1:
            distinct = ", ".join(
                f"{row.state}={int(row.best_sigma)}" for row in table.itertuples()
            )
            max_gain = float(table["gain_vs_sigma3"].max())
            signal_lines.append(
                f"- {STATE_NAMES[feature]} 分组最优值不完全相同（{distinct}），"
                f"最大组内均值差为 {max_gain:.3f} m ADE。"
            )
            if max_gain > 0:
                state_signal = True
    if not signal_lines:
        signal_lines.append("- 四种状态的低/中/高组没有出现不同的组均值最优 σ。")

    distribution_sections = []
    for feature in STATE_FEATURES:
        subset = oracle_distribution.loc[
            oracle_distribution["state_feature"] == feature
        ].copy()
        if subset.empty:
            continue
        subset["best_sigma"] = subset["best_sigma"].astype(int)
        wide = subset.pivot(index="state_group", columns="best_sigma", values="proportion")
        wide = wide.reindex(index=["low", "medium", "high"], columns=[1, 2, 3, 4, 5]).fillna(0)
        wide.columns = [f"σ={sigma}" for sigma in wide.columns]
        wide.insert(0, "State group", wide.index)
        distribution_sections.append(
            f"**{STATE_NAMES[feature]} 的 oracle best-σ 比例**\n\n{fmt_markdown(wide.reset_index(drop=True), 3)}"
        )

    case_text = "没有找到同时满足对照条件的 σ=1 / 3 / 5 典型样本；不绘制误导性案例。"
    if not cases.empty:
        case_rows = []
        for row in cases.itertuples():
            case_rows.append(
                {"Case": row.case_type, "Scene": row.scene,
                 "Target": row.target_pedestrian_id, "Start frame": int(row.start_frame),
                 "ADE σ1": row.ADE_sigma1, "ADE σ3": row.ADE_sigma3,
                 "ADE σ5": row.ADE_sigma5}
            )
        case_text = fmt_markdown(pd.DataFrame(case_rows), 3)

    baseline_table = baseline[[
        "scene", "measured_ADE", "paper_ADE", "ADE_relative_difference_pct",
        "measured_FDE", "paper_FDE", "FDE_relative_difference_pct",
    ]].rename(columns={
        "scene": "Scene", "measured_ADE": "ADE ours", "paper_ADE": "ADE paper",
        "ADE_relative_difference_pct": "ADE diff (%)", "measured_FDE": "FDE ours",
        "paper_FDE": "FDE paper", "FDE_relative_difference_pct": "FDE diff (%)",
    })
    sigma_table = overall.rename(columns={"sigma": "σ", "AVG_ADE": "Mean ADE", "AVG_FDE": "Mean FDE"})
    scene_optimum_table = scene_best.rename(columns={"Scene": "Scene", "best sigma": "Best σ", "ADE": "Best ADE"})
    state_section = "\n\n".join(
        "### " + STATE_NAMES[feature] + "\n\n" + fmt_markdown(table, 3)
        for feature, table in state_tables.items()
    )
    signal_section = "\n".join(signal_lines)
    oracle_distribution_section = "\n\n".join(distribution_sections)
    speed_answer = fmt_markdown(state_tables["v_mean"], 3)
    density_answer = fmt_markdown(state_tables["density"], 3)

    report = f"""# IGGCN 固定 Gaussian 带宽敏感性与状态依赖诊断

## 摘要

本报告检验固定 Gaussian 带宽 σ 是否在 ETH/UCY 不同场景和 observed-only 行人状态下表现不同。第一轮使用 seed=42，比较 σ∈{{1,2,3,4,5}}。全部 25 个 leave-one-scene-out 运行已写入 manifest。原文没有发布 IGGCN 作者代码；本项目依据用户提供的论文 PDF 进行复现实现，模型细节假设见 `SOURCE_AUDIT.md` 与配置。

## 1. 代码来源与实验环境

- 论文：Wangxing Chen et al., *Digital Signal Processing* 156 (2025), 104862, DOI [10.1016/j.dsp.2024.104862](https://doi.org/10.1016/j.dsp.2024.104862)。本地来源 PDF SHA-256：`c8e022306aaf58bc67bfeafd4563fdf1785e21cf0ce05d817c37a79676b11714`。
- 作者 IGGCN 实现：审计未找到。模型是按论文结构重新实现；ETH/UCY 文本数据来自固定版本 Social-STGCNN 数据镜像，未使用其模型代码。数据校验和见数据盘 `dataset_manifest.json`。
- Python 3.10.9、PyTorch 2.0.1+cu117、torchvision 0.15.2+cu117、NVIDIA GeForce RTX 3080 10 GB。完整环境记录见 `PRE_RUN_ENVIRONMENT.md`。
- 实验配置：[`configs/iggcn_sigma_diagnostic.yaml`](../../configs/iggcn_sigma_diagnostic.yaml)。每次训练使用 4 场景训练、1 场景测试，8 帧 observed、12 帧预测、2.5 Hz；Adam、lr=0.01、每 50 epoch ×0.1、150 epoch、hidden=64、4 层 3×3 deformable convolution、5 层 TCN。
- 验证集取每个训练 clip 时间轴最后 10%，并在训练侧留出 19 个重叠窗口；以 validation ADE 选 checkpoint。论文没有说明这一选模规则。Gaussian σ 始终是不可训练的固定常数；没有加入新模块、修改 decoder 或替换 loss。
- Seed=42 固定初始化和样本顺序；cuDNN bitwise determinism 关闭、benchmark 关闭。实测开启确定性算法会使代表性更新耗时约 15 秒，默认 kernel 路径约 0.05 秒；此策略对所有 σ 一致，并写入 manifest。
- 论文未规定的 node encoding、TCN 细节、activation、temporal mask 具体落点和坐标参数化均按版本化配置实现，属于复现假设。模型参数量为约 31.9K，与论文报告 32.5K 接近，但架构细节不完整仍限制逐行复现。

## 2. σ=3 baseline 与论文结果

论文参考值为 ETH 0.57/0.92、HOTEL 0.29/0.45、UNIV 0.36/0.68、ZARA1 0.27/0.49、ZARA2 0.23/0.41，AVG 0.34/0.59（ADE/FDE，m）。本次 mean 以五个场景指标的算术平均计算。

{fmt_markdown(baseline_table, 3)}

AVG 相对论文差异：ADE {baseline_ade_gap:+.1f}%，FDE {baseline_fde_gap:+.1f}%。

## 3. 固定 σ sweep

{fmt_markdown(sigma_table.rename(columns={"σ": "sigma"}), 3)}

{fmt_markdown(scene_optimum_table, 3)}

场景 × σ ADE 热图：`results/iggcn_sigma_diagnostic/plots/scene_sigma_ade_heatmap.png`。原始折结果见 `sigma_sweep_seed42.csv`；汇总绘图数据见 `sigma_scene_summary.csv`。

## 4. Observed-only 状态分组

状态量只用 observed 8 帧：`v_mean`、`density`、`turning`、`a_mean`。每个 test scene 内按该状态量的秩分成三等分 low/medium/high；并列值使用平均秩，不用 future GT 生成分组。具体定义见配置。下表给出每一组的均值 ADE 最优固定 σ、其 ADE 和固定 σ=3 的差。

{state_section}

状态 × σ 指标表：`state_group_metrics.csv`（分 test scene 明细）与 `state_group_summary.csv`（按样本量汇总）。Speed、density、turning 热图在 `plots/`；acceleration 表格也保留在 CSV。

## 5. Oracle per-sample σ

对每个相同 `sample_id`，按五个模型 ADE 最小的 σ 作为 oracle；同一 σ 下的 FDE 一并报告。Oracle 使用了测试误差选择 σ，只是不可部署的理论下界，不是可实现模型结果。

{fmt_markdown(oracle_table, 3)}

Oracle 样本表包含 sample id、场景、状态、best sigma、五个 σ 的 ADE/FDE、observed / ground-truth / prediction 轨迹，路径：`results/iggcn_sigma_diagnostic/oracle_sigma_analysis.csv`。

{oracle_distribution_section}

## 6. 典型行人轨迹案例

案例仅从数据中实际存在的样本选择，不人为构造：

{case_text}

轨迹图路径：`results/iggcn_sigma_diagnostic/plots/case_*.png`；选中记录在 `typical_case_samples.csv`。

## 7. 研究问题回答

**Q1 全局平均哪个 σ 最好？** ADE 最优为 σ={int(best_global['sigma'])}（{best_global['AVG_ADE']:.3f} m）；FDE 最优为 σ={int(best_global_fde['sigma'])}（{best_global_fde['AVG_FDE']:.3f} m）。

**Q2 不同场景的最优 σ 是否一致？** 见第 3 节逐场景最优表。

**Q3 低速与高速最优 σ 是否一致？**

{speed_answer}

**Q4 低密度与高密度最优 σ 是否一致？**

{density_answer}

**Q5 直行与明显转向最优 σ 是否一致？** turning 的 low/high 分组结果见第 4 节。该变量是累计绝对 wrapped heading change，而不是 future 转向标签。

**Q6 σ=3 是否在某些状态有系统性损失？** 与每组最优固定 σ 的组均值差见第 4 节；Oracle paired error gain 的均值为 {oracle_mean_gain:.3f} m ADE（{oracle_relative_gain:.1f}%）。这是 seed=42 的描述性结果，没有跨 seed 置信区间。

**Q7 Oracle 相对 σ=3 理论提升多少？** ADE 改善 {oracle_ade3-oracle_ade:.3f} m（{oracle_relative_gain:.1f}%）；按 ADE 选中的 σ，FDE 改善 {oracle_fde3-oracle_fde_selected:.3f} m。独立逐样本最小化 FDE 的更低数值仅作额外下界。

**Q8 是否支持后续 state-dependent 带宽？** 观测到的描述性状态信号：

{signal_section}

当前实验决策：{"存在 seed=42 的状态组最优 σ 差异；可按原计划做多 seed 验证，但本报告不实现 Adaptive Sigma。" if state_signal else "当前 seed 未显示明确的状态组最优 σ 分化，不足以支持 Adaptive Sigma。按任务要求不进入多 seed sweep，等待进一步决定。"}

## 8. 局限与下一步

- 没有官方模型源码；论文省略若干关键张量和 TCN 细节，复现结果可能受实现假设影响。
- 单 seed、重叠窗口和同一 scene 内的样本相关性意味着组间差异不能当作统计显著性证据。
- Oracle 使用每个样本 future error 选 σ，必然乐观；只能作为可达上界诊断。
- 若状态组之间呈现稳定分化，按用户任务先重复 seed=42、123、2024 的固定 σ 实验并核对跨 seed 一致性；在得到确认前不实现 adaptive sigma。

## 9. 关键结果与图表

- sigma vs ADE：`results/iggcn_sigma_diagnostic/plots/sigma_vs_avg_ade.png`
- sigma vs FDE：`results/iggcn_sigma_diagnostic/plots/sigma_vs_avg_fde.png`
- scene × sigma ADE：`results/iggcn_sigma_diagnostic/plots/scene_sigma_ade_heatmap.png`
- speed / density / turning heatmaps：`results/iggcn_sigma_diagnostic/plots/`
- oracle state distribution：`results/iggcn_sigma_diagnostic/plots/oracle_best_sigma_state_distribution.png`
- 运行环境与来源：`reports/iggcn_sigma_diagnostic/PRE_RUN_ENVIRONMENT.md`、`reports/iggcn_sigma_diagnostic/SOURCE_AUDIT.md`
- 运行 manifest：`results/iggcn_sigma_diagnostic/run_manifest.json`（记录 {len(manifest)} 个完成运行）
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
