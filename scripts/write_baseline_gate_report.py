"""Write the required stop-gate report when the sigma=3 baseline deviates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = {
    "ETH": (0.57, 0.92),
    "HOTEL": (0.29, 0.45),
    "UNIV": (0.36, 0.68),
    "ZARA1": (0.27, 0.49),
    "ZARA2": (0.23, 0.41),
}
SCENES = list(PAPER)


def table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    output = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column, value in row.items():
            if pd.isna(value):
                values.append("—")
            elif column in {"best_epoch", "targets"}:
                values.append(str(int(value)))
            elif isinstance(value, (float, int)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


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
    baseline = pd.read_csv(root / "baseline_reproduction.csv")
    if set(baseline["test_scene"]) != set(SCENES) or len(baseline) != len(SCENES):
        raise ValueError("baseline gate report requires exactly one sigma=3 row for each ETH/UCY scene")

    rows = []
    for scene in SCENES:
        result = baseline.loc[baseline["test_scene"] == scene].iloc[0]
        paper_ade, paper_fde = PAPER[scene]
        rows.append({
            "Scene": scene,
            "ADE ours": result["ADE"],
            "ADE paper": paper_ade,
            "ADE diff (%)": 100 * (result["ADE"] - paper_ade) / paper_ade,
            "FDE ours": result["FDE"],
            "FDE paper": paper_fde,
            "FDE diff (%)": 100 * (result["FDE"] - paper_fde) / paper_fde,
            "best_epoch": int(result["best_epoch"]),
            "targets": int(result["test_targets"]),
        })
    comparison = pd.DataFrame(rows)
    mean_ade = comparison["ADE ours"].mean()
    mean_fde = comparison["FDE ours"].mean()
    paper_mean_ade, paper_mean_fde = 0.34, 0.59
    ade_gap = 100 * (mean_ade - paper_mean_ade) / paper_mean_ade
    fde_gap = 100 * (mean_fde - paper_mean_fde) / paper_mean_fde
    comparison.loc[len(comparison)] = {
        "Scene": "AVG",
        "ADE ours": mean_ade,
        "ADE paper": paper_mean_ade,
        "ADE diff (%)": ade_gap,
        "FDE ours": mean_fde,
        "FDE paper": paper_mean_fde,
        "FDE diff (%)": fde_gap,
        "best_epoch": float("nan"),
        "targets": int(baseline["test_targets"].sum()),
    }
    comparison.to_csv(root / "baseline_vs_paper.csv", index=False)

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    x = range(len(SCENES))
    width = 0.36
    for axis, ours_col, paper_col, metric in [
        (axes[0], "ADE ours", "ADE paper", "ADE (m)"),
        (axes[1], "FDE ours", "FDE paper", "FDE (m)"),
    ]:
        scene_rows = comparison.iloc[: len(SCENES)]
        axis.bar([item - width / 2 for item in x], scene_rows[paper_col], width, label="Paper")
        axis.bar([item + width / 2 for item in x], scene_rows[ours_col], width, label="Reproduction")
        axis.set_xticks(list(x), SCENES)
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.savefig(plot_dir / "baseline_vs_paper.png", dpi=220)
    plt.close(fig)

    gate_failed = ade_gap > 15 or fde_gap > 15
    gate_status = "触发，停止后续 sigma sweep" if gate_failed else "未由均值误差触发"
    next_steps = """
1. 对照 benchmark 常用 ETH/UCY 坐标归一化与预测目标参数化；当前代码保留原坐标尺度并用最后 observed 点作平移原点。
2. 复核 temporal graph 上三角矩阵在 QK / adjacency 中的准确使用位置，以及空间/时间 QK 的索引方向。
3. 获取或确认论文未公开的 TCN 通道、卷积、激活和输出投影；目前选择是实现假设，不应继续用调参来掩盖基线差异。
4. 保留逐样本 ADE/FDE 检查，并核对 per-target 窗口协议是否与论文实现使用的样本权重一致。
5. 修正上述协议/结构差异并重新通过 σ=3 baseline gate 后，才重跑固定 σ=1–5 对照。
""".strip()
    report = f"""# IGGCN 固定 Gaussian 带宽诊断：基线停止门报告

## 执行结论

σ=3 的五折基线完整跑完。平均 ADE 相对论文高 {ade_gap:.1f}%，平均 FDE 高 {fde_gap:.1f}%；按任务停止规则，{gate_status}。目前没有运行 σ=1/2/4/5，没有进行状态分组、oracle σ 或典型案例比较，也没有实现 Adaptive Sigma。

## 基线结果

论文 ETH/UCY 报告值为 ETH 0.57/0.92、HOTEL 0.29/0.45、UNIV 0.36/0.68、ZARA1 0.27/0.49、ZARA2 0.23/0.41、AVG 0.34/0.59（ADE/FDE，米）。本项目按五个场景的指标算术平均计算 AVG。

{table(comparison)}

逐场景可视化：`results/iggcn_sigma_diagnostic/plots/baseline_vs_paper.png`。五折运行、best epoch、样本数及时间见 `results/iggcn_sigma_diagnostic/run_manifest.json`。

## 基线审计

- 数据是 ETH、HOTEL、UNIV 两个独立 clip、ZARA1、ZARA2；源文件和 SHA-256 在数据盘 `/media/lrj/54926A1D926A0438/datasets/iggcn_eth_ucy/raw/dataset_manifest.json`。项目不把原始轨迹提交到 Git。
- 数据来自固定版本的 [Social-STGCNN data mirror](https://github.com/abduallahmohamed/Social-STGCNN/tree/333d3a57b4d2705e129b21aefefa09c79b2b9ae1/datasets/raw/all_data)，仅用其 ETH/UCY 数据文件，没有复用该项目的模型。
- 当前抽窗检查仅接受相邻采样点 raw frame stride=10，观察 8 帧、未来 12 帧；原数据帧率对应 2.5 Hz。UNIV 两个片段分开处理；34,161 个测试目标窗口与样本索引数量完全一致。
- split 为 4 个训练 scene、1 个测试 scene；验证数据从训练 clip 尾段按时间保留，并 purge 重叠窗口。
- 训练设置：seed=42、150 epochs、batch size 128、Adam、初始 lr=0.01，每 50 epochs 学习率乘 0.1、hidden=64、4 层 3×3 deformable convolution、5 层 TCN、固定 σ=3。按训练 clip 尾段验证集的最低 ADE 选模型，这是论文没有说明的实现选择。
- Python 3.10.9、PyTorch 2.0.1+cu117、torchvision 0.15.2+cu117、RTX 3080 10 GB；seed 固定初始化和样本顺序，cuDNN deterministic=false / benchmark=false。
- 坐标未做 per-scene 平移或尺度归一化；图节点用相邻 observed 位置差，预测坐标以最后 observed 点为原点。原文没有给出额外坐标 normalization 细节，因此仍需与官方预处理确认。
- ADE 是 12 帧欧氏距离的逐点均值，FDE 是终点欧氏距离；每个目标窗口都保存了单独误差。
- Gaussian 公式为 `exp(-||R_ij-R_ii||²/(2σ²))`，σ=3 是固定 Python scalar，不属于 optimizer；空间和时间分支使用同一个固定值。单元检查已覆盖 σ=1–5 的衰减方向与自相似为 1。
- 在[作者公开 GitHub 账户](https://github.com/Chenwangxing)中未找到与 IGGCN 对应的实现仓库；TCN 通道/卷积、节点张量细节、时间图 mask 位置和 checkpoint 规则在论文中没有完整定义，因此本次结果只能称作基于论文的实现基线，不能视为官方复现。

## 停止后的排查次序

{next_steps}

## 当前阶段结论

该基线没有通过论文量级核验，现有数据不能回答固定 σ 是否存在 state-dependent limitation。state 分析留待基线问题排除后再做。固定 σ sweep 已停止；不继续多 seed，也不实现新模块、decoder 或 loss。

代码与实验目录：`/home/lrj/IGGCN_Project`，实验分支 `exp/iggcn_sigma_diagnostic`。实验源代码按用户指定保存在 [IGGCN_NEW](https://github.com/duchangchen11/IGGCN_NEW) 对应 Git 分支。
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote baseline stop-gate report: {args.output}")
    print(f"AVG ADE={mean_ade:.4f} ({ade_gap:+.1f}%), FDE={mean_fde:.4f} ({fde_gap:+.1f}%), gate_failed={gate_failed}")


if __name__ == "__main__":
    main()
