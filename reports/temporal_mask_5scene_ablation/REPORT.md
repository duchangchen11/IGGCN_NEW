# IGGCN 五场景 Temporal Mask 严格对照实验

## 实验协议

- 唯一模型行为变量：时间邻接掩码方向 `torch.triu` / `torch.tril`；共用同一份 `research/iggcn/model.py`。
- 原始数据：`/media/lrj/54926A1D926A0438/datasets/iggcn_eth_ucy/raw/`，六个 ETH/UCY 原始文件，SHA-256 记录于 `results/temporal_mask_5scene_ablation/raw_dataset_manifest.json`。本实验从只读挂载的原始数据读取，未使用 CSV 重建轨迹。
- Leave-one-scene-out；8 observed / 12 predicted，2.5 Hz，σ=3，seed=42，150 epochs，batch=128，Adam 0.01，每 50 epochs 学习率乘 0.1，hidden=64，4 层 deformable convolution，5 层 TCN。
- 验证集划分、checkpoint 选择、训练目标与评估方式沿用当前 baseline 的统一实现。每对使用相同初始化 state hash、训练样本顺序和测试 sample IDs。
- GPU: RTX 3080；cuDNN deterministic=false。指标为描述性单 seed 对照，不代表跨 seed 显著性结论。
- 代码 commit（每个运行 manifest 记录）见 `run_manifest.json`；模型源码 SHA-256：`2e24669cab4c8c8d2755ca768f66d6ae0406d75e059ce6a0bbf2f0247c57df02`。

## 五场景结果

ADE/FDE 单位为米；AVG 对五个场景等权宏平均，不按 pedestrian target 数加权。

| Scene | Paper_ADE | Paper_FDE | CV_ADE | CV_FDE | triu_ADE | triu_FDE | tril_ADE | tril_FDE | ADE_improve_pct | FDE_improve_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETH | 0.570 | 0.920 | 1.118 | 2.329 | 0.991 | 1.990 | 1.010 | 2.051 | -1.937 | -3.053 |
| HOTEL | 0.290 | 0.450 | 0.246 | 0.467 | 1.219 | 2.233 | 1.218 | 2.243 | 0.074 | -0.455 |
| UNIV | 0.360 | 0.680 | 0.689 | 1.392 | 0.790 | 1.536 | 0.794 | 1.537 | -0.446 | -0.069 |
| ZARA1 | 0.270 | 0.490 | 0.562 | 1.149 | 0.439 | 0.969 | 0.427 | 0.945 | 2.579 | 2.388 |
| ZARA2 | 0.230 | 0.410 | 0.428 | 0.871 | 0.335 | 0.723 | 0.349 | 0.727 | -4.213 | -0.548 |
| AVG | 0.344 | 0.590 | 0.609 | 1.242 | 0.755 | 1.490 | 0.760 | 1.501 | -0.652 | -0.709 |

triu macro: **0.7547/1.4901**. tril macro: **0.7596/1.5006**. Equal-weight paper reference used by the provided scene values is **0.344/0.590**.

## Paired sample analysis

`paired_scene_metrics.csv` reports each scene's mean delta (`tril − triu`), relative improvement, percentage of targets where tril is better, and paired delta p50/p75/p90/p95/p99. Negative deltas favor tril.

| Scene | test_targets | ADE_delta_tril_minus_triu | ADE_improve_pct | ADE_tril_better_fraction | FDE_delta_tril_minus_triu | FDE_improve_pct | FDE_tril_better_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETH | 364 | 0.0192 | -1.9373 | 0.4698 | 0.0608 | -3.0533 | 0.4698 |
| HOTEL | 1197 | -0.0009 | 0.0739 | 0.4678 | 0.0102 | -0.4548 | 0.4495 |
| UNIV | 24334 | 0.0035 | -0.4462 | 0.5148 | 0.0011 | -0.0690 | 0.5047 |
| ZARA1 | 2356 | -0.0113 | 2.5789 | 0.5323 | -0.0231 | 2.3878 | 0.5407 |
| ZARA2 | 5910 | 0.0141 | -4.2132 | 0.3623 | 0.0040 | -0.5481 | 0.3868 |

Paired delta quantiles (meters; negative favors tril):

| Scene | ADE_delta_p50 | ADE_delta_p75 | ADE_delta_p90 | ADE_delta_p95 | ADE_delta_p99 | FDE_delta_p50 | FDE_delta_p75 | FDE_delta_p90 | FDE_delta_p95 | FDE_delta_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETH | 0.0031 | 0.1783 | 0.4314 | 0.6422 | 1.1244 | 0.0120 | 0.3529 | 0.7723 | 1.2725 | 2.3388 |
| HOTEL | 0.0050 | 0.0676 | 0.1004 | 0.1286 | 0.2781 | 0.0289 | 0.1435 | 0.2820 | 0.3621 | 0.5668 |
| UNIV | -0.0037 | 0.1119 | 0.2972 | 0.4152 | 0.6789 | -0.0023 | 0.1861 | 0.4476 | 0.6425 | 1.0893 |
| ZARA1 | -0.0049 | 0.0408 | 0.0932 | 0.1315 | 0.2348 | -0.0199 | 0.0851 | 0.1820 | 0.2602 | 0.4469 |
| ZARA2 | 0.0016 | 0.0479 | 0.1660 | 0.2698 | 0.6205 | 0.0040 | 0.0868 | 0.3249 | 0.5083 | 1.0285 |

Marginal p90 errors by mask (meters; equal-weight AVG row):

| Scene | ADE_p90_triu | ADE_p90_tril | ADE_p90_delta_tril_minus_triu | FDE_p90_triu | FDE_p90_tril | FDE_p90_delta_tril_minus_triu |
| --- | --- | --- | --- | --- | --- | --- |
| ETH | 2.2158 | 2.1550 | -0.0608 | 4.8546 | 4.7738 | -0.0808 |
| HOTEL | 3.7335 | 3.7562 | 0.0227 | 6.8139 | 6.9263 | 0.1124 |
| UNIV | 1.6352 | 1.6733 | 0.0381 | 3.1405 | 3.1907 | 0.0503 |
| ZARA1 | 0.8357 | 0.8097 | -0.0260 | 1.8282 | 1.7927 | -0.0355 |
| ZARA2 | 0.8537 | 0.8402 | -0.0134 | 1.8947 | 1.7702 | -0.1245 |
| AVG | 1.8548 | 1.8469 | -0.0079 | 3.7064 | 3.6907 | -0.0156 |

Per-sample paired distribution summaries:

- ETH: ADE paired median delta 0.003 m, p90 0.431 m; FDE paired median delta 0.012 m, p90 0.772 m.
- HOTEL: ADE paired median delta 0.005 m, p90 0.100 m; FDE paired median delta 0.029 m, p90 0.282 m.
- UNIV: ADE paired median delta -0.004 m, p90 0.297 m; FDE paired median delta -0.002 m, p90 0.448 m.
- ZARA1: ADE paired median delta -0.005 m, p90 0.093 m; FDE paired median delta -0.020 m, p90 0.182 m.
- ZARA2: ADE paired median delta 0.002 m, p90 0.166 m; FDE paired median delta 0.004 m, p90 0.325 m.

By mean metrics, strongest ADE benefit is ZARA1 and strongest FDE benefit is ZARA1. Smallest/negative changes are ZARA2 for ADE and ETH for FDE. Scenes improving both ADE and FDE: ZARA1. Scenes with regression in at least one metric: ETH, HOTEL, UNIV, ZARA2.

## Constant Velocity comparison

The already validated same-split per-target CV predictions were matched by sample ID against this raw-data run before producing the means below; the ID sets and target counts must match exactly.

- ETH: tril beats CV (tril 1.010/2.051; CV 1.118/2.329).
- HOTEL: tril does not beat on both metrics CV (tril 1.218/2.243; CV 0.246/0.467).
- UNIV: tril does not beat on both metrics CV (tril 0.794/1.537; CV 0.689/1.392).
- ZARA1: tril beats CV (tril 0.427/0.945; CV 0.562/1.149).
- ZARA2: tril beats CV (tril 0.349/0.727; CV 0.428/0.871).

## Temporal Gaussian off-diagonal statistics

`gaussian_statistics.csv` preserves mean, median, p10, p50, p90, p99 and all four requested threshold proportions for both raw similarity and post-mask effective adjacency. `raw_similarity_*` summarizes off-diagonal temporal Gaussian similarities before the binary temporal mask. `effective_adjacency_*` summarizes the corresponding post-mask off-diagonal weights; triu can intentionally zero causal-incompatible edges. Threshold proportions are fractions, not percentages.

| scene | mask_direction | raw_similarity_mean | raw_similarity_median | raw_similarity_p10 | raw_similarity_p50 | raw_similarity_p90 | raw_similarity_p99 | raw_similarity_lt_1e-4 | raw_similarity_lt_1e-6 | raw_similarity_ge_0.9 | raw_similarity_ge_0.99 | effective_adjacency_lt_1e-4 | effective_adjacency_ge_0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETH | triu | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| ETH | tril | 0.0011 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9966 | 0.9957 | 0.0006 | 0.0002 | 0.9971 | 0.0006 |
| HOTEL | triu | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9927 | 0.9889 | 0.0004 | 0.0002 | 1.0000 | 0.0000 |
| HOTEL | tril | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0044 | 0.9864 | 0.9771 | 0.0003 | 0.0001 | 0.9864 | 0.0003 |
| UNIV | triu | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| UNIV | tril | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9988 | 0.9985 | 0.0001 | 0.0000 | 0.9988 | 0.0001 |
| ZARA1 | triu | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| ZARA1 | tril | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9979 | 0.9973 | 0.0002 | 0.0001 | 0.9979 | 0.0002 |
| ZARA2 | triu | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| ZARA2 | tril | 0.0039 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0708 | 0.9861 | 0.9849 | 0.0017 | 0.0005 | 0.9861 | 0.0017 |

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

`validation_summary.json` status: **passed**; completed runs: **10/10**; parameter count(s): **[31940]**; shared initial parameter-state hash: `4d905e592846b9b8fa3875a71c12a9092270c25f7731cfd269c24ae3e4e35f17`. All five paired ID sets, finite metrics, σ=3, seed=42, 150 epochs and per-scene target counts were checked. There is one shared model source and no Adaptive Sigma or new model module.

## 研究问题与结论

1. **tril 是否一致优于 triu？** 两项均改善的场景：ZARA1；需结合五场景结果表判断，不把单个 HOTEL 结果外推。
2. **最大收益场景？** ADE: ZARA1; FDE: ZARA1。
3. **收益最小/退化场景？** ADE 最小改善: ZARA2; FDE 最小改善: ETH。
4. **宏平均？** triu 0.7547/1.4901; tril 0.7596/1.5006。
5. **tril 是否超过 CV？** 见上方逐场景 CV 比较；超过需 ADE 和 FDE 同时更低。
6. **离论文差距？** tril macro 与等权论文参考 0.344/0.590 的差分别为 0.4156/0.9106。
7. **整体改善还是 tail 改善？** 分布表现混合：配对中位数在 ADE 2/5、FDE 2/5 场景改善；配对 p90 在 ADE 0/5、FDE 0/5 场景改善。 这是配对分布的描述性结论，未做跨 seed 显著性检验。
8. **Gaussian saturation？** 十个 run 的原始 Gaussian 非对角权重均有过半低于 1e-4，存在广泛饱和。 raw off-diagonal 权重小于 1e-4 的比例范围为 0.986–1.000，大于等于 0.99 的比例范围为 0.000–0.001。低于 1e-4 的比例在 5/5 个成对场景中 triu 高于 tril（单 seed 描述性结果）。 有效邻接 post-mask 统计单独列出，避免将结构性掩码零值误判为 Gaussian 饱和。
9. **是否可固定 tril？** 否，当前五场景结果未满足任务设定的候选判据（至少四场景 ADE/FDE 均改善且两项 macro 均下降）。
10. **是否进入 sigma diagnostic？** **否。** 本报告仅验证 mask 方向；仍需依据 baseline 复现量级和多 seed 稳健性另行科研判断。本实验没有运行 sigma sweep。

## 输出清单

- `results/temporal_mask_5scene_ablation/summary.csv`
- `results/temporal_mask_5scene_ablation/run_manifest.json`
- `results/temporal_mask_5scene_ablation/validation_summary.json`
- `results/temporal_mask_5scene_ablation/paired_scene_metrics.csv`
- `results/temporal_mask_5scene_ablation/gaussian_statistics.csv`
- `results/temporal_mask_5scene_ablation/per_sample/`
- `reports/temporal_mask_5scene_ablation/REPORT.md`
