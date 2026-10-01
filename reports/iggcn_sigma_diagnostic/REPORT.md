# IGGCN 固定 Gaussian 带宽诊断：基线停止门报告

## 执行结论

σ=3 的五折基线完整跑完。平均 ADE 相对论文高 122.9%，平均 FDE 高 153.1%；按任务停止规则，触发，停止后续 sigma sweep。目前没有运行 σ=1/2/4/5，没有进行状态分组、oracle σ 或典型案例比较，也没有实现 Adaptive Sigma。

## 基线结果

论文 ETH/UCY 报告值为 ETH 0.57/0.92、HOTEL 0.29/0.45、UNIV 0.36/0.68、ZARA1 0.27/0.49、ZARA2 0.23/0.41、AVG 0.34/0.59（ADE/FDE，米）。本项目按五个场景的指标算术平均计算 AVG。

| Scene | ADE ours | ADE paper | ADE diff (%) | FDE ours | FDE paper | FDE diff (%) | best_epoch | targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETH | 0.981 | 0.570 | 72.094 | 1.969 | 0.920 | 114.014 | 77 | 364 |
| HOTEL | 1.199 | 0.290 | 313.383 | 2.195 | 0.450 | 387.694 | 64 | 1197 |
| UNIV | 0.835 | 0.360 | 132.076 | 1.592 | 0.680 | 134.057 | 16 | 24334 |
| ZARA1 | 0.435 | 0.270 | 61.057 | 0.970 | 0.490 | 98.008 | 120 | 2356 |
| ZARA2 | 0.339 | 0.230 | 47.379 | 0.740 | 0.410 | 80.523 | 63 | 5910 |
| AVG | 0.758 | 0.340 | 122.885 | 1.493 | 0.590 | 153.069 | — | 34161 |

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

1. 对照 benchmark 常用 ETH/UCY 坐标归一化与预测目标参数化；当前代码保留原坐标尺度并用最后 observed 点作平移原点。
2. 复核 temporal graph 上三角矩阵在 QK / adjacency 中的准确使用位置，以及空间/时间 QK 的索引方向。
3. 获取或确认论文未公开的 TCN 通道、卷积、激活和输出投影；目前选择是实现假设，不应继续用调参来掩盖基线差异。
4. 保留逐样本 ADE/FDE 检查，并核对 per-target 窗口协议是否与论文实现使用的样本权重一致。
5. 修正上述协议/结构差异并重新通过 σ=3 baseline gate 后，才重跑固定 σ=1–5 对照。

## 当前阶段结论

该基线没有通过论文量级核验，现有数据不能回答固定 σ 是否存在 state-dependent limitation。state 分析留待基线问题排除后再做。固定 σ sweep 已停止；不继续多 seed，也不实现新模块、decoder 或 loss。

代码与实验目录：`/home/lrj/IGGCN_Project`，实验分支 `exp/iggcn_sigma_diagnostic`。实验源代码按用户指定保存在 [IGGCN_NEW](https://github.com/duchangchen11/IGGCN_NEW) 对应 Git 分支。
