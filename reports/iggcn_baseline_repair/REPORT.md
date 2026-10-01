# IGGCN baseline reproduction audit and repair

## Scope and outcome

This branch audits the existing fixed-σ baseline, fixes the confirmed temporal-graph causality error, and revalidates σ=3. It does not run σ=1–5, add an adaptive bandwidth, or change the model parameter count. Existing artifacts under `reports/iggcn_sigma_diagnostic/` and `results/iggcn_sigma_diagnostic/` are preserved; all new outputs are under `reports/iggcn_baseline_repair/` and `results/iggcn_baseline_repair/`.

The confirmed error was the orientation of the temporal adjacency mask. For the implementation's matrix convention, rows are target times, columns are source times, and `A @ X` aggregates along source columns. The old upper triangle let target time `t` use source times after `t`. The repair uses `source <= target`, processes each deformable temporal-map prefix only through its target time, and adds no parameters. Two temporal-causality tests pass.

The paper's Table 1 reports mean ADE/FDE of 0.34/0.59 m across ETH/UCY. The old σ=3 result was 0.7578/1.4931 m. The full repaired five-scene mean is **0.7664/1.5196 m** (2.25× / 2.58× the paper values), so this repair does not pass the reproduction gate: the errors remain well above the reported values, and HOTEL/UNIV are clear scene-level anomalies against constant velocity. This conclusion applies the task's qualitative “near the paper magnitude, without obvious scene anomalies” condition; no percentage cutoff was added. The full fixed-σ=3 run completed all five 150-epoch folds. No σ sweep was run.

## Evaluation and data protocol audit

The paper reports 2.5 Hz data, 8 observed frames, 12 predicted frames, leave-one-scene-out testing, and ADE/FDE averaged over pedestrians. The selected Social-STGCNN text files have a per-track frame-gap mode of exactly 10 raw frames (100% of consecutive gaps); this pipeline uses stride 10 and 0.4 s per sample, matching 2.5 Hz. Test IDs and counts were verified against the original saved per-sample table.

The implementation uses the predicted bivariate-Gaussian mean for ADE/FDE. It predicts future offsets relative to the last observation and adds that position once; it does not cumulatively integrate predicted increments. Translation by the last position leaves ADE/FDE unchanged. The paper defines the bivariate-Gaussian NLL and pointwise ADE/FDE, but does not specify Gaussian sampling or best-of-K. No best-of-K score is used.

To make the distributional alternative explicit, existing checkpoints were also evaluated using 20 fixed-seed draws, independently sampled at each future time from the predicted bivariate Gaussian. The reported sample metric is the average over all 20 draws, without selecting the best draw. This is a diagnostic protocol, not a paper-comparable metric. Recomputed direct-mean metrics reproduce the old per-sample CSV within 1e-6 m.

| Scene | Old mean ADE/FDE | 20-draw mean ADE/FDE | Constant-velocity ADE/FDE |
|---|---:|---:|---:|
| ETH | 0.981 / 1.969 | 1.167 / 2.353 | 1.118 / 2.329 |
| HOTEL | 1.199 / 2.195 | 1.528 / 2.851 | 0.246 / 0.467 |
| UNIV | 0.835 / 1.592 | 1.079 / 2.036 | 0.689 / 1.392 |
| ZARA1 | 0.435 / 0.970 | 0.739 / 1.600 | 0.562 / 1.149 |
| ZARA2 | 0.339 / 0.740 | 0.548 / 1.158 | 0.428 / 0.871 |
| **Equal-weight scene mean** | **0.758 / 1.493** | **1.012 / 2.000** | **0.609 / 1.242** |

The mean prediction is substantially better than the sampled-path mean. HOTEL and UNIV are also clearly worse than the constant-velocity reference before repair, so they were selected for the required targeted rerun.

## Social graph composition

Current graph nodes are pedestrians with a complete, contiguous 8+12 window. For each test graph, raw pedestrians present during the 8 observed frames but absent from that complete-target set were counted as context-only candidates. The paper defines a set of N pedestrians for observation and prediction, but does not say whether pedestrians lacking a complete future should still enter the graph as context. The counts below establish a substantial omission from raw observed-frame context; they do not by themselves prove that the implementation violates the paper. No graph-construction change was made without an authoritative protocol for such partial tracks.

| Scene | Graph windows | Mean complete target nodes | Raw peds/frame | Context candidates omitted/frame | Observed person-frames omitted | Single-node graph fraction | Graph neighbors/target-frame | Raw same-frame neighbors/target-frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETH | 253 | 1.44 | 7.59 | 6.15 | 81.1% | 72.3% | 0.90 | 7.44 |
| HOTEL | 445 | 2.69 | 7.47 | 4.78 | 64.0% | 32.4% | 2.64 | 7.89 |
| UNIV | 947 | 25.70 | 40.79 | 15.09 | 37.0% | 0.0% | 28.74 | 44.33 |
| ZARA1 | 705 | 3.34 | 6.39 | 3.05 | 47.7% | 14.6% | 3.76 | 6.86 |
| ZARA2 | 998 | 5.92 | 9.54 | 3.62 | 37.9% | 7.7% | 6.49 | 10.09 |

“Graph neighbors” here means other complete target nodes available to the dense spatial graph, not a radius-filtered neighborhood. Raw same-frame neighbors count every other pedestrian ID in the source file at the target's observed frame. Full per-scene counts and definitions are in `results/iggcn_baseline_repair/pre_fix_audit/social_graph_composition.csv`.

## Temporal graph mask

The model computes temporal QK with `einsum("bntd,bnqd->bntq")`; therefore `t` is the target row and `q` is the source column. It multiplies adjacency by features as `A[target, source] @ X[source]`. The old upper-triangle row for target index 3 allowed source indices 3, 4, 5, 6, and 7, including four future frames, while excluding earlier sources 0–2. This conflicts with the paper's stated rule that a current frame is independent of future states. The paper's text calls its matrix upper-triangular but does not specify an alternate transpose convention; the executable convention makes the causal lower triangle necessary.

The temporal deformable convolution previously saw the full time-by-time QK matrix before masking. The repair runs it on each prefix `[0..t, 0..t]` and keeps the final target row, then applies a lower-triangular adjacency. Thus later observations cannot enter an earlier target row through either the adjacency or the deformable convolution. The model parameter count remains 31,940 (paper: 32.5K).

`tests/test_iggcn_temporal_causality.py` checks (1) perturbing any observation after `t` leaves temporal-branch features through `t` unchanged, and (2) row `t` has no source columns after `t`. Both tests pass. The one-checkpoint probe showed the legacy mask's future columns explicitly; for that HOTEL example their Gaussian weights had underflowed to zero, so the orientation error was real even though that particular sample did not aggregate nonzero future weights.

## Gaussian weights and gradients

Diagnostics were collected on up to 64 test graphs per scene using the existing best checkpoints, before and after applying the causal mask. The reported QK ranges and deformable outputs are in `results/iggcn_baseline_repair/post_mask_checkpoint_audit/gaussian_and_gradient_diagnostics.json`. QK and DCNN values can be very large relative to σ=3; the learned temporal weights are strongly concentrated near zero. The Gaussian equation was not changed because it matches the paper's stated `exp(-||R_ij-R_ii||²/(2σ²))` formula, and a large value alone does not establish a formula error.

| Scene | Spatial off-diagonal mean | Spatial `<1e-4` | Spatial `>=0.99` | Causal temporal off-diagonal mean | Temporal `<1e-4` | Temporal `>=0.99` |
|---|---:|---:|---:|---:|---:|---:|
| ETH | 0.490 | 51.0% | 48.9% | 0.000166 | 99.86% | 0.00% |
| HOTEL | 0.632 | 35.4% | 63.2% | 5.2e-9 | 100.0% | 0.00% |
| UNIV | 0.174 | 71.1% | 13.1% | 0.0114 | 94.35% | 0.095% |
| ZARA1 | 0.130 | 85.5% | 12.6% | 0.00291 | 97.71% | 0.067% |
| ZARA2 | 0.125 | 87.2% | 12.5% | 0.0151 | 94.35% | 0.165% |

All probed parameter tensors had finite gradients. Across a single test-batch NLL probe, 41–79% of parameter tensors had at least one nonzero gradient, varying by scene. Saturated weights can suppress some gradients; this probe does not establish that every branch learns effectively over the full training set. The precise QK/DCNN ranges, quantiles, gradient norms, and per-parameter details remain in the JSON diagnostic.

## Model implementation review

| Component | Current implementation | Paper comparison / unresolved point |
|---|---|---|
| Spatial graph | Pedestrian nodes, dense all-pairs QK, four residual 3×3 deformable layers, Gaussian adjacency plus identity, graph aggregation | Main structure and equations match. Exact inclusion of partial-track context is unspecified. |
| Temporal graph | Per-pedestrian time nodes; QK row=target/column=source; causal lower triangle; prefix deformable processing | Causal semantics now hold. The paper's upper-triangle sentence lacks an explicit matrix orientation and conflicts with target-row/source-column multiplication. |
| Query/Key | Separate linear projections; unscaled `QKᵀ`; 64-d hidden features | Consistent with displayed equations; no `sqrt(d)` factor is shown in the paper. Input feature parameterization is not explicit. |
| Deformable convolution | Four residual layers; 3×3; offset predictor maps 1 channel to 18 offsets, zero-initialized; same padding; no activation inside branch | Four layers and residual equations are stated. Offset initialization, padding, and activation are not fully specified. |
| Spatial/temporal fusion | Linear graph transforms, sum, then ReLU | Sum is Eq. (6); activation `δ` is unspecified. |
| TCN | Five causal Conv1d blocks, 24 channels, kernel 3, dilations 1/2/4/8/16; final observed output projected to 12×5 | Five layers are stated. Width, kernel, dilation, padding, and projection are not. Width was chosen before this audit and was not changed to chase the reported parameter count. |
| Output coordinates | Bivariate Gaussian parameters over future offsets from the last observed position; add the last position once for metrics | Paper defines μ/σ/ρ and NLL, but does not state absolute-vs-relative target parameterization. ADE/FDE are translation-invariant under this conversion. |
| Loss | Per-agent sum of 12 bivariate Gaussian NLL terms, then mean over valid targets | Matches the paper's summed-per-time NLL form. |
| Checkpoint | Lowest validation ADE on chronological final 10% of training clips; 19-window purge | Not stated in the paper; retained unchanged for controlled comparison. |

No model modules, Gaussian formula, output head, or loss were added or replaced. No attempt was made to adjust widths to force an exact 32.5K parameter count.

## Before/after checkpoint and training results

First, existing checkpoints were evaluated with the repaired temporal branch to isolate the mask code change without retraining. The targeted HOTEL and UNIV folds were then trained under the fixed σ=3 protocol. The full five-fold repaired baseline then retrained all folds and evaluated the identical 34,161 test sample IDs. The old and new sample-ID sets match exactly; all new ADE/FDE values are finite, and per-sample averages recreate the fold CSV.

| Scene | Test targets | Old baseline | Old checkpoint + causal mask | Retrained targeted folds | Full repaired σ=3 baseline | Constant velocity |
|---|---:|---:|---:|---:|---:|---:|
| ETH | 364 | 0.981 / 1.969 | 0.981 / 1.969 | — | 1.040 / 2.101 | 1.118 / 2.329 |
| HOTEL | 1,197 | 1.199 / 2.195 | 1.199 / 2.195 | 1.197 / 2.161 | 1.202 / 2.216 | 0.246 / 0.467 |
| UNIV | 24,334 | 0.835 / 1.592 | 0.836 / 1.592 | 0.802 / 1.543 | 0.814 / 1.570 | 0.689 / 1.392 |
| ZARA1 | 2,356 | 0.435 / 0.970 | 0.443 / 0.979 | — | 0.438 / 0.974 | 0.562 / 1.149 |
| ZARA2 | 5,910 | 0.339 / 0.740 | 0.355 / 0.768 | — | 0.338 / 0.737 | 0.428 / 0.871 |
| **Equal-weight scene mean** | **34,161** | **0.758 / 1.493** | **0.763 / 1.501** | **0.999 / 1.852** (HOTEL+UNIV) | **0.766 / 1.520** | **0.609 / 1.242** |

The same-checkpoint mask-only comparison shows that the causal fix does not itself explain the large baseline gap. Targeted HOTEL/UNIV retraining modestly improves their two-scene mean, but the full retrained run is slightly worse overall than the old baseline: ADE rises by 0.0086 m and FDE by 0.0265 m. HOTEL and UNIV remain substantially worse than constant velocity. The remaining discrepancy must be interpreted alongside the unresolved graph-composition and model-specification choices rather than hidden by a metric change.

## Final repaired σ=3 baseline

The fixed-sigma revalidation used seed 42, obs=8, pred=12, σ=3, four training scenes and one held-out test scene per fold, 150 epochs, and the existing validation checkpoint rule. The five folds contain exactly the same held-out targets as the old run. The repaired equal-weight mean is 0.7664/1.5196 m versus 0.7578/1.4931 m before the change and 0.34/0.59 m in the paper. HOTEL and UNIV remain conspicuously worse than the constant-velocity baseline. **The reproduction gate is not passed.** Do not proceed to a σ sweep or Adaptive Sigma without a separate research decision.

The training runner retains its generic filename `sigma_sweep_seed42.csv`; the file in this repair directory has only five rows and every row is σ=3. It records the five-scene baseline and is not a σ sweep.

## Reproducibility files

- Existing baseline source: commit `e4877b18dfb41238c807bc2ff1008607efe375c9`.
- Repair code/audit commit: `0b841a4cbf1992efc92d89e3e459f1970880ae02`.
- New configuration: `configs/iggcn_baseline_repair.yaml` (targeted HOTEL/UNIV) and `configs/iggcn_baseline_repair_full.yaml` (full σ=3 validation).
- Full audit script: `scripts/audit_baseline_reproduction.py`.
- Result-integrity validator: `scripts/validate_repair_results.py`; its sample-ID, finite-value, per-scene aggregation, and gate summary is `results/iggcn_baseline_repair/full_sigma3_baseline/validation_summary.json`.
- Temporal-mask probe: `scripts/probe_temporal_mask.py`.
- Original article: Chen et al., *Digital Signal Processing* 156 (2025), 104862, [DOI 10.1016/j.dsp.2024.104862](https://doi.org/10.1016/j.dsp.2024.104862).

The requested unit tests and repository checks passed at final review: 14 tests passed, Python sources compiled, and `git diff --check` reported no whitespace errors. Checkpoint binaries are retained locally but excluded from the commit; result tables, manifests, configs, diagnostics, and per-sample CSVs are versioned.
