# IGGCN evaluation and coordinate protocol audit

## Paper protocol

The source audit for the user-supplied 12-page PDF records the paper's bivariate-Gaussian NLL, ADE/FDE, 8 observed / 12 predicted frames at 2.5 Hz, and leave-one-scene-out ETH/UCY evaluation. The PDF's recorded SHA-256 is `c8e022306aaf58bc67bfeafd4563fdf1785e21cf0ce05d817c37a79676b11714`. The paper presents Gaussian parameters and an NLL, then reports ADE/FDE, but does not explicitly say whether those point metrics use the Gaussian mean, random Gaussian draws, or a best-of-K selection. The accessible source gives no best-of-K instruction. The exact ADE/FDE decoder therefore remains underspecified in the paper; sampling a trajectory and selecting a favorable draw would be unjustified.

The paper's stated interaction equation is the Gaussian similarity

\[
E_{ij}=\exp\left(-\frac{\|R_{ij}-R_{ii}\|^2}{2\sigma^2}\right).
\]

The article is Chen et al., *Digital Signal Processing* 156 (2025), 104862, [DOI 10.1016/j.dsp.2024.104862](https://doi.org/10.1016/j.dsp.2024.104862), also indexed on [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S105120042400486X).

**Source-access limitation for this phase:** the previously supplied PDF and raw ETH/UCY volume are no longer mounted at `/media/lrj/54926A1D926A0438`; the stored source audit and results remain in the repository. This phase did not re-open the local PDF. The source statements above are cross-checked against the saved `reports/iggcn_sigma_diagnostic/SOURCE_AUDIT.md` and the publisher's article record.

## Current implementation

- Training targets are relative offsets: `future_xy - last_observed_xy`.
- `IGGCN.forward()` returns five bivariate-Gaussian parameters per future frame; the first two values are the mean offset \(\mu_t\).
- `IGGCN.predict_positions()` returns `mu_t + last_observed_xy` exactly once for each future frame. It does not cumulatively integrate the outputs.
- ADE is the mean Euclidean error over the 12 decoded positions; FDE is the Euclidean error at the final position. Per-target errors are averaged within each scene; the headline result is the equal-weight mean of the five scene metrics.
- Evaluation uses the Gaussian mean trajectory. Sampling and best-of-K are not used. The prior audit evaluated 20 seeded sample paths as a separate diagnostic and found their mean error worse; it did not use those scores to replace the mean metric.

## Comparison and conclusion

There is no confirmed evaluation mismatch to repair. Recalculation from the saved trajectory strings reproduces per-target ADE/FDE within `4.3e-6 m` / `1.1e-5 m`; the small residual comes from trajectories serialized to seven significant digits. The original saved metric columns remain the source of record: equal-scene mean `0.757810 / 1.493105 m`. No evaluation code or reported score was changed.

For the saved example `ETH:biwi_eth:2:800`, the last observed point is `(7.17, 6.62) m`, the first ground-truth future point is `(6.47, 6.68) m`, and the first predicted mean point is `(6.376697, 6.682224) m`. The first predicted offset is `(-0.793303, 0.062224) m`, which is decoded relative to the observed point, not interpreted as a new absolute point or integrated with the previous forecast. A synthetic unit test verifies the addition and rejects cumulative integration.

The paper's exact mean-versus-sampling choice remains unspecified. The current direct-mean metric is transparent and consistent with deterministic ADE/FDE reporting, but exact protocol identity cannot be claimed from the article text available in this environment.
