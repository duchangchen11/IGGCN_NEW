# IGGCN fixed Gaussian bandwidth diagnostic

This repository contains an isolated reproduction and diagnostic study of the fixed Gaussian bandwidth used by IGGCN. The current phase evaluates fixed `sigma` values only; it does not implement learnable or state-conditioned bandwidths.

## Project layout

- `research/iggcn/`: model implementation and reusable experiment code.
- `scripts/`: executable training, evaluation, analysis, and plotting entry points.
- `configs/`: versioned experiment configurations.
- `results/iggcn_sigma_diagnostic/`: metrics, per-sample tables, plots, and local checkpoints.
- `reports/iggcn_sigma_diagnostic/`: written experiment report.

Programs, result tables, and plots are kept in separate directories. Raw datasets and derived dataset caches are not stored in Git.

## Dataset location

Raw ETH/UCY files and manifests are stored on the data disk, under:

`/media/lrj/54926A1D926A0438/datasets/iggcn_eth_ucy/`

The files are under `raw/`. Their pinned source revision, URLs, row counts, and SHA-256 checksums are recorded in the external `dataset_manifest.json`. Raw trajectories remain outside Git.

## Paper source and implementation

The model is reimplemented from the user-provided IGGCN article PDF because no author IGGCN implementation was found. The paper gives the main graph equations and training protocol, but leaves several tensor, TCN, and checkpoint details unspecified. Those choices are listed in `reports/iggcn_sigma_diagnostic/SOURCE_AUDIT.md` and the experiment YAML.

The active branch is `exp/iggcn_sigma_diagnostic`. `scripts/train_eval_iggcn.py` runs fixed-sigma leave-one-scene-out folds; it accepts only sigma values 1 through 5. It writes metrics and per-target errors into `results/iggcn_sigma_diagnostic/` and keeps model checkpoints local under the ignored `checkpoints/` directory.
