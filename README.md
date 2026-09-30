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

Store ETH/UCY files on the data disk, under:

`/media/lrj/54926A1D926A0438/datasets/iggcn_eth_ucy/`

The exact raw-data subdirectory and checksums will be recorded in the experiment configuration and manifest after the dataset source is verified.
