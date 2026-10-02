# Baseline snapshot provenance

- Snapshot created from branch `fix/iggcn_baseline_reproduction`, HEAD `fb54893e750390e6ee343670391df8ed6d5829d3`; initial worktree status was clean and tracked `origin/fix/iggcn_baseline_reproduction`.
- The baseline score in `baseline_original.csv` is the pre-causal-mask σ=3 result (`0.7578096/1.4931050` equal-scene ADE/FDE). Its five checkpoints were trained under the original upper-triangular temporal graph and came from source commit `733a777695116e293fa91f3b0312bc3dfb645486`, as recorded in `original_run_manifest.json`.
- `configs/baseline_original.yaml` and `configs/original_folds/` preserve the actual baseline config and per-fold config snapshots.
- `checkpoint_snapshot/original_triu/` contains the five original fold checkpoints. `checkpoint_sha256.txt` records their hashes.
- `checkpoint_snapshot/repaired_tril/sigma3_hotel_seed42.pt` is the existing targeted causal-mask HOTEL checkpoint; its run settings are in `repaired_hotel_run_manifest.json`.
- `dataset_graph_statistics.csv` and `constant_velocity_baseline.csv` are copies of the prior raw-data audit outputs, not newly recomputed in this phase.
- The external volume is not mounted in this environment on 2026-10-02. No raw data has been copied into this repository.
