# Pre-run environment record

Recorded before any IGGCN model training, on 2026-10-01 (Asia/Shanghai).

## Repository state

- Requested repository: `https://github.com/duchangchen11/IGGCN_NEW`
- Repository was public but empty when inspected; it had no `main` commit or files.
- The local `main` baseline was initialized with the project layout and data-location instructions: `bf07c1e4f58c9a54d87bdf33c9a2b2714179f47b`.
- Diagnostic branch: `exp/iggcn_sigma_diagnostic`, created from that baseline.
- The separate JAAD repository `/home/lrj/ped_intent_project` was clean on `main` at `cb53ed12693f760bfd08637ae4beabb02bad0055`; no JAAD source files were changed.

## Runtime

The system interpreter is Python 3.8.10 and does not have PyTorch installed. The usable Conda base runtime is:

- Python: 3.10.9
- PyTorch: 2.0.1+cu117
- PyTorch CUDA runtime: 11.7
- CUDA available to PyTorch: yes
- GPU: NVIDIA GeForce RTX 3080, 10 GB
- NVIDIA driver: 570.133.07
- CUDA version reported by `nvidia-smi`: 12.8
- Installed experiment packages: torchvision 0.15.2+cu117, NumPy 1.23.5, pandas 1.5.3, matplotlib 3.7.0, seaborn 0.12.2, SciPy 1.10.0, PyYAML 6.0, pytest 7.1.2.

The system-level `pytest` plugin autoload discovers an incompatible ROS plugin. Run the requested focused checks with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` under this Conda interpreter.

The random seed controls Python, NumPy, PyTorch initialization, and training-sample order. cuDNN deterministic algorithms are disabled (`benchmark=False`); on this RTX 3080, forcing deterministic cuDNN made one representative model update about 15 seconds versus about 0.05 seconds with the default kernel path. This trades bitwise repeatability for feasible runtime; the kernel policy is recorded in each run manifest and is identical across sigma runs.

## Data disk

`/media/lrj/54926A1D926A0438` is mounted with about 1.8 TB available. Dataset files are stored under `/media/lrj/54926A1D926A0438/datasets/iggcn_eth_ucy/raw/`, outside Git.
