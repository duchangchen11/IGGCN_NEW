"""Train and evaluate one or more fixed-sigma ETH/UCY IGGCN folds."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.iggcn.graph_data import (
    SceneGraphDataset,
    collate_scene_graphs,
    chronological_train_validation_split,
    load_scene_graph_windows,
)
from research.iggcn.model import IGGCN, bivariate_gaussian_nll


DEFAULT_CONFIG = PROJECT_ROOT / "configs/iggcn_sigma_diagnostic.yaml"
SCENE_COMPONENTS = {
    "ETH": {"biwi_eth": "biwi_eth.txt"},
    "HOTEL": {"biwi_hotel": "biwi_hotel.txt"},
    "UNIV": {
        "students001": "students001.txt",
        "students003": "students003.txt",
    },
    "ZARA1": {"crowds_zara01": "crowds_zara01.txt"},
    "ZARA2": {"crowds_zara02": "crowds_zara02.txt"},
}
SCENES = tuple(SCENE_COMPONENTS)
STATE_NAMES = ("v_last", "v_mean", "a_mean", "turning", "density")
METRIC_FIELDS = [
    "run_id",
    "sigma",
    "seed",
    "test_scene",
    "ADE",
    "FDE",
    "best_epoch",
    "best_validation_ADE",
    "test_targets",
]
SAMPLE_FIELDS = [
    "run_id",
    "sample_id",
    "scene",
    "component",
    "target_pedestrian_id",
    "start_frame",
    "sigma",
    "ADE",
    "FDE",
    *STATE_NAMES,
    "observed_trajectory",
    "ground_truth_future",
    "predicted_future_mean",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"NaN or Inf detected in {name}")


def load_component_windows(config: dict, data_root: Path, selected_scenes: list[str]):
    data_cfg = config["dataset"]
    component_paths = [
        (scene, component, data_root / filename)
        for scene in selected_scenes
        for component, filename in SCENE_COMPONENTS[scene].items()
    ]
    return load_scene_graph_windows(
        component_paths,
        observed_length=int(data_cfg["observed_frames"]),
        predicted_length=int(data_cfg["predicted_frames"]),
        raw_frame_stride=int(data_cfg["raw_frame_stride"]),
        sample_period_seconds=float(data_cfg["sample_period_seconds"]),
        density_radius_meters=float(
            config["state_features"]["density"]["radius_meters"]
        ),
    )


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, device: torch.device):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_scene_graphs,
        generator=generator,
    )


def batch_to_device(batch: dict, device: torch.device):
    return (
        batch["observed_xy"].to(device, non_blocking=True),
        batch["future_xy"].to(device, non_blocking=True),
        batch["pedestrian_mask"].to(device, non_blocking=True),
    )


def evaluate_validation(model: IGGCN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_distance = 0.0
    total_points = 0
    with torch.no_grad():
        for batch in loader:
            observed, future, mask = batch_to_device(batch, device)
            parameters = model(observed, mask)
            predicted = model.predict_positions(parameters, observed[:, -1])
            finite_or_raise("validation predictions", predicted)
            distances = torch.linalg.vector_norm(
                predicted - future.transpose(1, 2), dim=-1
            )
            total_distance += float((distances * mask[:, :, None]).sum().item())
            total_points += int(mask.sum().item()) * future.shape[1]
    if total_points == 0:
        raise ValueError("validation split has no target pedestrians")
    return total_distance / total_points


def train_one_epoch(
    model: IGGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    batches = 0
    for batch in loader:
        observed, future, mask = batch_to_device(batch, device)
        parameters = model(observed, mask)
        target_offsets = future.transpose(1, 2) - observed[:, -1][:, :, None, :]
        per_agent_nll = bivariate_gaussian_nll(parameters, target_offsets)
        finite_or_raise("training NLL", per_agent_nll)
        loss = (per_agent_nll * mask).sum() / mask.sum().clamp_min(1)
        finite_or_raise("training loss", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                finite_or_raise(f"gradient {name}", parameter.grad)
        optimizer.step()
        loss_sum += float(loss.detach().item())
        batches += 1
    if batches == 0:
        raise ValueError("training split has no graph windows")
    return loss_sum / batches


def format_trajectory(values: np.ndarray) -> str:
    return ";".join(
        f"{point[0]:.7g}:{point[1]:.7g}" for point in np.asarray(values)
    )


def test_model(
    model: IGGCN,
    loader: DataLoader,
    device: torch.device,
    sigma: int,
    seed: int,
    run_id: str,
):
    model.eval()
    sample_rows = []
    with torch.no_grad():
        for batch in loader:
            observed, future, mask = batch_to_device(batch, device)
            parameters = model(observed, mask)
            predicted = model.predict_positions(parameters, observed[:, -1])
            finite_or_raise("test predictions", predicted)
            distances = torch.linalg.vector_norm(
                predicted - future.transpose(1, 2), dim=-1
            )
            distances_np = distances.cpu().numpy()
            prediction_np = predicted.cpu().numpy()
            ground_truth_np = future.transpose(1, 2).cpu().numpy()
            observed_np = observed.transpose(1, 2).cpu().numpy()
            for batch_index, metadata in enumerate(batch["metadata"]):
                for pedestrian_index, pedestrian_id in enumerate(
                    metadata.pedestrian_ids
                ):
                    ade = float(distances_np[batch_index, pedestrian_index].mean())
                    fde = float(distances_np[batch_index, pedestrian_index, -1])
                    row = {
                        "run_id": run_id,
                        "sample_id": (
                            f"{metadata.scene}:{pedestrian_id}:{metadata.start_frame}"
                        ),
                        "scene": metadata.scene,
                        "component": metadata.component,
                        "target_pedestrian_id": pedestrian_id,
                        "start_frame": metadata.start_frame,
                        "sigma": sigma,
                        "ADE": ade,
                        "FDE": fde,
                        "observed_trajectory": format_trajectory(
                            observed_np[batch_index, pedestrian_index]
                        ),
                        "ground_truth_future": format_trajectory(
                            ground_truth_np[batch_index, pedestrian_index]
                        ),
                        "predicted_future_mean": format_trajectory(
                            prediction_np[batch_index, pedestrian_index]
                        ),
                    }
                    for name in STATE_NAMES:
                        row[name] = float(
                            metadata.state_features[name][pedestrian_index]
                        )
                    sample_rows.append(row)
    if not sample_rows:
        raise ValueError("test split has no target pedestrians")
    return sample_rows


def append_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def train_fold(
    config: dict,
    config_path: Path,
    sigma: int,
    test_scene: str,
    seed: int,
    max_epochs: int | None,
) -> dict:
    training_cfg = config["training"]
    data_cfg = config["dataset"]
    output_root = PROJECT_ROOT / config["outputs"]["results_root"]
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    config_snapshots = output_root / "configs"
    config_snapshots.mkdir(parents=True, exist_ok=True)
    run_id = f"sigma{sigma}_{test_scene.lower()}_seed{seed}"
    config_snapshot = config_snapshots / f"{run_id}.yaml"
    if not config_snapshot.exists():
        shutil.copy2(config_path, config_snapshot)
    manifest_path = output_root / "run_manifest.json"
    completed = read_manifest(manifest_path)
    if any(run["run_id"] == run_id for run in completed):
        raise RuntimeError(f"run already recorded: {run_id}; remove it explicitly to rerun")

    start_time = utc_now()
    start_clock = time.monotonic()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_scenes = [scene for scene in SCENES if scene != test_scene]
    training_windows = load_component_windows(config, Path(data_cfg["raw_root"]), training_scenes)
    test_windows = load_component_windows(config, Path(data_cfg["raw_root"]), [test_scene])
    train_indices, validation_indices = chronological_train_validation_split(
        training_windows,
        validation_fraction=0.1,
        purge_windows=19,
    )
    train_dataset = SceneGraphDataset(training_windows, train_indices)
    validation_dataset = SceneGraphDataset(training_windows, validation_indices)
    test_dataset = SceneGraphDataset(test_windows, list(range(len(test_windows))))
    batch_size = int(training_cfg["batch_size"])
    train_loader = make_loader(train_dataset, batch_size, True, seed, device)
    validation_loader = make_loader(
        validation_dataset, batch_size, False, seed, device
    )
    test_loader = make_loader(test_dataset, batch_size, False, seed, device)

    model = IGGCN(
        sigma=sigma,
        hidden_dimension=int(training_cfg["hidden_dimension"]),
        tcn_channels=int(training_cfg["tcn_channels"]),
        deformable_layers=int(training_cfg["deformable_convolution_layers"]),
        tcn_layers=int(training_cfg["tcn_layers"]),
        prediction_steps=int(data_cfg["predicted_frames"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_cfg["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training_cfg["learning_rate_decay_every_epochs"]),
        gamma=float(training_cfg["learning_rate_decay_factor"]),
    )
    epochs = max_epochs or int(training_cfg["epochs"])
    checkpoint_path = checkpoints / f"{run_id}.pt"
    best_validation_ade = float("inf")
    best_epoch = 0

    print(
        f"[{run_id}] device={device} graphs(train/val/test)="
        f"{len(train_dataset)}/{len(validation_dataset)}/{len(test_dataset)} "
        f"parameters={parameter_count} epochs={epochs}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        train_nll = train_one_epoch(model, train_loader, optimizer, device)
        validation_ade = evaluate_validation(model, validation_loader, device)
        scheduler.step()
        if validation_ade < best_validation_ade:
            best_validation_ade = validation_ade
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "sigma": sigma,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_ADE": best_validation_ade,
                    "parameter_count": parameter_count,
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{run_id}] epoch={epoch:03d}/{epochs} "
                f"nll={train_nll:.5f} val_ADE={validation_ade:.5f} "
                f"best={best_validation_ade:.5f}@{best_epoch} "
                f"elapsed_min={(time.monotonic() - start_clock) / 60:.1f}",
                flush=True,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    sample_rows = test_model(model, test_loader, device, sigma, seed, run_id)
    sample_path = output_root / "per_sample_errors.csv"
    append_csv(sample_path, SAMPLE_FIELDS, sample_rows)
    test_scene_rows = [row for row in sample_rows if row["scene"] == test_scene]
    metrics = {
        "run_id": run_id,
        "sigma": sigma,
        "seed": seed,
        "test_scene": test_scene,
        "ADE": float(np.mean([row["ADE"] for row in test_scene_rows])),
        "FDE": float(np.mean([row["FDE"] for row in test_scene_rows])),
        "best_epoch": best_epoch,
        "best_validation_ADE": best_validation_ade,
        "test_targets": len(test_scene_rows),
    }
    metrics_path = output_root / "sigma_sweep_seed42.csv"
    append_csv(metrics_path, METRIC_FIELDS, [metrics])
    if sigma == int(config["fixed_sigma_sweep"]["baseline_sigma"]):
        append_csv(output_root / "baseline_reproduction.csv", METRIC_FIELDS, [metrics])

    run_record = {
        **metrics,
        "dataset": data_cfg["name"],
        "training_scenes": training_scenes,
        "graph_windows_train": len(train_dataset),
        "graph_windows_validation": len(validation_dataset),
        "graph_windows_test": len(test_dataset),
        "epoch": epochs,
        "epochs": epochs,
        "parameter_count": parameter_count,
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_snapshot),
        "git_commit": git_commit(),
        "git_dirty_at_run": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        ),
        "started_at_utc": start_time,
        "ended_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - start_clock,
        "device": str(device),
        "notes": "Paper-based implementation; assumptions are listed in the versioned YAML and SOURCE_AUDIT.md.",
    }
    completed.append(run_record)
    manifest_path.write_text(
        json.dumps(completed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[{run_id}] test ADE={metrics['ADE']:.5f} FDE={metrics['FDE']:.5f} "
        f"targets={metrics['test_targets']} best_epoch={best_epoch}",
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sigma", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--folds", default="all", help="all or comma-separated scene names")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, help="short smoke run; omit for configured 150 epochs")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    folds = list(SCENES) if args.folds == "all" else args.folds.split(",")
    unknown = sorted(set(folds) - set(SCENES))
    if unknown:
        raise ValueError(f"unknown scene(s): {unknown}")
    for scene in folds:
        train_fold(config, args.config, args.sigma, scene, args.seed, args.max_epochs)


if __name__ == "__main__":
    main()
