"""Audit the existing sigma=3 checkpoints without retraining them.

Writes protocol, constant-velocity, graph-composition, Gaussian-kernel, and
gradient diagnostics under results/iggcn_baseline_repair/pre_fix_audit/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.iggcn.graph_data import (  # noqa: E402
    SceneGraphDataset,
    collate_scene_graphs,
    load_scene_graph_windows,
)
from research.iggcn.interaction import gaussian_similarity  # noqa: E402
from research.iggcn.model import IGGCN, bivariate_gaussian_nll  # noqa: E402
from scripts.train_eval_iggcn import SCENE_COMPONENTS, set_seed  # noqa: E402


SCENES = tuple(SCENE_COMPONENTS)
K_SAMPLES = 20
DIAGNOSTIC_GRAPHS_PER_SCENE = 64
BASE_RESULTS = PROJECT_ROOT / "results/iggcn_sigma_diagnostic"
OUT_ROOT = PROJECT_ROOT / "results/iggcn_baseline_repair/pre_fix_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_prior_sample_metrics(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    metrics = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            metrics[(row["run_id"], row["sample_id"])] = (
                float(row["ADE"]),
                float(row["FDE"]),
            )
    return metrics


def summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    quantiles = np.quantile(array, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1])
    return {
        "count": int(array.size),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def add_stat_parts(store: dict, name: str, values: torch.Tensor) -> None:
    clean = values.detach().float().reshape(-1)
    clean = clean[torch.isfinite(clean)]
    if clean.numel():
        store.setdefault(name, []).append(clean.cpu().numpy())


def summarize_parts(store: dict) -> dict:
    return {name: summary(np.concatenate(parts)) for name, parts in store.items()}


def load_windows(config: dict, scene: str):
    dataset_cfg = config["dataset"]
    root = Path(dataset_cfg["raw_root"])
    component_paths = [
        (scene, component, root / filename)
        for component, filename in SCENE_COMPONENTS[scene].items()
    ]
    return load_scene_graph_windows(
        component_paths,
        observed_length=int(dataset_cfg["observed_frames"]),
        predicted_length=int(dataset_cfg["predicted_frames"]),
        raw_frame_stride=int(dataset_cfg["raw_frame_stride"]),
        sample_period_seconds=float(dataset_cfg["sample_period_seconds"]),
        density_radius_meters=float(
            config["state_features"]["density"]["radius_meters"]
        ),
    )


def model_for_checkpoint(config: dict, checkpoint_path: Path, device: torch.device):
    training = config["training"]
    model = IGGCN(
        sigma=3,
        hidden_dimension=int(training["hidden_dimension"]),
        tcn_channels=int(training["tcn_channels"]),
        deformable_layers=int(training["deformable_convolution_layers"]),
        tcn_layers=int(training["tcn_layers"]),
        prediction_steps=int(config["dataset"]["predicted_frames"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def graph_composition_audit(config: dict) -> list[dict]:
    """Compare current complete target nodes with all peds seen in each obs clip."""
    raw_root = Path(config["dataset"]["raw_root"])
    stride = int(config["dataset"]["raw_frame_stride"])
    observed_steps = int(config["dataset"]["observed_frames"])
    results = []
    for scene in SCENES:
        windows = load_windows(config, scene)
        by_component_start: dict[tuple[str, int], list] = defaultdict(list)
        for window in windows:
            by_component_start[(window.component, window.start_frame)].append(window)

        raw_by_component: dict[str, dict[int, set[float]]] = {}
        for component, filename in SCENE_COMPONENTS[scene].items():
            rows = np.loadtxt(raw_root / filename, dtype=np.float64, ndmin=2)
            people_by_frame: dict[int, set[float]] = defaultdict(set)
            for frame, pedestrian_id, _x, _y in rows:
                people_by_frame[int(frame)].add(float(pedestrian_id))
            raw_by_component[component] = people_by_frame

        graph_sizes = []
        raw_people_per_frame = []
        context_people_per_frame = []
        context_unique_per_graph = []
        target_graph_neighbors = []
        raw_neighbors_per_target_frame = []
        omitted_neighbor_counts = []
        single_node_graphs = 0
        observed_person_frames = 0
        omitted_person_frames = 0
        for (component, start_frame), grouped_targets in by_component_start.items():
            target_ids = {
                float(pedestrian_id.rsplit(":", 1)[-1])
                for graph_window in grouped_targets
                for pedestrian_id in graph_window.pedestrian_ids
            }
            graph_sizes.append(len(target_ids))
            single_node_graphs += len(target_ids) == 1
            observed_frames = [start_frame + step * stride for step in range(observed_steps)]
            all_observed_people = set()
            people_by_frame = raw_by_component[component]
            for frame in observed_frames:
                present = people_by_frame.get(frame, set())
                all_observed_people.update(present)
                raw_people_per_frame.append(len(present))
                context = present - target_ids
                context_people_per_frame.append(len(context))
                observed_person_frames += len(present)
                omitted_person_frames += len(context)
                omitted_neighbor_counts.extend(len(context - {target_id}) for target_id in target_ids)
                raw_neighbors_per_target_frame.extend(
                    len(present - {target_id}) for target_id in target_ids
                )
            context_unique_per_graph.append(len(all_observed_people - target_ids))
            target_graph_neighbors.extend([len(target_ids) - 1] * len(target_ids) * observed_steps)

        results.append(
            {
                "scene": scene,
                "test_graph_windows": len(graph_sizes),
                "mean_complete_target_nodes_per_graph": float(np.mean(graph_sizes)),
                "median_complete_target_nodes_per_graph": float(np.median(graph_sizes)),
                "mean_raw_pedestrians_per_observed_frame": float(np.mean(raw_people_per_frame)),
                "mean_observed_context_pedestrians_omitted_per_frame": float(
                    np.mean(context_people_per_frame)
                ),
                "mean_unique_context_pedestrians_omitted_per_graph": float(
                    np.mean(context_unique_per_graph)
                ),
                "graphs_with_one_valid_node": int(single_node_graphs),
                "graphs_with_one_valid_node_fraction": float(single_node_graphs / len(graph_sizes)),
                "mean_graph_candidate_neighbors_per_target_frame": float(
                    np.mean(target_graph_neighbors)
                ),
                "mean_raw_same_frame_neighbors_per_target_frame": float(
                    np.mean(raw_neighbors_per_target_frame)
                ),
                "mean_raw_neighbors_missing_from_graph_per_target_frame": float(
                    np.mean(omitted_neighbor_counts)
                ),
                "observed_pedestrian_frames": int(observed_person_frames),
                "observed_pedestrian_frames_omitted_from_graph": int(omitted_person_frames),
                "omitted_observed_pedestrian_frame_fraction": float(
                    omitted_person_frames / observed_person_frames
                ),
                "construction_note": (
                    "Current graph nodes are complete 8+12 targets. Raw observed agents "
                    "without a complete future are counted as context-only candidates; "
                    "the paper does not specify whether these incomplete-future agents "
                    "belong in the graph."
                ),
            }
        )
    return results


def constant_velocity_predictions(observed_xy: torch.Tensor) -> torch.Tensor:
    """OLS velocity over the 8 observations, extrapolated 12 frames."""
    observed = observed_xy.detach().cpu().numpy()  # [B,T,N,2]
    times = np.arange(observed.shape[1], dtype=np.float64)
    centered_time = times - times.mean()
    denominator = float(np.square(centered_time).sum())
    centered_positions = observed - observed.mean(axis=1, keepdims=True)
    velocity_per_frame = np.einsum(
        "t,btnc->bnc", centered_time, centered_positions
    ) / denominator
    last = observed[:, -1]
    steps = np.arange(1, 13, dtype=np.float64)[None, None, :, None]
    return torch.from_numpy(last[:, :, None] + velocity_per_frame[:, :, None] * steps)


def sample_gaussian_paths(
    parameters: torch.Tensor,
    last_observed: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    mu = parameters[..., :2]
    sigma_x = parameters[..., 2].clamp(-7, 7).exp()
    sigma_y = parameters[..., 3].clamp(-7, 7).exp()
    rho = parameters[..., 4].tanh().clamp(-0.999, 0.999)
    shape = (count, *sigma_x.shape)
    z_x = torch.randn(shape, dtype=mu.dtype, device=mu.device, generator=generator)
    z_y = torch.randn(shape, dtype=mu.dtype, device=mu.device, generator=generator)
    dx = sigma_x.unsqueeze(0) * z_x
    dy = sigma_y.unsqueeze(0) * (
        rho.unsqueeze(0) * z_x
        + (1 - rho.square()).clamp_min(1e-6).sqrt().unsqueeze(0) * z_y
    )
    samples = torch.stack((dx, dy), dim=-1) + mu.unsqueeze(0)
    return samples + last_observed[None, :, :, None, :]


def gaussian_and_gradient_diagnostics(
    model: IGGCN,
    batch: dict,
    device: torch.device,
    scene: str,
    graph_limit: int,
) -> dict:
    observed = batch["observed_xy"].to(device)
    future = batch["future_xy"].to(device)
    pedestrian_mask = batch["pedestrian_mask"].to(device)
    actual_batch, observed_steps, pedestrians, _ = observed.shape
    take = min(actual_batch, graph_limit)
    model.zero_grad(set_to_none=True)
    parameters, diagnostics = model(
        observed, pedestrian_mask, return_diagnostics=True
    )

    valid = pedestrian_mask[:take]
    graph_count = int(valid.sum().item())
    pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
    n = pedestrians
    spatial_qk = diagnostics["spatial_qk"][:take]
    spatial_maps = diagnostics["spatial_maps"][:take]
    temporal_qk = diagnostics["temporal_qk"][:take]
    temporal_maps = diagnostics["temporal_maps"][:take]
    spatial_offdiag = pair_valid.expand(-1, observed_steps, -1, -1) & ~torch.eye(
        n, device=device, dtype=torch.bool
    )[None, None]
    temporal_valid = valid[:, :, None, None].expand(-1, -1, observed_steps, observed_steps)
    temporal_offdiag = temporal_valid & ~torch.eye(
        observed_steps, device=device, dtype=torch.bool
    )[None, None]

    spatial_similarity = gaussian_similarity(spatial_maps.unsqueeze(-1), model.sigma)
    temporal_similarity = gaussian_similarity(temporal_maps.unsqueeze(-1), model.sigma)
    current_mask = torch.tril(
        torch.ones(observed_steps, observed_steps, device=device, dtype=torch.bool), diagonal=0
    )
    temporal_effective_offdiag = temporal_offdiag & current_mask[None, None]

    logs: dict[str, list[np.ndarray]] = {}
    add_stat_parts(logs, "spatial_qk_valid", spatial_qk[pair_valid.expand_as(spatial_qk)])
    add_stat_parts(logs, "temporal_qk_valid", temporal_qk[temporal_valid])
    add_stat_parts(logs, "spatial_dcnn_output_valid", spatial_maps[pair_valid.expand_as(spatial_maps)])
    add_stat_parts(logs, "temporal_dcnn_output_valid", temporal_maps[temporal_valid])
    add_stat_parts(logs, "spatial_gaussian_all_valid", spatial_similarity[pair_valid.expand_as(spatial_similarity)])
    add_stat_parts(logs, "temporal_gaussian_all_valid", temporal_similarity[temporal_valid])
    add_stat_parts(logs, "spatial_gaussian_non_diagonal", spatial_similarity[spatial_offdiag])
    add_stat_parts(logs, "temporal_gaussian_non_diagonal_raw", temporal_similarity[temporal_offdiag])
    add_stat_parts(logs, "temporal_effective_non_diagonal_weight", temporal_similarity[temporal_effective_offdiag])
    summaries = summarize_parts(logs)
    summaries["effective_weight_saturation"] = {
        "spatial_fraction_below_1e-4": float((spatial_similarity[spatial_offdiag] < 1e-4).float().mean().item()),
        "spatial_fraction_at_least_0.99": float((spatial_similarity[spatial_offdiag] >= 0.99).float().mean().item()),
        "temporal_fraction_below_1e-4": float((temporal_similarity[temporal_effective_offdiag] < 1e-4).float().mean().item()),
        "temporal_fraction_at_least_0.99": float((temporal_similarity[temporal_effective_offdiag] >= 0.99).float().mean().item()),
    }

    target_offsets = future.transpose(1, 2) - observed[:, -1][:, :, None]
    per_agent_nll = bivariate_gaussian_nll(parameters, target_offsets)
    loss = (per_agent_nll * pedestrian_mask).sum() / pedestrian_mask.sum().clamp_min(1)
    loss.backward()
    grad_entries = {}
    total_parameters = 0
    finite_parameters = 0
    nonzero_parameters = 0
    squared_norm = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            grad_entries[name] = {"has_gradient": False}
            continue
        grad = parameter.grad.detach().float()
        finite = bool(torch.isfinite(grad).all().item())
        norm = float(torch.linalg.vector_norm(grad).item()) if finite else None
        nonzero = bool(torch.count_nonzero(grad).item() > 0)
        grad_entries[name] = {"has_gradient": True, "finite": finite, "l2_norm": norm, "nonzero": nonzero}
        total_parameters += 1
        finite_parameters += int(finite)
        nonzero_parameters += int(nonzero)
        if finite and norm is not None:
            squared_norm += norm**2
    model.zero_grad(set_to_none=True)
    return {
        "scene": scene,
        "diagnostic_graphs": take,
        "valid_target_nodes": graph_count,
        "training_nll_on_test_probe": float(loss.detach().item()),
        "gaussian_statistics": summaries,
        "gradient_health": {
            "parameter_tensors_with_grad": total_parameters,
            "finite_gradient_fraction": finite_parameters / max(total_parameters, 1),
            "nonzero_gradient_fraction": nonzero_parameters / max(total_parameters, 1),
            "global_l2_norm_across_parameter_tensors": math.sqrt(squared_norm),
            "by_parameter": grad_entries,
        },
    }


def audit_fold(
    scene: str,
    windows: list,
    config: dict,
    prior_metrics: dict,
    device: torch.device,
    sampling_seed: int,
) -> tuple[dict, dict, dict]:
    run_id = f"sigma3_{scene.lower()}_seed42"
    checkpoint_path = BASE_RESULTS / "checkpoints" / f"{run_id}.pt"
    model, checkpoint = model_for_checkpoint(config, checkpoint_path, device)
    dataset = SceneGraphDataset(windows, list(range(len(windows))))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_scene_graphs,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(sampling_seed)
    k_ade_sums = torch.zeros(K_SAMPLES, dtype=torch.float64)
    k_fde_sums = torch.zeros(K_SAMPLES, dtype=torch.float64)
    target_count = 0
    mean_ades, mean_fdes = [], []
    cv_ades, cv_fdes = [], []
    old_ade_abs, old_fde_abs = [], []
    sample_rows = []
    first_batch = None
    for batch in loader:
        if first_batch is None:
            first_batch = batch
        observed = batch["observed_xy"].to(device)
        future = batch["future_xy"].to(device).transpose(1, 2)
        mask = batch["pedestrian_mask"].to(device)
        with torch.no_grad():
            output = model(observed, mask)
            mean_position = model.predict_positions(output, observed[:, -1])
            distances = torch.linalg.vector_norm(mean_position - future, dim=-1)
            sample_paths = sample_gaussian_paths(
                output, observed[:, -1], K_SAMPLES, rng
            )
            sampled_distances = torch.linalg.vector_norm(
                sample_paths - future.unsqueeze(0), dim=-1
            )
        cv_prediction = constant_velocity_predictions(observed.cpu()).to(future.dtype)
        cv_distance = torch.linalg.vector_norm(cv_prediction.to(device) - future, dim=-1)

        for b_index, metadata in enumerate(batch["metadata"]):
            for pedestrian_index, pedestrian_id in enumerate(metadata.pedestrian_ids):
                mean_ade = float(distances[b_index, pedestrian_index].mean().item())
                mean_fde = float(distances[b_index, pedestrian_index, -1].item())
                cv_ade = float(cv_distance[b_index, pedestrian_index].mean().item())
                cv_fde = float(cv_distance[b_index, pedestrian_index, -1].item())
                key = (run_id, f"{metadata.scene}:{pedestrian_id}:{metadata.start_frame}")
                previous = prior_metrics.get(key)
                if previous:
                    old_ade_abs.append(abs(mean_ade - previous[0]))
                    old_fde_abs.append(abs(mean_fde - previous[1]))
                mean_ades.append(mean_ade)
                mean_fdes.append(mean_fde)
                cv_ades.append(cv_ade)
                cv_fdes.append(cv_fde)
                sample_rows.append(
                    {
                        "scene": scene,
                        "sample_id": key[1],
                        "mean_ADE": mean_ade,
                        "mean_FDE": mean_fde,
                        "constant_velocity_ADE": cv_ade,
                        "constant_velocity_FDE": cv_fde,
                    }
                )
        valid = mask[None].to(sampled_distances.dtype)
        counts = mask.sum().to(torch.float64)
        per_k_ade = (sampled_distances.mean(dim=-1) * valid).sum(dim=(1, 2)) / counts
        per_k_fde = (sampled_distances[..., -1] * valid).sum(dim=(1, 2)) / counts
        k_ade_sums += per_k_ade.cpu().double() * counts.cpu()
        k_fde_sums += per_k_fde.cpu().double() * counts.cpu()
        target_count += int(mask.sum().item())

    if first_batch is None:
        raise ValueError(f"No test windows found for {scene}")
    k_ades = (k_ade_sums / target_count).numpy()
    k_fdes = (k_fde_sums / target_count).numpy()
    fold = {
        "scene": scene,
        "run_id": run_id,
        "targets": target_count,
        "mean_prediction_ADE": float(np.mean(mean_ades)),
        "mean_prediction_FDE": float(np.mean(mean_fdes)),
        "sampling_protocol": f"{K_SAMPLES} independent per-timestep bivariate Gaussian paths; fixed seed {sampling_seed}; average over samples; no best-of-K selection",
        "sampled_ADE_mean_over_K": float(k_ades.mean()),
        "sampled_ADE_std_across_K": float(k_ades.std(ddof=0)),
        "sampled_ADE_p05_across_K": float(np.quantile(k_ades, 0.05)),
        "sampled_ADE_p95_across_K": float(np.quantile(k_ades, 0.95)),
        "sampled_FDE_mean_over_K": float(k_fdes.mean()),
        "sampled_FDE_std_across_K": float(k_fdes.std(ddof=0)),
        "sampled_FDE_p05_across_K": float(np.quantile(k_fdes, 0.05)),
        "sampled_FDE_p95_across_K": float(np.quantile(k_fdes, 0.95)),
        "constant_velocity_ADE": float(np.mean(cv_ades)),
        "constant_velocity_FDE": float(np.mean(cv_fdes)),
        "previous_csv_max_abs_ADE_difference": float(max(old_ade_abs)) if old_ade_abs else None,
        "previous_csv_max_abs_FDE_difference": float(max(old_fde_abs)) if old_fde_abs else None,
        "paper_sampling_note": "The paper defines a bivariate Gaussian NLL and ADE/FDE, but does not state Gaussian sampling or best-of-K. Sampled metrics are diagnostic only; mean metrics are the comparison protocol used here.",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("best_epoch", -1)),
    }
    gauss = gaussian_and_gradient_diagnostics(
        model,
        first_batch,
        device,
        scene,
        DIAGNOSTIC_GRAPHS_PER_SCENE,
    )
    return fold, gauss, {"sample_rows": sample_rows, "sampled_ade_by_draw": k_ades.tolist(), "sampled_fde_by_draw": k_fdes.tolist()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/iggcn_sigma_diagnostic.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty diagnostic directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_metrics = read_prior_sample_metrics(BASE_RESULTS / "per_sample_errors.csv")
    graph_rows = graph_composition_audit(config)
    fold_rows = []
    gaussian_rows = []
    all_sample_rows = []
    sample_draws = {}
    for scene in SCENES:
        print(f"[audit] scene={scene} device={device}", flush=True)
        windows = load_windows(config, scene)
        fold, gaussian, draws = audit_fold(
            scene, windows, config, prior_metrics, device, args.seed
        )
        fold_rows.append(fold)
        gaussian_rows.append(gaussian)
        all_sample_rows.extend(draws["sample_rows"])
        sample_draws[scene] = {
            "ADE_per_draw": draws["sampled_ade_by_draw"],
            "FDE_per_draw": draws["sampled_fde_by_draw"],
        }
        print(
            f"[audit] {scene} mean={fold['mean_prediction_ADE']:.4f}/{fold['mean_prediction_FDE']:.4f} "
            f"CV={fold['constant_velocity_ADE']:.4f}/{fold['constant_velocity_FDE']:.4f} "
            f"sampled={fold['sampled_ADE_mean_over_K']:.4f}/{fold['sampled_FDE_mean_over_K']:.4f}",
            flush=True,
        )

    fold_rows.append(
        {
            "scene": "UNWEIGHTED_SCENE_MEAN",
            "targets": int(sum(row["targets"] for row in fold_rows)),
            "mean_prediction_ADE": float(np.mean([row["mean_prediction_ADE"] for row in fold_rows])),
            "mean_prediction_FDE": float(np.mean([row["mean_prediction_FDE"] for row in fold_rows])),
            "sampling_protocol": "Per-scene sampled metrics are averaged with equal scene weight.",
            "sampled_ADE_mean_over_K": float(np.mean([row["sampled_ADE_mean_over_K"] for row in fold_rows])),
            "sampled_FDE_mean_over_K": float(np.mean([row["sampled_FDE_mean_over_K"] for row in fold_rows])),
            "constant_velocity_ADE": float(np.mean([row["constant_velocity_ADE"] for row in fold_rows])),
            "constant_velocity_FDE": float(np.mean([row["constant_velocity_FDE"] for row in fold_rows])),
        }
    )
    write_csv(args.output / "protocol_and_constant_velocity.csv", fold_rows)
    write_csv(args.output / "constant_velocity_per_target.csv", all_sample_rows)
    write_csv(args.output / "social_graph_composition.csv", graph_rows)
    (args.output / "gaussian_and_gradient_diagnostics.json").write_text(
        json.dumps(gaussian_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "sampling_draw_metrics.json").write_text(
        json.dumps(sample_draws, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "seed": args.seed,
        "device": str(device),
        "sigma": 3,
        "observed_frames": int(config["dataset"]["observed_frames"]),
        "predicted_frames": int(config["dataset"]["predicted_frames"]),
        "raw_frame_stride": int(config["dataset"]["raw_frame_stride"]),
        "sample_period_seconds": float(config["dataset"]["sample_period_seconds"]),
        "sampling_count": K_SAMPLES,
        "sampling_seed": args.seed,
        "diagnostic_graphs_per_scene": DIAGNOSTIC_GRAPHS_PER_SCENE,
        "created_from_existing_checkpoints": True,
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "model_source_sha256": sha256(PROJECT_ROOT / "research/iggcn/model.py"),
        "git_dirty_at_audit": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        ),
        "output_directory": str(args.output),
    }
    (args.output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
