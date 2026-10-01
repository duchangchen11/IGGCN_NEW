"""Measure the legacy temporal-mask leakage on one saved HOTEL test graph."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.iggcn.graph_data import SceneGraphDataset, collate_scene_graphs, load_scene_graph_windows
from research.iggcn.interaction import gaussian_similarity
from scripts.train_eval_iggcn import SCENE_COMPONENTS

BASELINE_COMMIT = "124a06f7db2e4b6b61dba10f53cae580f3107446"
OUTPUT = PROJECT_ROOT / "results/iggcn_baseline_repair/pre_fix_audit/temporal_causality_probe.json"


def main() -> None:
    config_path = PROJECT_ROOT / "configs/iggcn_sigma_diagnostic.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:research/iggcn/model.py"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    legacy_path = Path("/tmp/iggcn_baseline_model_before_causal_fix.py")
    legacy_path.write_text(source, encoding="utf-8")
    module_name = "research.iggcn._baseline_model_before_causal_fix"
    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    scene = "HOTEL"
    dataset_cfg = config["dataset"]
    component_paths = [
        (
            scene,
            component,
            Path(dataset_cfg["raw_root"]) / filename,
        )
        for component, filename in SCENE_COMPONENTS[scene].items()
    ]
    windows = load_scene_graph_windows(
        component_paths,
        observed_length=int(dataset_cfg["observed_frames"]),
        predicted_length=int(dataset_cfg["predicted_frames"]),
        raw_frame_stride=int(dataset_cfg["raw_frame_stride"]),
        sample_period_seconds=float(dataset_cfg["sample_period_seconds"]),
        density_radius_meters=float(config["state_features"]["density"]["radius_meters"]),
    )
    batch = next(
        iter(
            DataLoader(
                SceneGraphDataset(windows, list(range(len(windows)))),
                batch_size=1,
                shuffle=False,
                collate_fn=collate_scene_graphs,
            )
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.IGGCN(sigma=3).to(device)
    checkpoint = torch.load(
        PROJECT_ROOT / "results/iggcn_sigma_diagnostic/checkpoints/sigma3_hotel_seed42.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    observed = batch["observed_xy"].to(device)
    mask = batch["pedestrian_mask"].to(device)
    target_time = 3
    changed = observed.clone()
    torch.manual_seed(20261001)
    changed[:, target_time + 1 :] += torch.randn_like(changed[:, target_time + 1 :]) * 5
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.temporal_deformable.register_forward_hook(capture)
    with torch.no_grad():
        model(observed, mask)
        old_maps = captured.pop().reshape(
            observed.shape[0], observed.shape[2], observed.shape[1], observed.shape[1]
        )
        model(changed, mask)
        changed_maps = captured.pop().reshape_as(old_maps)
    handle.remove()

    old_upper = torch.triu(torch.ones(8, 8, device=device, dtype=torch.bool))
    historical_row = slice(0, target_time + 1)
    causal_map_diff = (old_maps[:, :, target_time, historical_row] - changed_maps[:, :, target_time, historical_row]).abs()
    old_sim = gaussian_similarity(old_maps.unsqueeze(-1), 3.0)
    allowed_future = old_upper[target_time, target_time + 1 :]
    result = {
        "baseline_source_commit": BASELINE_COMMIT,
        "scene": scene,
        "graph_start_frame": int(batch["metadata"][0].start_frame),
        "target_time_index": target_time,
        "qk_axes_from_baseline_source": "row=target time; column=source time; adjacency @ features aggregates columns",
        "upper_mask_row_source_indices": torch.where(old_upper[target_time])[0].cpu().tolist(),
        "future_sources_allowed_by_old_mask_at_target": torch.where(allowed_future)[0].add(target_time + 1).cpu().tolist(),
        "historical_map_max_abs_change_after_future_perturbation": float(causal_map_diff.max().item()),
        "historical_map_mean_abs_change_after_future_perturbation": float(causal_map_diff.mean().item()),
        "future_non_diagonal_edges_allowed_by_old_mask": int(allowed_future.sum().item()),
        "old_gaussian_weights_to_future_sources": old_sim[0, 0, target_time, target_time + 1 :].cpu().tolist(),
        "test": "Later observed positions were changed while retaining the same saved checkpoint. A causal temporal target row should be invariant to this perturbation.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
