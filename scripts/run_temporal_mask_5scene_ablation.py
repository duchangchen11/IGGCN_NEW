"""Run the seed-42 ETH/UCY triu-vs-tril controlled ablation on raw data."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MASTER_CONFIG = ROOT / "configs/iggcn_temporal_mask_5scene_ablation.yaml"
OUTPUT_ROOT = ROOT / "results/temporal_mask_5scene_ablation"
SCENES = ("ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2")
DIRECTIONS = ("triu", "tril")
REQUIRED_RAW_FILES = (
    "biwi_eth.txt",
    "biwi_hotel.txt",
    "students001.txt",
    "students003.txt",
    "crowds_zara01.txt",
    "crowds_zara02.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_manifest(config: dict) -> dict:
    raw_root = Path(config["dataset"]["raw_root"])
    missing = [name for name in REQUIRED_RAW_FILES if not (raw_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "RAW DATA UNAVAILABLE: missing " + ", ".join(missing) + f" under {raw_root}"
        )
    return {
        "raw_root": str(raw_root),
        "files": {
            name: {
                "path": str(raw_root / name),
                "bytes": (raw_root / name).stat().st_size,
                "sha256": sha256_file(raw_root / name),
            }
            for name in REQUIRED_RAW_FILES
        },
        "read_only_source": True,
    }


def run_config(base: dict, scene: str, direction: str, raw_manifest: dict) -> tuple[dict, Path]:
    config = json.loads(json.dumps(base))
    config["experiment"]["status"] = "running_raw_data_ablation"
    config["experiment"]["test_scene"] = scene
    config["experiment"]["temporal_mask_direction"] = direction
    config["experiment"]["seed"] = 42
    config["training"]["temporal_mask_direction"] = direction
    config["dataset"]["raw_sha256"] = {
        name: item["sha256"] for name, item in raw_manifest["files"].items()
    }
    run_name = f"{scene.lower()}_{direction}"
    run_root = f"results/temporal_mask_5scene_ablation/runs/{run_name}"
    config["outputs"]["results_root"] = run_root
    config["outputs"]["local_checkpoints"] = f"{run_root}/checkpoints"
    config_path = OUTPUT_ROOT / "configs" / f"{run_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"refusing to overwrite changed run config: {config_path}")
    if not config_path.exists():
        config_path.write_text(serialized, encoding="utf-8")
    return config, config_path


def read_run_records(run_root: Path) -> list[dict]:
    path = run_root / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_combined_manifest(records: list[dict], raw_manifest: dict) -> None:
    payload = {
        "experiment": "IGGCN five-scene temporal mask controlled ablation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dataset": raw_manifest,
        "expected_runs": [
            {"scene": scene, "mask_direction": direction, "seed": 42, "sigma": 3}
            for scene in SCENES
            for direction in DIRECTIONS
        ],
        "runs": records,
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    base = yaml.safe_load(MASTER_CONFIG.read_text(encoding="utf-8"))
    raw_manifest = load_raw_manifest(base)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in ("runs", "logs", "per_sample", "plots"):
        (OUTPUT_ROOT / directory).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "raw_dataset_manifest.json").write_text(
        json.dumps(raw_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    records: list[dict] = []
    for scene in SCENES:
        for direction in DIRECTIONS:
            config, config_path = run_config(base, scene, direction, raw_manifest)
            run_root = ROOT / config["outputs"]["results_root"]
            run_id = f"sigma3_{scene.lower()}_seed42"
            manifest_records = read_run_records(run_root)
            completed = [row for row in manifest_records if row["run_id"] == run_id]
            if completed:
                if len(completed) != 1 or completed[0].get("temporal_mask_direction") != direction:
                    raise RuntimeError(f"invalid existing run record for {scene}/{direction}")
                record = dict(completed[0])
                record.update({"scene": scene, "mask_direction": direction})
                records.append(record)
                print(f"[{scene}-{direction}] already complete; preserving recorded output", flush=True)
                save_combined_manifest(records, raw_manifest)
                continue

            checkpoint = run_root / "checkpoints" / f"{run_id}.pt"
            if checkpoint.exists():
                raise RuntimeError(
                    f"incomplete checkpoint exists without a completed manifest: {checkpoint}"
                )

            log_path = OUTPUT_ROOT / "logs" / f"{scene.lower()}_{direction}.log"
            command = [
                sys.executable,
                str(ROOT / "scripts/train_eval_iggcn.py"),
                "--config",
                str(config_path),
                "--sigma",
                "3",
                "--folds",
                scene,
                "--seed",
                "42",
            ]
            print(f"[{scene}-{direction}] starting 150-epoch raw-data run", flush=True)
            with log_path.open("w", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_stream.write(line)
                return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

            manifest_records = read_run_records(run_root)
            matching = [row for row in manifest_records if row["run_id"] == run_id]
            if len(matching) != 1:
                raise RuntimeError(f"training finished without one manifest record: {scene}/{direction}")
            record = dict(matching[0])
            record.update({"scene": scene, "mask_direction": direction})
            records.append(record)
            save_combined_manifest(records, raw_manifest)

    if len(records) != len(SCENES) * len(DIRECTIONS):
        raise RuntimeError(f"expected 10 completed runs, got {len(records)}")
    print(f"All {len(records)} runs completed. Manifest: {OUTPUT_ROOT / 'run_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
