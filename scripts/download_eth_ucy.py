"""Download the ETH/UCY trajectory text files to the configured data disk.

The GitHub source is pinned to a commit. Dataset files remain outside this
repository because their underlying ETH/UCY terms are separate from the
Social-STGCNN code license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


SOURCE_REPOSITORY = "https://github.com/abduallahmohamed/Social-STGCNN"
SOURCE_COMMIT = "333d3a57b4d2705e129b21aefefa09c79b2b9ae1"
SOURCE_BASE = (
    "https://raw.githubusercontent.com/abduallahmohamed/Social-STGCNN/"
    f"{SOURCE_COMMIT}/datasets/raw/all_data"
)
FILES = {
    "ETH": "biwi_eth.txt",
    "HOTEL": "biwi_hotel.txt",
    "UNIV_students001": "students001.txt",
    "UNIV_students003": "students003.txt",
    "ZARA1": "crowds_zara01.txt",
    "ZARA2": "crowds_zara02.txt",
}
DEFAULT_DESTINATION = Path(
    "/media/lrj/54926A1D926A0438/datasets/iggcn_eth_ucy/raw"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_trajectory_file(path: Path) -> tuple[int, int]:
    row_count = 0
    pedestrian_ids: set[float] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected 4 columns, got {len(fields)}"
                )
            values = [float(value) for value in fields]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number}: non-finite value")
            pedestrian_ids.add(values[1])
            row_count += 1
    if row_count == 0:
        raise ValueError(f"{path}: empty trajectory file")
    return row_count, len(pedestrian_ids)


def download_file(filename: str, destination: Path, force: bool) -> dict[str, object]:
    path = destination / filename
    if path.exists() and not force:
        rows, pedestrians = validate_trajectory_file(path)
        return {
            "file": filename,
            "status": "already_present",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": rows,
            "pedestrians": pedestrians,
        }

    url = f"{SOURCE_BASE}/{filename}"
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Empty response while downloading {url}")

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination, prefix=f".{filename}.", suffix=".part", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(payload)
    try:
        validate_trajectory_file(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

    rows, pedestrians = validate_trajectory_file(path)
    return {
        "file": filename,
        "status": "downloaded",
        "source_url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "pedestrians": pedestrians,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true", help="replace existing files")
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    files = [download_file(name, args.destination, args.force) for name in FILES.values()]
    manifest = {
        "dataset": "ETH/UCY pedestrian trajectories",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination": str(args.destination.resolve()),
        "scene_mapping": FILES,
        "format": "frame_id pedestrian_id x y; whitespace-delimited",
        "license_note": (
            "The source repository's MIT license covers its code. It does not replace "
            "the original ETH/UCY dataset terms; files are kept locally and are not "
            "committed or redistributed by this project."
        ),
        "files": files,
    }
    manifest_path = args.destination / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
