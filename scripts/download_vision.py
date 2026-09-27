"""Download only the pinned original train/test parquet shards required by the paper."""

import argparse
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def download(raw_root, tasks=None):
    sources = json.loads((ROOT / "manifests/data_sources.json").read_text())
    manifests = sorted((ROOT / "manifests/splits/clip8").glob("*/split_manifest.json"))
    if tasks and set(tasks) - {p.parent.name for p in manifests}:
        raise ValueError("unknown vision task")
    for path in manifests:
        if tasks and path.parent.name not in tasks:
            continue
        manifest = json.loads(path.read_text())
        for entries in manifest["source_files"].values():
            for entry in entries:
                relative = Path(entry["path"].removeprefix("${RAW_DATA_ROOT}/clip8/"))
                collection = relative.parts[0]
                source = sources[collection]
                hf_hub_download(
                    repo_id=source["repo_id"],
                    repo_type="dataset",
                    revision=source["revision"],
                    filename=Path(*relative.parts[1:]).as_posix(),
                    local_dir=Path(raw_root) / "clip8" / collection,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    download(args.raw_root, args.tasks)
