"""Reconstruct the paper splits from verified original files and frozen indices."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

MANIFESTS = Path(__file__).resolve().parents[1] / "manifests" / "splits"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare_trace(raw_root, destination, manifest, task):
    source = raw_root / "trace" / task
    for split, expected in manifest["source_sha256"].items():
        if sha256(source / f"{split}.json") != expected:
            raise ValueError(f"TRACE source checksum differs: {source.name}/{split}")
    rows = json.loads((source / "train.json").read_text())
    for split in ("train", "train_unique", "dev"):
        write_json(
            destination / f"{split}.json",
            [rows[i] for i in manifest[split + "_indices"]],
        )
    shutil.copyfile(source / "eval.json", destination / "eval.json")
    for split, expected in manifest["output_sha256"].items():
        if sha256(destination / f"{split}.json") != expected:
            raise ValueError(
                f"Reconstructed TRACE checksum differs: {source.name}/{split}"
            )


def prepare_clip(raw_root, destination, manifest):
    import pyarrow.parquet as pq

    for entries in manifest["source_files"].values():
        for entry in entries:
            source = raw_root / entry["path"].removeprefix("${RAW_DATA_ROOT}/")
            if (
                source.stat().st_size != entry["size_bytes"]
                or sha256(source) != entry["sha256"]
            ):
                raise ValueError(
                    f"Vision source checksum differs: {destination.name}/{source.name}"
                )
            if pq.ParquetFile(source).metadata.num_rows != entry["num_rows"]:
                raise ValueError(f"Vision row count differs: {source.name}")
            # Paths are resolved at preparation, so loaders need no extra environment variable.
            entry["path"] = str(source.resolve())
    groups = manifest["group_ids"]
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if set(groups[left]) & set(groups[right]):
                raise ValueError(
                    f"Split identity overlap: {destination.name}/{left}/{right}"
                )


def prepare(raw_root, output, suite="all", tasks=None):
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    paths = sorted(MANIFESTS.glob("*/*/split_manifest.json"))
    selected = [
        p
        for p in paths
        if (suite == "all" or p.parts[-3] == suite)
        and (not tasks or p.parent.name in tasks)
    ]
    if not selected or (tasks and set(tasks) - {p.parent.name for p in selected}):
        raise ValueError("No matching manifests or unknown task")
    for source in selected:
        relative = source.relative_to(MANIFESTS)
        destination = output / relative.parent
        if destination.exists():
            raise FileExistsError(
                f"Refusing to replace an existing prepared task: {destination}"
            )
        temporary = destination.with_name(destination.name + ".partial")
        temporary.mkdir(parents=True)
        manifest = json.loads(source.read_text())
        # Keep the canonical task name while using an isolated staging directory.
        if relative.parts[0] == "trace":
            prepare_trace(raw_root, temporary, manifest, destination.name)
            manifest["source"] = str(raw_root / "trace" / destination.name)
        else:
            prepare_clip(raw_root, temporary, manifest)
        write_json(temporary / "split_manifest.json", manifest)
        temporary.rename(destination)
        print(f"Verified {relative.parent}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "trace", "clip8"), default="all")
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    prepare(args.raw_root, args.output, args.suite, args.tasks)
