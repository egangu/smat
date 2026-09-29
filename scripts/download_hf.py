"""Download the frozen SMAT Table 1/2 data or experts at pinned Hub revisions."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
TRACE_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

CLIP_TASKS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]


def check(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}. Preserve/inspect this file before retrying.")


def download_data(record, directory, tasks):
    for name, expected in record["files"].items():
        if name.split("/")[1] not in tasks:
            continue
        target = directory / name
        if not target.exists():
            hf_hub_download(record["repo_id"], name, repo_type="dataset",
                            revision=record["revision"], local_dir=directory)
        check(target, expected)
        print(target, flush=True)


def download_expert(record, directory):
    identity = {k: record[k] for k in ("repo_id", "revision")}
    marker = directory / ".hf-release.json"
    if directory.exists() and any(directory.iterdir()):
        if not marker.exists() or json.loads(marker.read_text()) != identity:
            raise ValueError(f"Refusing to overwrite an existing checkpoint: {directory}. Use a fresh output root.")
    directory.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    # Never replace an existing corrupt or independently modified payload silently.
    for name, expected in record["files"].items():
        if (directory / name).exists():
            check(directory / name, expected)
    snapshot_download(record["repo_id"], revision=record["revision"], local_dir=directory)
    for name, expected in record["files"].items():
        check(directory / name, expected)
    print(directory, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="download frozen TRACE or CLIP8 splits")
    data.add_argument("--data-root", required=True, type=Path)
    data.add_argument("--suite", choices=["trace", "clip8"], default="trace")
    experts = commands.add_parser("experts", help="download trained FT or SMAT task experts")
    experts.add_argument("--output-root", required=True, type=Path)
    experts.add_argument("--model", choices=["llama1b", "llama8b", "vitb32", "vitl14"], required=True)
    experts.add_argument("--method", choices=["ft", "smat"], required=True)
    for command in (data, experts):
        command.add_argument("--tasks", nargs="+", choices=TRACE_TASKS + CLIP_TASKS)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifests/hf_release.json").read_text())
    vision = args.suite == "clip8" if args.command == "data" else args.model.startswith("vit")
    tasks = CLIP_TASKS if vision else TRACE_TASKS
    args.tasks = args.tasks or tasks
    if set(args.tasks) - set(tasks):
        parser.error("Tasks do not match the selected suite/model")
    if args.command == "data":
        download_data(manifest["vision_dataset" if vision else "dataset"], args.data_root.expanduser(), args.tasks)
    else:
        backbone = args.model.removeprefix("vit" if vision else "llama")
        optimizer = "adam" if vision else "adamw"
        case = f"{backbone}_{optimizer}_{args.method}"
        selected = [m for m in manifest["models"] if m["case"].startswith(case + "_") and m["task"] in args.tasks]
        if len(selected) != len(set(args.tasks)):
            raise ValueError("The release manifest does not contain every requested expert")
        run = f"{args.model}_{optimizer}_{args.method}"
        for model in selected:
            download_expert(model, args.output_root.expanduser() / "training/experts" / run / model["task"])


if __name__ == "__main__":
    main()
