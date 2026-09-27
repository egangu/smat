"""Download model assets and require the published checkpoint byte checksums."""

import argparse
import hashlib
import json
import shutil
from urllib.parse import urlencode
from urllib.request import urlopen
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "llama1b": ("LLM-Research/Llama-3.2-1B-Instruct", "modelscope"),
    "llama8b": ("LLM-Research/Llama-3.1-8B-Instruct", "modelscope"),
    "vitb32": (
        "openai/clip-vit-base-patch32",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    ),
    "vitl14": (
        "openai/clip-vit-large-patch14",
        "32bd64288804d66eefd0ccbe215aa642df71cc41",
    ),
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check(path, expected):
    if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
        raise ValueError(
            f"Model byte checksum differs: {path.name}; do not silently substitute a different checkpoint"
        )


def download_llama(path, source, expected):
    """Fetch the byte-matched public mirror at a fixed per-file Git revision."""
    query = urlencode({"Revision": source["revision"], "FilePath": path.name})
    url = f"https://modelscope.cn/api/v1/models/{source['repository']}/repo?{query}"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    check(partial, expected)
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    checksums = json.loads((ROOT / "manifests/model_checksums.json").read_text())
    llama_sources = json.loads((ROOT / "manifests/llama_sources.json").read_text())
    for name in args.models:
        repo, revision = MODELS[name]
        directory = repo.split("/")[-1]
        for relative, expected in checksums.items():
            folder, filename = relative.split("/", 1)
            if folder != directory:
                continue
            path = args.model_root / directory / filename
            if not path.exists() and not args.verify_only:
                if revision == "modelscope":
                    download_llama(path, llama_sources[relative], expected)
                else:
                    hf_hub_download(
                        repo_id=repo,
                        revision=revision,
                        filename=filename,
                        local_dir=args.model_root / directory,
                    )
            check(path, expected)
        print(f"Verified {name}", flush=True)


if __name__ == "__main__":
    main()
