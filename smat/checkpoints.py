"""Hugging Face safetensors checkpoint adapter for streaming merge kernels."""

from __future__ import annotations

import shutil
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .contracts import CheckpointRef


def _checkpoint_files(checkpoint: CheckpointRef) -> list[Path]:
    path = Path(checkpoint)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"expected a safetensors file, got {path}")
        return [path]
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint in {path}")
    return files


def warm_checkpoint_files(checkpoints: Iterable[CheckpointRef]) -> int:
    """Read shared checkpoints sequentially before tensor-wise mmap access.

    Four readers use at most 64 MiB of application buffers. The OS owns the
    page cache; checkpoint contents and merge arithmetic are unchanged.
    """
    files = dict.fromkeys(
        file for checkpoint in checkpoints for file in _checkpoint_files(checkpoint)
    )

    def read(file):
        buffer = bytearray(16 * 1024 * 1024)
        total = 0
        with file.open("rb", buffering=0) as stream:
            while count := stream.readinto(buffer):
                total += count
        return total

    with ThreadPoolExecutor(max_workers=4) as pool:
        return sum(pool.map(read, files))


class _ReadStore:
    def __init__(self, handles: dict[str, object]):
        self._handles = handles
        self._keys = tuple(handles)

    def keys(self) -> Iterable[str]:
        return self._keys

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._handles[key].get_tensor(key)  # type: ignore[union-attr]

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        raise TypeError("a read-only safetensors store cannot be written")


class _WriteStore:
    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def keys(self) -> Iterable[str]:
        return self.tensors.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        self.tensors[key] = tensor.detach().cpu().contiguous()


def _directory(path: CheckpointRef) -> Path:
    path = Path(path)
    return path.parent if path.suffix == ".safetensors" else path


def _copy_assets(template: Path, output: Path) -> None:
    """Copy HF configuration/tokenizer assets, not the source tensor weights."""

    if not template.is_dir():
        return
    allowed = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
    }
    for source in template.iterdir():
        if not source.is_file() or source.name not in allowed:
            continue
        destination = output / source.name
        if source.suffix == ".json":
            content = json.loads(source.read_text())
            if isinstance(content, dict):
                content.pop("_name_or_path", None)
                content.pop("name_or_path", None)
            destination.write_text(json.dumps(content, indent=2) + "\n")
        else:
            shutil.copyfile(source, destination)


class HuggingFaceSafetensorsAdapter:
    """Expose HF checkpoint directories as named CPU tensors.

    Reads support both a single safetensors checkpoint and standard HF shards.
    Merged output is deliberately one safetensors file: both current SMAT
    backbones fit below the format's file limit, and this keeps the adapter
    small.  The template's tokenizer/config files are copied unchanged.
    """

    @contextmanager
    def read(self, checkpoint: CheckpointRef):
        with ExitStack() as stack:
            handles: dict[str, object] = {}
            for file in _checkpoint_files(checkpoint):
                handle = stack.enter_context(
                    safe_open(file, framework="pt", device="cpu")
                )
                for key in handle.keys():
                    if key in handles:
                        raise ValueError(
                            f"duplicate tensor {key!r} across shards in {checkpoint}"
                        )
                    handles[key] = handle
            yield _ReadStore(handles)

    @contextmanager
    def write(self, output: CheckpointRef, *, template: CheckpointRef):
        output_path = Path(output)
        output_dir = _directory(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        _copy_assets(Path(template), output_dir)
        store = _WriteStore()
        try:
            yield store
        except BaseException:
            raise
        else:
            output_file = (
                output_path
                if output_path.suffix == ".safetensors"
                else output_dir / "model.safetensors"
            )
            save_file(store.tensors, str(output_file))
