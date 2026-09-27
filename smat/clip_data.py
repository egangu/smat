"""Small local-data adapter for the FusionBench CLIP eight-task collection.

This module deliberately reads *only* materialized local data.  A preparation
job downloads the Hugging Face parquet files in a user-selected directory;
training and evaluation then use the same files without depending on the
FusionBench runtime or on network access.  The adapter also accepts datasets
saved with :meth:`datasets.DatasetDict.save_to_disk`, which is convenient for
one-time preprocessing jobs.
"""

from __future__ import annotations

import io
import json
import os
import random
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


# The official SUN397 parquet contains a TIFF that Pillow decodes successfully
# while emitting this warning once per worker process.  Suppress only that
# exact, non-fatal decoder message so real image errors still surface.
warnings.filterwarnings(
    "ignore",
    message="Truncated File Read",
    category=UserWarning,
    module="PIL.TiffImagePlugin",
)


@dataclass(frozen=True)
class ClipTaskSpec:
    """Names used by the local provider and its original Hugging Face source."""

    name: str
    dataset_id: str
    directory_names: tuple[str, ...]


CLIP8_TASKS: dict[str, ClipTaskSpec] = {
    "Cars": ClipTaskSpec("Cars", "tanganke/stanford_cars", ("cars", "stanford_cars")),
    "DTD": ClipTaskSpec("DTD", "tanganke/dtd", ("dtd",)),
    "EuroSAT": ClipTaskSpec("EuroSAT", "tanganke/eurosat", ("eurosat",)),
    "GTSRB": ClipTaskSpec("GTSRB", "tanganke/gtsrb", ("gtsrb",)),
    "MNIST": ClipTaskSpec("MNIST", "ylecun/mnist", ("mnist",)),
    "RESISC45": ClipTaskSpec("RESISC45", "tanganke/resisc45", ("resisc45",)),
    "SUN397": ClipTaskSpec("SUN397", "tanganke/sun397", ("sun397",)),
    # The HF repository also contains ``full_numbers`` with a different
    # schema.  Pin the FusionBench protocol to ``cropped_digits`` so a
    # recursive parquet scan cannot mix the two configurations.
    "SVHN": ClipTaskSpec(
        "SVHN", "ufldl-stanford/svhn", ("svhn/cropped_digits", "svhn")
    ),
}


def canonical_task(task: str) -> str:
    """Validate one canonical FusionBench CLIP8 task name."""

    if task not in CLIP8_TASKS:
        choices = ", ".join(CLIP8_TASKS)
        raise ValueError(f"unknown CLIP8 task {task!r}; expected one of: {choices}")
    return task


def task_spec(task: str) -> ClipTaskSpec:
    return CLIP8_TASKS[canonical_task(task)]


@dataclass(frozen=True)
class VisionTaskData:
    """One materialized split together with its invariant task metadata."""

    task: str
    split: str
    dataset: Dataset
    classnames: tuple[str, ...]
    sample_class_weights: dict[int, float] | None = None
    sampling: dict[str, Any] | None = None


def _require_datasets():
    try:
        import datasets
    except ImportError as error:  # pragma: no cover - exercised on GPU runtime
        raise ImportError(
            "fusionbench_hf requires the optional `datasets` package. "
            "Install it in the SMAT vision environment."
        ) from error
    return datasets


def _task_directory(root: Path, spec: ClipTaskSpec) -> Path:
    for name in (spec.name, *spec.directory_names):
        candidate = root / name
        if candidate.exists():
            return candidate
    tried = ", ".join(str(root / name) for name in (spec.name, *spec.directory_names))
    raise FileNotFoundError(
        f"no local FusionBench data for {spec.name}; looked for {tried}. "
        "Download parquet data first; runtime loading never fetches a dataset."
    )


def _parquet_files(directory: Path, split: str) -> list[Path]:
    """Find a normal Hugging Face ``train`` or ``test`` parquet split only."""

    found: list[Path] = []
    for path in directory.rglob("*.parquet"):
        relative = path.relative_to(directory).as_posix().lower()
        filename = path.name.lower()
        split_parts = {part for part in relative.split("/")[:-1]}
        if (
            split in split_parts
            or filename.startswith(f"{split}-")
            or filename.startswith(f"{split}_")
        ):
            found.append(path)
    return sorted(found)


def _load_local_splits(directory: Path):
    datasets = _require_datasets()
    # The packaged parquet builder can still emit a download-count HEAD
    # request. This adapter reads local files, so disable that optional call.
    datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
    manifest_path = directory / "split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "smat-indexed-splits-v1":
            raise ValueError(f"unsupported split manifest: {manifest_path}")
        files = {}
        for split, entries in manifest["source_files"].items():
            files[split] = []
            for entry in entries:
                path = Path(os.path.expandvars(entry["path"]))
                if path.stat().st_size != entry["size_bytes"]:
                    raise ValueError(f"source size changed since splitting: {path}")
                files[split].append(str(path))
        source = datasets.load_dataset("parquet", data_files=files)
        for split, entries in manifest["source_files"].items():
            if len(source[split]) != sum(entry["num_rows"] for entry in entries):
                raise ValueError(f"source row count changed: {directory}/{split}")
        result = {}
        for split, selection in manifest["splits"].items():
            data = source[selection["source"]]
            indices = selection["indices"]
            result[split] = data if indices is None else data.select(indices)
        return datasets.DatasetDict(result)
    # ``DatasetDict.save_to_disk`` is the compact format used by a preparation
    # script.  It preserves ClassLabel metadata when that information exists.
    if (directory / "dataset_dict.json").exists():
        loaded = datasets.load_from_disk(str(directory))
        if "train" in loaded and "test" in loaded:
            return loaded

    # Individual saved Dataset directories are also accepted.
    saved = {}
    for split in ("train", "test"):
        path = directory / split
        if (path / "state.json").exists():
            saved[split] = datasets.load_from_disk(str(path))
    if len(saved) == 2:
        return saved

    files = {
        split: [str(path) for path in _parquet_files(directory, split)]
        for split in ("train", "test")
    }
    missing = [split for split, paths in files.items() if not paths]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing local {', '.join(missing)} parquet split(s). "
            "Only ordinary train/test splits are valid for fusionbench_hf."
        )
    return datasets.load_dataset("parquet", data_files=files)


def _read_sidecar_classnames(directory: Path) -> tuple[str, ...] | None:
    """Use an explicit sidecar only when parquet metadata lacks ClassLabel names."""

    for name in ("classnames.json", "class_names.json"):
        path = directory / name
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("classnames", value.get("class_names"))
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        raise ValueError(
            f"{path} must be a JSON string list or {{'classnames': [...]}}"
        )
    return None


def _classnames(dataset: Any, label_column: str, directory: Path) -> tuple[str, ...]:
    feature = dataset.features.get(label_column)
    names = getattr(feature, "names", None)
    if names:
        return tuple(str(name) for name in names)
    sidecar = _read_sidecar_classnames(directory)
    if sidecar:
        return sidecar
    raise ValueError(
        f"{directory} has no ClassLabel names for {label_column!r}. "
        "Place the official FusionBench names in classnames.json beside the parquet files; "
        "zero-shot CLIP evaluation must not invent class labels."
    )


def _column(columns: Iterable[str], choices: Sequence[str], what: str) -> str:
    columns = set(columns)
    for name in choices:
        if name in columns:
            return name
    raise ValueError(
        f"dataset has no {what} column; available columns: {sorted(columns)}"
    )


def load_fusionbench_task(
    task: str,
    split: str,
    *,
    root: str | Path,
    max_samples: int | None = None,
    sample_seed: int = 0,
) -> VisionTaskData:
    """Load one deterministic FusionBench-HF split from local disk.

    The protocol uses the materialized train split for experts. An explicit
    indexed manifest can reserve a separate dev split without copying images.
    """

    if split not in {"full_train", "dev", "test"}:
        raise ValueError("split must be full_train, dev or test")
    spec = task_spec(task)
    directory = _task_directory(Path(root), spec)
    splits = _load_local_splits(directory)
    if "train" not in splits or "test" not in splits:
        raise ValueError(f"{directory} must expose ordinary train and test splits")
    source_split = "train" if split == "full_train" else split
    if source_split not in splits:
        raise ValueError(f"{directory} has no reserved {source_split} split")
    selected = splits[source_split]
    image_column = _column(selected.column_names, ("image", "img"), "image")
    label_column = _column(selected.column_names, ("label", "labels"), "integer label")
    sample_class_weights = None
    sampling = None
    if max_samples is not None and split == "test":
        groups: dict[int, list[int]] = {}
        for index, label in enumerate(selected[label_column]):
            groups.setdefault(int(label), []).append(index)
        size = min(max_samples, len(selected))
        if size < len(groups):
            raise ValueError(f"max_samples must cover all {len(groups)} test classes")
        # Round-robin quotas cover small classes too. Frequency weights below
        # retain the original example-micro test objective, not class-macro.
        labels = sorted(groups)
        counts = dict.fromkeys(labels, 0)
        remaining = size
        while remaining:
            for label in labels:
                if counts[label] < len(groups[label]):
                    counts[label] += 1
                    remaining -= 1
                    if not remaining:
                        break
        rng = random.Random(sample_seed)
        indices = sorted(
            index
            for label in labels
            for index in rng.sample(groups[label], counts[label])
        )
        sample_class_weights = {
            label: len(groups[label]) / (len(selected) * counts[label])
            for label in labels
        }
        sampling = {
            "rule": "class-stratified; original-test-frequency weighting",
            "seed": sample_seed,
            "total_examples": len(selected),
            "sample_examples": size,
            "class_counts": {label: len(groups[label]) for label in labels},
            "sample_class_counts": counts,
            "indices": indices,
        }
        selected = selected.select(indices)
    elif max_samples is not None:
        selected = selected.select(range(min(max_samples, len(selected))))
    # Attach names here rather than trusting the train-only split's features:
    # materialized datasets commonly preserve the ClassLabel only on one split.
    names = _classnames(splits["train"], label_column, directory)
    return VisionTaskData(
        task=spec.name,
        split=split,
        dataset=HfImageClassificationDataset(selected, image_column, label_column),
        classnames=names,
        sample_class_weights=sample_class_weights,
        sampling=sampling,
    )


class HfImageClassificationDataset(Dataset):
    """Turn one decoded local HF image dataset into a torch vision dataset."""

    def __init__(self, dataset: Any, image_column: str, label_column: str):
        self.dataset = dataset
        self.image_column = image_column
        self.label_column = label_column
        self.transform: Callable[[Any], torch.Tensor] | None = None

    def with_transform(
        self, transform: Callable[[Any], torch.Tensor] | None
    ) -> "HfImageClassificationDataset":
        self.transform = transform
        return self

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _image(value: Any):
        # ``datasets`` normally decodes Image to PIL.  This small fallback also
        # handles a direct parquet ``{bytes, path}`` representation.
        if isinstance(value, dict) and value.get("bytes") is not None:
            from PIL import Image

            return Image.open(io.BytesIO(value["bytes"]))
        return value

    def __getitem__(self, index: int):
        row = self.dataset[index]
        image = self._image(row[self.image_column])
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row[self.label_column])
