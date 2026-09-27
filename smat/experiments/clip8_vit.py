"""HF CLIP adapter for the SMAT eight-expert vision benchmark.

The public functions at the end of this file are intentionally shaped like the
LLM adapter consumed by the generic SMAT runner.  They leave optimizer,
scheduler and the chosen update rule to the runner, while this file owns only
the TA8 image/task boundary: local data, frozen zero-shot heads and top-1
evaluation.

Prompt templates below are a verbatim, data-only transcription of
``sgd_saft/src/tv_datasets/templates.py`` from the official SAFT-Merge
repository (commit source: https://github.com/baiklab/SAFT-Merge).  This is
only provenance for static strings: the canonical protocol has no SAFT runtime
or OpenCLIP dependency.  Class names come from the local FusionBench parquet
ClassLabel metadata (or its explicit ``classnames.json`` sidecar), so they
cannot silently drift from the downloaded labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, default_collate

from ..clip_data import (
    CLIP8_TASKS,
    VisionTaskData,
    canonical_task,
    load_fusionbench_task,
)
from ..contracts import CheckpointRef, TensorStore
from ..runtime import atomic_write_json

# Canonical FusionBench task-vector order.  A mapping, rather than a bare list,
# makes task metadata available to callers without a second registry.
TASKS = CLIP8_TASKS
# The common runner usually builds the model before its loader.  ViT relies on
# that order to cache the exact HF preprocessing transform once, rather than
# constructing a second CLIP model merely to obtain a transform.
MODEL_BEFORE_LOADER = True


# Static SAFT-Merge template transcription; string templates are less magical
# than lambdas and are straightforward to serialize in experiment metadata.
_PROMPTS: dict[str, tuple[str, ...]] = {
    "Cars": (
        "a photo of a {classname}.",
        "a photo of the {classname}.",
        "a photo of my {classname}.",
        "i love my {classname}!",
        "a photo of my dirty {classname}.",
        "a photo of my clean {classname}.",
        "a photo of my new {classname}.",
        "a photo of my old {classname}.",
    ),
    "DTD": (
        "a photo of a {classname} texture.",
        "a photo of a {classname} pattern.",
        "a photo of a {classname} thing.",
        "a photo of a {classname} object.",
        "a photo of the {classname} texture.",
        "a photo of the {classname} pattern.",
        "a photo of the {classname} thing.",
        "a photo of the {classname} object.",
    ),
    "EuroSAT": (
        "a centered satellite photo of {classname}.",
        "a centered satellite photo of a {classname}.",
        "a centered satellite photo of the {classname}.",
    ),
    "GTSRB": (
        'a zoomed in photo of a "{classname}" traffic sign.',
        'a centered photo of a "{classname}" traffic sign.',
        'a close up photo of a "{classname}" traffic sign.',
    ),
    "MNIST": ('a photo of the number: "{classname}".',),
    "RESISC45": (
        "satellite imagery of {classname}.",
        "aerial imagery of {classname}.",
        "satellite photo of {classname}.",
        "aerial photo of {classname}.",
        "satellite view of {classname}.",
        "aerial view of {classname}.",
        "satellite imagery of a {classname}.",
        "aerial imagery of a {classname}.",
        "satellite photo of a {classname}.",
        "aerial photo of a {classname}.",
        "satellite view of a {classname}.",
        "aerial view of a {classname}.",
        "satellite imagery of the {classname}.",
        "aerial imagery of the {classname}.",
        "satellite photo of the {classname}.",
        "aerial photo of the {classname}.",
        "satellite view of the {classname}.",
        "aerial view of the {classname}.",
    ),
    "SUN397": ("a photo of a {classname}.", "a photo of the {classname}."),
    "SVHN": ('a photo of the number: "{classname}".',),
}


def _data_root(config: Mapping[str, Any]) -> Path:
    return Path(config["data_root"])


def task_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Canonicalize a config task list, preserving the eight-task order."""

    names = tuple(canonical_task(task) for task in config["tasks"])
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate CLIP8 task in config: {names}")
    return names


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = config.get("device")
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrozenZeroShotHead(nn.Module):
    """CLIP text classifier with a fixed, normalized class-text matrix."""

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights.float(), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Keep the head in fp32 even under a mixed-precision image encoder.  The
        # official SAFT head serializes fp32 zero-shot weights as well.
        return F.normalize(features.float(), dim=-1) @ self.weights.T


class _HfClipPreprocess:
    """Pickle-friendly wrapper around the published CLIP image processor."""

    def __init__(self, image_processor: Any):
        self.image_processor = image_processor

    def __call__(self, image: Any) -> torch.Tensor:
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"][
            0
        ]


@dataclass(frozen=True)
class _ClipModelParts:
    """Pieces shared by an image-only expert and its frozen CLIP classifier."""

    full_model: nn.Module
    encoder: nn.Module
    projection: nn.Module
    tokenizer: Any
    train_preprocess: Any
    eval_preprocess: Any


_HF_VISION_PREFIXES = ("vision_model.", "module.vision_model.")


def _image_features(
    encoder: nn.Module, projection: nn.Module, images: torch.Tensor
) -> torch.Tensor:
    return projection(encoder(pixel_values=images).pooler_output)


def _text_features(
    full_model: nn.Module, tokenizer: Any, prompts: Sequence[str], device: torch.device
) -> torch.Tensor:
    encoded = tokenizer(list(prompts), padding=True, return_tensors="pt").to(device)
    return full_model.text_projection(full_model.text_model(**encoded).pooler_output)


def _zero_shot_class_weights(
    full_model: nn.Module,
    tokenizer: Any,
    classnames: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Build each zero-shot class weight with its own prompt encoding pass."""

    weights = []
    for classname in classnames:
        prompts = [template.format(classname=classname) for template in templates]
        embeddings = F.normalize(
            _text_features(full_model, tokenizer, prompts, device), dim=-1
        ).mean(dim=0)
        weights.append(F.normalize(embeddings, dim=-1))
    return torch.stack(weights)


class ClipVisionBundle(nn.Module):
    """Image encoder plus lazily materialized, task-specific frozen text heads.

    ``_text_model`` is intentionally unregistered: the only serializable
    component is ``encoder``.  It is discarded as soon as the one required
    training/evaluation head is built, which is the standard independent-expert
    workflow and prevents text-tower parameters entering any task vector.
    """

    def __init__(
        self,
        *,
        parts: _ClipModelParts,
        device: torch.device,
        data_root: Path,
    ):
        super().__init__()
        # FusionBench keeps this projection frozen beside the swapped
        # ``vision_model``.  It remains outside the expert checkpoint.
        self.encoder = parts.encoder
        self.projection = parts.projection
        self.projection.requires_grad_(False)
        self.heads = nn.ModuleDict()
        self.train_preprocess = parts.train_preprocess
        self.eval_preprocess = parts.eval_preprocess
        self._task_info: dict[str, tuple[str, ...]] = {}
        self._head_device = device
        self._data_root = data_root
        # nn.Module.__setattr__ would register the full CLIP model as a
        # submodule.  It remains outside the image-only expert state dict.
        self.__dict__["_text_model"] = parts.full_model
        self.__dict__["_tokenizer"] = parts.tokenizer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _image_features(self.encoder, self.projection, images)

    def register_task(self, task: str, classnames: Sequence[str]) -> None:
        canonical = canonical_task(task)
        names = tuple(str(name) for name in classnames)
        if not names:
            raise ValueError(f"{canonical} has no class names")
        prior = self._task_info.get(canonical)
        if prior is not None and prior != names:
            raise ValueError(f"inconsistent class names registered for {canonical}")
        self._task_info[canonical] = names

    @torch.no_grad()
    def _build_head(self, task: str) -> FrozenZeroShotHead:
        canonical = canonical_task(task)
        names = self._task_info.get(canonical)
        if names is None:
            raise RuntimeError(
                f"register a loader for {canonical} before computing its CLIP loss"
            )
        text_model = self.__dict__.get("_text_model")
        if text_model is None:
            raise RuntimeError(
                "this bundle already released its text tower after building another task head; "
                "use one model bundle per independent expert/evaluation task"
            )
        encoder_training = self.encoder.training
        text_model.to(self._head_device).eval()
        try:
            weights = (
                _zero_shot_class_weights(
                    text_model,
                    self.__dict__["_tokenizer"],
                    names,
                    _PROMPTS[canonical],
                    self._head_device,
                ).float()
                * text_model.logit_scale.exp().float()
            )
        finally:
            # The full model's vision branch and ``self.encoder`` are shared;
            # text_model.eval() above recurses into it.  Restore the caller's
            # mode so frozen-head creation cannot silently disable ViT training.
            self.encoder.train(encoder_training)
        head = FrozenZeroShotHead(weights).to(self._head_device)
        head.requires_grad_(False)
        self.heads[canonical] = head
        # The visual encoder is shared with ``text_model`` but remains held by
        # ``self.encoder``.  Deleting the full CLIP container releases text-only
        # parameters immediately after its one-time head construction.
        self.__dict__["_text_model"] = None
        return head

    def logits(self, task: str, images: torch.Tensor) -> torch.Tensor:
        canonical = canonical_task(task)
        head = (
            self.heads[canonical]
            if canonical in self.heads
            else self._build_head(canonical)
        )
        return head(self(images))

    def ensure_task(self, task: str) -> None:
        """Load only local metadata when a generic runner has not registered a loader."""

        canonical = canonical_task(task)
        if canonical not in self._task_info:
            metadata = load_fusionbench_task(
                canonical, "full_train", root=self._data_root, max_samples=0
            )
            self.register_task(canonical, metadata.classnames)


_PREPROCESS_CACHE: dict[str, tuple[Any, Any]] = {}


def _hf_model_source(config: Mapping[str, Any]) -> str:
    return str(config["base_model"])


def _preprocess_key(config: Mapping[str, Any]) -> str:
    return _hf_model_source(config)


def _build_hf_model(config: Mapping[str, Any]) -> _ClipModelParts:
    """Load the local FusionBench CLIP and freeze every non-vision component."""

    try:
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel
    except ImportError as error:  # pragma: no cover - depends on vision env
        raise ImportError(
            "the FusionBench HF CLIP backend requires transformers"
        ) from error
    source = _hf_model_source(config)
    full_model = CLIPModel.from_pretrained(source, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    # Published FusionBench experts swap only vision_model.  The text tower,
    # visual projection and logit scale therefore stay frozen at the base.
    full_model.text_model.requires_grad_(False)
    full_model.visual_projection.requires_grad_(False)
    full_model.logit_scale.requires_grad_(False)
    preprocess = _HfClipPreprocess(image_processor)
    return _ClipModelParts(
        full_model=full_model,
        encoder=full_model.vision_model,
        projection=full_model.visual_projection,
        tokenizer=tokenizer,
        train_preprocess=preprocess,
        eval_preprocess=preprocess,
    )


def _fallback_preprocess(config: Mapping[str, Any], training: bool):
    """Return the already-instantiated model's exact image transform."""

    key = _preprocess_key(config)
    if key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[key][0 if training else 1]
    raise RuntimeError(
        "build_model must run before build_loader for the HF CLIP backend"
    )


def _collate(examples: list[tuple[torch.Tensor, int]], *, task: str) -> dict[str, Any]:
    images, labels = default_collate(examples)
    return {"images": images, "labels": labels, "task": task}


def build_loader(
    config: Mapping[str, Any],
    task: str,
    split: str = "train",
    max_samples: int | None = None,
    sample_seed: int = 0,
) -> DataLoader:
    """Return an exact local TA8 loader; batches include their task identity."""

    if split == "train" and not config["data"]["train_full_split"]:
        raise ValueError("CLIP8 protocol requires data.train_full_split=true")
    source_split = "full_train" if split == "train" else split
    loaded: VisionTaskData = load_fusionbench_task(
        task,
        source_split,
        root=_data_root(config),
        max_samples=max_samples,
        sample_seed=sample_seed,
    )
    transform = _fallback_preprocess(config, training=split == "train")
    dataset = loaded.dataset.with_transform(transform)
    train = split == "train"
    phase_config = config["train" if train else "evaluation"]
    batch_size = int(phase_config["batch_size"])
    workers = int(phase_config["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator if train else None,
        collate_fn=partial(_collate, task=loaded.task),
    )
    # DataLoader has no metadata channel.  These tiny attributes let a generic
    # runner register the frozen task head without a vision-specific branch.
    loader.smat_task = loaded.task  # type: ignore[attr-defined]
    loader.smat_classnames = loaded.classnames  # type: ignore[attr-defined]
    loader.smat_sample_class_weights = loaded.sample_class_weights  # type: ignore[attr-defined]
    loader.smat_sampling = loaded.sampling  # type: ignore[attr-defined]
    return loader


def build_model(
    config: Mapping[str, Any],
    checkpoint: str | Path | None = None,
    training: bool = True,
) -> ClipVisionBundle:
    """Load one local CLIP backbone, retaining only its trainable vision tower."""

    smoothing = (
        float(config.get("train", {}).get("label_smoothing", 0.0)) if training else 0.0
    )
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("label_smoothing must be between zero and one")
    temperature = (
        float(config.get("train", {}).get("loss_temperature", 1.0)) if training else 1.0
    )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("loss_temperature must be finite and positive")
    device = _device(config)
    parts = _build_hf_model(config)
    bundle = ClipVisionBundle(
        parts=parts,
        device=device,
        data_root=_data_root(config),
    ).to(device)
    bundle.label_smoothing = smoothing
    bundle.loss_temperature = temperature
    _PREPROCESS_CACHE[_preprocess_key(config)] = (
        parts.train_preprocess,
        parts.eval_preprocess,
    )
    if checkpoint is not None:
        bundle.encoder.load_state_dict(_image_encoder_state(checkpoint), strict=True)
    if training and config.get("train", {}).get("gradient_checkpointing", False):
        bundle.encoder.gradient_checkpointing_enable()
    bundle.train(training)
    return bundle


def _checkpoint_file(checkpoint: str | Path) -> Path:
    """Resolve SMAT image-encoder checkpoint directories to one small file."""

    path = Path(checkpoint)
    if path.is_dir():
        for name in ("encoder.pt", "model.safetensors", "pytorch_model.bin"):
            candidate = path / name
            if candidate.exists():
                return candidate
        return path / "encoder.pt"
    if path.suffix in {".pt", ".bin", ".safetensors"}:
        return path
    return path / "encoder.pt"


def checkpoint_path(config: Mapping[str, Any], directory: str | Path) -> Path:
    """Common-runner hook: experts/merges are directories, tensors are one file."""

    del config
    return _checkpoint_file(directory)


def _torch_load(path: Path) -> Any:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _image_encoder_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Read an expert state or select the backend's image module from a base."""

    path = _checkpoint_file(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"image encoder checkpoint not found: {path}")
    raw = _torch_load(path)
    if isinstance(raw, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = raw.get(key)
            if isinstance(candidate, Mapping):
                raw = candidate
                break
    if not isinstance(raw, Mapping):
        raise ValueError(f"unsupported image-encoder checkpoint: {path}")
    tensors = {
        str(key): value for key, value in raw.items() if isinstance(value, torch.Tensor)
    }
    for prefix in _HF_VISION_PREFIXES:
        vision = {
            key.removeprefix(prefix): value
            for key, value in tensors.items()
            if key.startswith(prefix)
        }
        if vision:
            # Some serialized HF CLIP bases retain this deterministic cache,
            # while the current runtime registers it as a non-persistent
            # buffer.  It is not a model parameter and must not enter task
            # vectors or compact encoder checkpoints.
            vision.pop("embeddings.position_ids", None)
            return vision
    if tensors and all(key.startswith("encoder.") for key in tensors):
        tensors = {
            key.removeprefix("encoder."): value for key, value in tensors.items()
        }
    tensors.pop("embeddings.position_ids", None)
    return tensors


class _TorchStateStore(TensorStore):
    """Named tensors from a compact image-only torch state dict."""

    def __init__(self, tensors: Mapping[str, torch.Tensor] | None = None):
        self.tensors = dict(tensors or {})

    def keys(self) -> Iterable[str]:
        return self.tensors.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        self.tensors[key] = tensor.detach().cpu().contiguous()


class _ImageEncoderCheckpointAdapter:
    """Adapter that projects complete CLIP bases to image-encoder tensors."""

    @contextmanager
    def read(self, checkpoint: CheckpointRef):
        yield _TorchStateStore(_image_encoder_state(checkpoint))

    @contextmanager
    def write(self, output: CheckpointRef, *, template: CheckpointRef):
        del template
        store = _TorchStateStore()
        try:
            yield store
        except BaseException:
            raise
        else:
            path = _checkpoint_file(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(store.tensors, path)


def checkpoint_adapter(config: Mapping[str, Any]) -> _ImageEncoderCheckpointAdapter:
    """Common-runner hook used for TA/WA/TIES over image encoder tensors only."""

    del config
    return _ImageEncoderCheckpointAdapter()


def _ensure_batch_task(model: ClipVisionBundle, batch: Mapping[str, Any]) -> str:
    task = canonical_task(str(batch["task"]))
    model.ensure_task(task)
    return task


def register_loader(model: ClipVisionBundle, loader: DataLoader) -> None:
    """Attach loader metadata to a generic runner's model without a special case."""

    model.register_task(loader.smat_task, loader.smat_classnames)  # type: ignore[attr-defined]


def named_parameters(
    config: Mapping[str, Any], model: ClipVisionBundle
) -> list[tuple[str, torch.nn.Parameter]]:
    """Common-runner hook: optimizer and perturbations see vision parameters only."""

    del config
    return list(model.encoder.named_parameters())


def loss(
    model: ClipVisionBundle, batch: Mapping[str, Any], device: str | torch.device
) -> torch.Tensor:
    """Cross entropy over the frozen task-specific zero-shot CLIP head."""

    # The head belongs to the underlying bundle, but encoder features must go
    # through DDP.forward when the runner wraps it.  Calling ``owner.logits``
    # would bypass DDP's reducer and leave image-encoder gradients local.
    owner = getattr(model, "module", model)
    owner._head_device = torch.device(device)
    task = _ensure_batch_task(owner, batch)
    head = owner.heads[task] if task in owner.heads else owner._build_head(task)
    images = batch["images"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    logits = head(model(images))
    temperature = getattr(owner, "loss_temperature", 1.0)
    if temperature != 1.0:
        logits = logits / temperature
    objective = F.cross_entropy(
        logits,
        labels,
        label_smoothing=getattr(owner, "label_smoothing", 0.0),
    )
    # Cancel the explicit 1/T gradient factor; the softened probabilities remain.
    return objective if temperature == 1.0 else temperature * objective


def save_model(model: ClipVisionBundle, output: str | Path) -> Path:
    """Persist precisely the image encoder state dict, never a text head/tower."""

    output = _checkpoint_file(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # This is deliberately a plain image-encoder state dict.  It is both the
    # expert artifact and the tensor namespace consumed by merge kernels.
    torch.save(model.encoder.state_dict(), output)
    return output


class _ClipTaskEvaluator:
    """One formal CLIP evaluator reused across checkpoints for one task only."""

    def __init__(
        self,
        config: Mapping[str, Any],
        task: str,
        max_samples: int | None,
    ) -> None:
        self.config = config
        self.task = canonical_task(task)
        self.split = str(config["evaluation"]["split"])
        self.device = _device(config)
        self.model = build_model(config, training=False)
        self.loader = build_loader(
            config, self.task, split=self.split, max_samples=max_samples
        )
        register_loader(self.model, self.loader)
        self.model.eval()
        # A bundle deliberately releases its text tower after one task head.
        # This session owns exactly one task, so building it once preserves that
        # contract while allowing every image-encoder checkpoint to reuse it.
        self.model._build_head(self.task)

    def evaluate(self, checkpoint: str | Path, output: str | Path) -> float:
        state = _image_encoder_state(checkpoint)
        self.model.encoder.load_state_dict(state, strict=True)
        del state
        correct = total = 0
        for batch in self.loader:
            labels = batch["labels"].to(self.device, non_blocking=True)
            predictions = self.model.logits(
                batch["task"], batch["images"].to(self.device, non_blocking=True)
            ).argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
        accuracy = correct / total if total else 0.0
        payload = {
            "protocol": "clip8",
            "provider": "fusionbench_hf",
            "task": self.task,
            "split": self.split,
            "checkpoint": str(Path(checkpoint)),
            "top1": accuracy,
            "correct": correct,
            "num_samples": total,
        }
        atomic_write_json(Path(output), payload)
        return accuracy


@torch.inference_mode()
def evaluate_many(
    config: Mapping[str, Any],
    task: str,
    requests: Sequence[tuple[str | Path, str | Path]],
    max_samples: int | None = None,
) -> list[float]:
    """Evaluate several checkpoints on one task through one formal session."""

    if not requests:
        return []
    evaluator = _ClipTaskEvaluator(config, task, max_samples)
    return [evaluator.evaluate(checkpoint, output) for checkpoint, output in requests]


def evaluate(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    task: str,
    output: str | Path,
    max_samples: int | None = None,
) -> float:
    """Evaluate one image-encoder checkpoint through the reusable formal path."""

    return evaluate_many(config, task, ((checkpoint, output),), max_samples)[0]
