"""TRACE LLM experiment backend used by the generic SMAT runner.

This module contains experiment-specific concerns only: TRACE JSON formatting,
Llama loading, generation and metrics.  Optimizer/perturbation updates live in
``smat.train.updates`` so FT and SMAT share one training loop.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import aiohttp
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..eval.metrics import task_metric
from ..runtime import atomic_write_json

# These are the released TRACE task schedules and decoding settings.  They are
# intentionally experiment data rather than a generic runner configuration.
TASKS: dict[str, dict[str, Any]] = {
    "C-STANCE": {"epochs": 5, "generation": {"max_new_tokens": 20, "do_sample": False}},
    "FOMC": {"epochs": 3, "generation": {"max_new_tokens": 20, "do_sample": False}},
    "MeetingBank": {
        "epochs": 7,
        "generation": {"max_new_tokens": 150, "temperature": 0.7, "do_sample": True},
    },
    "ScienceQA": {
        "epochs": 3,
        "generation": {"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
    },
    "NumGLUE-cm": {
        "epochs": 5,
        "generation": {"max_new_tokens": 20, "do_sample": False},
    },
    "NumGLUE-ds": {
        "epochs": 5,
        "generation": {"max_new_tokens": 20, "do_sample": False},
    },
    "20Minuten": {
        "epochs": 7,
        "generation": {"max_new_tokens": 150, "temperature": 0.7, "do_sample": True},
    },
}


def _device(config: Mapping[str, Any]) -> torch.device:
    configured = config.get("device")
    if configured is not None:
        return torch.device(configured)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _model_source(
    config: Mapping[str, Any], checkpoint: str | Path | None
) -> str | Path:
    return checkpoint if checkpoint is not None else str(config["base_model"])


def task_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(config["tasks"])


def task_spec(task: str) -> Mapping[str, Any]:
    return TASKS[task]


def generation_config(task: str) -> dict[str, Any]:
    return dict(TASKS[task]["generation"])


def task_epochs(config: Mapping[str, Any], task: str) -> int:
    del config
    return int(TASKS[task]["epochs"])


def _rows(
    config: Mapping[str, Any], task: str, split: str, max_samples: int | None = None
) -> list[dict[str, Any]]:
    path = Path(config["data_root"]) / task / f"{split}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows[:max_samples] if max_samples is not None else rows


def _tokenizer(config: Mapping[str, Any], checkpoint: str | Path | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        _model_source(config, checkpoint), local_files_only=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def build_loader(
    config: Mapping[str, Any],
    task: str,
    split: str = "train",
    max_samples: int | None = None,
) -> DataLoader:
    """Build the released TRACE causal-LM data loader.

    The canonical protocol dynamically pads each batch to a multiple of eight.
    Labels retain prompt tokens, matching the public training implementation.
    """

    if split != "train":
        raise ValueError("TRACE build_loader only supports the canonical train split")
    train = config["train"]
    tokenizer = _tokenizer(config)
    rows = _rows(config, task, split, max_samples)
    max_length = int(train["max_length"])
    if train["padding"] != "dynamic":
        raise ValueError("TRACE protocol requires train.padding=dynamic")
    append_eos = train.get("append_eos", True)
    if not isinstance(append_eos, bool):
        raise ValueError("train.append_eos must be a boolean")
    suffix = tokenizer.eos_token if append_eos else ""
    if suffix is None:
        raise ValueError("train.append_eos requires a tokenizer EOS token")
    texts = [f"{row['prompt']}\n{row['answer']}{suffix}" for row in rows]
    encoded = tokenizer(texts, truncation=True, max_length=max_length)
    dataset = list(zip(encoded.input_ids, encoded.attention_mask))

    def collate_fn(batch):
        padded = tokenizer.pad(
            [{"input_ids": ids, "attention_mask": mask} for ids, mask in batch],
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        labels = padded.input_ids.clone()
        labels[padded.attention_mask == 0] = -100
        return padded.input_ids, padded.attention_mask, labels

    return DataLoader(
        dataset,
        batch_size=int(train["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        pin_memory=True,
        collate_fn=collate_fn,
    )


def build_model(
    config: Mapping[str, Any],
    checkpoint: str | Path | None = None,
    training: bool = True,
):
    """Load the canonical Llama model in BF16 with SDPA attention."""

    source = _model_source(config, checkpoint)
    device_map = config.get("train", {}).get("device_map") if training else None
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    if device_map is None:
        model = model.to(_device(config))
    else:
        from accelerate import dispatch_model

        dispatch_model(
            model,
            device_map,
            main_device=_device(config),
            skip_keys=["logits", "loss"],
            force_hooks=True,
        )
    # Keep the matching tokenizer with the model so the generic runner can use
    # save_model(model, output) without knowing a backend-specific second item.
    model._smat_tokenizer = _tokenizer(config, source)  # type: ignore[attr-defined]

    if training:
        model.config.use_cache = False
        if bool(config["train"]["gradient_checkpointing"]):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model.train()
    else:
        model.eval()
    return model


def loss(model, batch: Any, device: str | torch.device) -> torch.Tensor:
    """Move one TRACE batch and return the causal-LM loss tensor."""

    target = torch.device(device)
    if isinstance(batch, Mapping):
        prepared = {
            key: value.to(target, non_blocking=True) for key, value in batch.items()
        }
    else:
        input_ids, attention_mask, labels = batch
        prepared = {
            "input_ids": input_ids.to(target, non_blocking=True),
            "attention_mask": attention_mask.to(target, non_blocking=True),
            "labels": labels.to(target, non_blocking=True),
        }
    return model(**prepared).loss.to(target)


def save_model(model, output: str | Path) -> Path:
    """Write one HF-safe checkpoint, preserving the attached tokenizer."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.safetensors")):
        raise FileExistsError(f"refusing to replace existing checkpoint in {output}")
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    tokenizer = getattr(model, "_smat_tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    return output


def _result(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    task: str,
    output: str | Path,
    rows: list[dict[str, Any]],
    predictions: list[str],
    seed: int,
    elapsed_seconds: float,
    **metadata: Any,
) -> float:
    inputs = [row["input"] if "input" in row else row["prompt"] for row in rows]
    metric = task_metric(
        task,
        inputs,
        predictions,
        [row["answer"] for row in rows],
    )
    payload: dict[str, Any] = {
        "model": str(Path(checkpoint).resolve()),
        "task": task,
        "seed": seed,
        "num_samples": len(rows),
        "metric": metric,
        "elapsed_seconds": elapsed_seconds,
        "backend": "sglang",
        **metadata,
    }
    if config["evaluation"]["save_samples"]:
        payload["samples"] = [
            {
                "prompt": row["prompt"],
                "prediction": prediction,
                "reference": row["answer"],
            }
            for row, prediction in zip(rows, predictions)
        ]
    atomic_write_json(Path(output), payload)
    shown = payload if "samples" not in payload else {**payload, "samples": "saved"}
    print(json.dumps(shown, ensure_ascii=False, indent=2), flush=True)
    return float(payload["metric"])


def _sampling_params(generation: Mapping[str, Any], seed: int) -> dict[str, Any]:
    params = {key: value for key, value in generation.items() if key != "do_sample"}
    if generation.get("do_sample", False):
        params["sampling_seed"] = seed
    else:
        params["temperature"] = 0
    return params


async def _sglang_generate_many(
    endpoint: str,
    prompts: list[str],
    generation: Mapping[str, Any],
    seed: int,
    concurrency: int,
) -> list[str]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def one(index: int, prompt: str) -> str:
            payload = {
                "text": prompt,
                "sampling_params": _sampling_params(generation, seed + index),
            }
            async with semaphore:
                async with session.post(
                    f"{endpoint.rstrip('/')}/generate", json=payload
                ) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(
                            f"SGLang request {index} failed with HTTP {response.status}: {body[:500]}"
                        )
                    data = json.loads(body)
                    if not isinstance(data.get("text"), str):
                        raise RuntimeError(
                            f"SGLang request {index} returned no text: {body[:500]}"
                        )
                    return data["text"]

        return list(
            await asyncio.gather(
                *(one(index, prompt) for index, prompt in enumerate(prompts))
            )
        )


def evaluate(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    task: str,
    output: str | Path,
    max_samples: int | None = None,
    *,
    endpoint: str,
    concurrency: int = 32,
    seed: int | None = None,
) -> float:
    """Evaluate TRACE through the required SGLang endpoint."""

    evaluation = config["evaluation"]
    resolved_seed = int(seed if seed is not None else config["seed"])
    split = str(evaluation["split"])
    if evaluation.get("numglue_cm_include_test", False):
        raise ValueError(
            "The paper evaluates NumGLUE-cm on 41 eval examples without test concatenation"
        )
    rows = _rows(config, task, split, max_samples)
    if not rows:
        raise ValueError(f"TRACE evaluation split is empty for task {task}")
    started = time.time()
    predictions = [
        prediction.strip()
        for prediction in asyncio.run(
            _sglang_generate_many(
                endpoint,
                [row["prompt"] for row in rows],
                generation_config(task),
                resolved_seed,
                concurrency,
            )
        )
    ]
    return _result(
        config,
        checkpoint,
        task,
        output,
        rows,
        predictions,
        resolved_seed,
        time.time() - started,
        split=split,
        numglue_cm_include_test=False,
        endpoint=endpoint,
        concurrency=concurrency,
        nonempty_fraction=sum(bool(prediction) for prediction in predictions)
        / len(predictions),
    )
