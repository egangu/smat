"""Fixed-parameter merging and complete evaluation of an expert suite.

Training configuration owns expert locations. ``merging`` owns only merger
parameters; changing it never requires copying or retraining experts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

from . import runtime

METHODS = ("wa", "ta", "ties", "dare", "della")


def portable(value):
    """Record portable paths relative to user-configured asset roots."""
    if isinstance(value, str):
        for key in ("DATA_ROOT", "MODEL_ROOT", "OUTPUT_ROOT"):
            prefix = os.environ.get(key)
            if prefix:
                value = value.replace(prefix, "${" + key + "}")
        return value
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def merge_settings(config):
    supplied = config.get("merging", {})
    unknown = set(supplied) - {
        "scale",
        "seed",
        "ties_retention",
        "dare_drop",
        "della_drop",
        "della_window",
        "ties_scale",
        "dare_scale",
        "della_scale",
    }
    if unknown:
        raise ValueError(f"unknown merging parameters: {sorted(unknown)}")
    settings = dict(
        scale=0.3,
        seed=42,
        ties_retention=0.2,
        dare_drop=0.5,
        della_drop=0.3,
        della_window=0.14,
    )
    settings.update(supplied)
    if not all(math.isfinite(settings[k]) for k in settings):
        raise ValueError("merging parameters must be finite")
    if settings["scale"] <= 0 or not 0 < settings["ties_retention"] <= 1:
        raise ValueError("require scale>0 and TIES retention in (0,1]")
    if any(
        settings.get(method + "_scale", settings["scale"]) <= 0
        for method in ("ties", "dare", "della")
    ):
        raise ValueError("per-method merge scales must be positive")
    if not 0 <= settings["dare_drop"] < 1:
        raise ValueError("DARE drop must be in [0,1)")
    lo = 1 - settings["della_drop"] - settings["della_window"] / 2
    hi = 1 - settings["della_drop"] + settings["della_window"] / 2
    if not 0 < lo <= hi < 1:
        raise ValueError("DELLA retention probability must stay inside (0,1)")
    if not isinstance(settings["seed"], int):
        raise ValueError("merging seed must be an integer")
    return settings


def method_settings(method, settings):
    if method == "base":
        return {}
    if method not in METHODS:
        raise ValueError(f"unknown merger: {method}")
    if method == "wa":
        return {}
    keys = ["scale"]
    if method in ("ties", "dare", "della"):
        keys += ["seed"]
    keys += {
        "ties": ["ties_retention"],
        "dare": ["dare_drop"],
        "della": ["della_drop", "della_window"],
    }.get(method, [])
    parameters = {k: settings[k] for k in keys}
    parameters["scale"] = settings.get(method + "_scale", settings["scale"])
    return parameters


def _identity(config, method, settings):
    identity = dict(
        method=method,
        parameters=method_settings(method, settings),
        experiment=config["experiment"],
        base_model=config["base_model"],
        expert_root=str(runtime.work_root(config)),
        name=config["name"],
        tasks=list(runtime.task_names(config)),
    )
    if "expert_overrides" in config:
        identity["expert_overrides"] = {
            t: str(runtime.expert_checkpoint(config, t))
            for t in runtime.task_names(config)
        }
    return portable(identity)


def merge_suite(config, output, methods=METHODS):
    from .merge import task_arithmetic, weight_average
    from .merge_sparse import sparse_merge

    settings = merge_settings(config)
    output = Path(output)
    module = runtime.experiment_module(config)
    adapter = runtime.checkpoint_adapter(module, config)
    tasks = runtime.task_names(config)
    experts = [
        runtime.checkpoint_path(module, config, runtime.expert_checkpoint(config, t))
        for t in tasks
    ]
    for task in tasks:
        if not (runtime.expert_checkpoint(config, task) / "metadata.json").is_file():
            raise FileNotFoundError(f"unfinished expert: {task}")
    paths = {}
    with runtime.exclusive_directory_lock(output, ".merge.lock"):
        for method in methods:
            identity = _identity(config, method, settings)
            destination = output / "merged" / method
            receipt = output / (method + "-merge.json")
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous["identity"] != identity:
                    raise ValueError(
                        f"merger configuration changed; use a new output directory: {receipt}"
                    )
                if runtime.checkpoint_exists(destination):
                    paths[method] = runtime.checkpoint_path(module, config, destination)
                    continue
            elif destination.exists():
                raise ValueError(f"checkpoint lacks a merge receipt: {destination}")
            partial = runtime.partial_directory(destination)
            partial.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = runtime.checkpoint_path(module, config, partial)
            started = time.monotonic()
            # A previous merger's temporary tensors can evict source pages.
            # Sequential reads avoid slow tensor-wise mmap faults on shared storage.
            if config["experiment"] == "trace_llm" and method != "base":
                from .checkpoints import warm_checkpoint_files

                size = warm_checkpoint_files(
                    [runtime.base_checkpoint(config), *experts]
                )
                print(
                    f"{method} input pre-read: {size / 1e9:.1f} GB in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
            if method == "base":
                task_arithmetic(
                    adapter, runtime.base_checkpoint(config), [], checkpoint, scale=0.0
                )
            elif method == "wa":
                weight_average(adapter, experts, checkpoint)
            elif method == "ta":
                task_arithmetic(
                    adapter,
                    runtime.base_checkpoint(config),
                    experts,
                    checkpoint,
                    scale=settings["scale"],
                )
            else:
                sparse_merge(
                    adapter,
                    runtime.base_checkpoint(config),
                    experts,
                    checkpoint,
                    method=method,
                    **method_settings(method, settings),
                )
            partial.rename(destination)
            runtime.atomic_write_json(
                receipt,
                dict(
                    identity=identity,
                    seconds=time.monotonic() - started,
                    implementation="smat.suite.v1",
                ),
            )
            paths[method] = runtime.checkpoint_path(module, config, destination)
    return paths


def _evaluation_tasks(config):
    tasks = runtime.task_names(config)
    selected = config.get("evaluation", {}).get("tasks", tasks)
    if (
        not isinstance(selected, (list, tuple))
        or not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(tasks)
    ):
        raise ValueError(
            "evaluation.tasks must be a nonempty unique subset of configured tasks"
        )
    return tuple(selected)


def evaluate_suite(config, output, methods=("expert", *METHODS)):
    # Reuse the existing task batching and SGLang lifecycle. The scheduler
    # launches this function in the evaluation runtime, not in the trainer.
    from .eval.suite import (
        EvalTarget,
        _eval_clip,
        _eval_trace,
        _clip_metrics,
        _trace_metrics,
        _expert_targets,
    )

    methods = tuple(methods)
    if not methods or set(methods) - {"expert", "base", *METHODS}:
        raise ValueError("invalid evaluation methods")
    settings = merge_settings(config)
    output = Path(output)
    tasks = _evaluation_tasks(config)
    module = runtime.experiment_module(config)
    repeats = int(config["evaluation"].get("repeats", 3))
    with runtime.exclusive_directory_lock(output, ".evaluation.lock"):
        recorded = {k: v for k, v in config.items() if k != "_config_path"}
        config_path = output / "evaluation-config.json"
        if config_path.exists() and portable(
            json.loads(config_path.read_text())
        ) != portable(recorded):
            raise ValueError(
                "evaluation configuration changed; use a new output directory"
            )
        runtime.atomic_write_json(config_path, recorded)
        targets = []
        for method in methods:
            if method == "expert":
                targets.extend(_expert_targets(config, tasks, output / "evaluations"))
                continue
            receipt = json.loads((output / (method + "-merge.json")).read_text())
            if receipt["identity"] != _identity(config, method, settings):
                raise ValueError(f"merge receipt does not match config: {method}")
            targets.append(
                EvalTarget(
                    method,
                    runtime.checkpoint_path(module, config, output / "merged" / method),
                    tasks,
                    output / "evaluations" / method,
                )
            )
        started = time.monotonic()
        if config["experiment"] == "clip8_vit":
            _eval_clip(config_path, targets)
            scores = {
                m: _clip_metrics(output / "evaluations" / m, tasks) for m in methods
            }
        else:
            _eval_trace(
                config_path,
                config,
                targets,
                output / "logs",
                repeats,
                config["evaluation"].get("max_samples"),
                int(os.environ.get("SMAT_EVAL_PORT", "19000")),
            )
            scores = {
                m: _trace_metrics(output / "evaluations" / m, tasks, repeats)
                for m in methods
            }
        summary_path = output / "summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        scores = {**previous.get("scores", {}), **scores}
        summary = dict(
            name=config["name"],
            scores=scores,
            merging=settings,
            evaluation=config["evaluation"],
            tasks=list(tasks),
            eval_seconds=previous.get("eval_seconds", 0) + time.monotonic() - started,
            training_metadata={
                t: json.loads(
                    (runtime.expert_checkpoint(config, t) / "metadata.json").read_text()
                )
                for t in tasks
            },
            merging_receipts={
                m: json.loads((output / (m + "-merge.json")).read_text())
                for m in scores
                if m != "expert"
            },
        )
        runtime.atomic_write_json(summary_path, summary)
        return summary
