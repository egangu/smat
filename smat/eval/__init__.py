"""Single-checkpoint evaluation through the formal experiment adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import runtime


def evaluate(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    task: str,
    output: str | Path,
    max_samples: int | None = None,
    *,
    endpoint: str | None = None,
    concurrency: int = 32,
    seed: int | None = None,
) -> float:
    """Evaluate one checkpoint using the experiment's one formal metric path."""

    if task not in runtime.task_names(config):
        raise ValueError(f"task {task!r} is not in this experiment")
    module = runtime.experiment_module(config)
    checkpoint_path = Path(checkpoint).expanduser()
    output_path = Path(output).expanduser()
    if runtime.experiment_name(config) == "trace_llm":
        if not endpoint:
            raise ValueError("trace_llm evaluation requires --endpoint")
        return float(
            runtime.require(module, "evaluate")(
                config,
                checkpoint_path,
                task,
                output_path,
                max_samples,
                endpoint=endpoint,
                concurrency=concurrency,
                seed=seed,
            )
        )
    if endpoint is not None:
        raise ValueError("--endpoint is only valid for trace_llm")
    if seed is not None:
        raise ValueError("--seed is only valid for trace_llm")
    return float(
        runtime.require(module, "evaluate")(
            config, checkpoint_path, task, output_path, max_samples
        )
    )
