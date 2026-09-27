#!/usr/bin/env python3
"""Evaluate fixed expert and merged checkpoints with task batching."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from smat import runtime
from smat.config import load_config


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _trace_metrics(
    directory: Path, tasks: tuple[str, ...], repeats: int
) -> dict[str, Any]:
    macro_scores = []
    task_scores = {task: [] for task in tasks}
    for repeat in range(1, repeats + 1):
        scores = []
        for task in tasks:
            path = directory / f"repeat_{repeat}" / f"{task}.json"
            value = float(json.loads(path.read_text())["metric"])
            scores.append(value)
            task_scores[task].append(value)
        macro_scores.append(statistics.mean(scores))
    mean, std = _mean_std(macro_scores)
    return {
        "macro_mean": mean,
        "macro_std": std,
        "tasks": {
            task: dict(zip(("mean", "std"), _mean_std(values)))
            for task, values in task_scores.items()
        },
    }


def _clip_metrics(directory: Path, tasks: tuple[str, ...]) -> dict[str, Any]:
    scores = {
        task: float(
            json.loads((directory / f"{task}.json").read_text(encoding="utf-8"))["top1"]
        )
        for task in tasks
    }
    return {"macro": statistics.mean(scores.values()), "tasks": scores}


@dataclass(frozen=True)
class EvalTarget:
    label: str
    checkpoint: Path
    tasks: tuple[str, ...]
    output: Path


@dataclass(frozen=True)
class ClipRequest:
    checkpoint: Path
    output: Path


@dataclass(frozen=True)
class ClipJob:
    task: str
    requests: tuple[ClipRequest, ...]
    manifest: Path


def _expert_targets(
    config: dict[str, Any], tasks: tuple[str, ...], output_root: Path
) -> tuple[EvalTarget, ...]:
    module = runtime.experiment_module(config)
    targets = tuple(
        EvalTarget(
            f"expert-{task}",
            runtime.checkpoint_path(
                module, config, runtime.expert_checkpoint(config, task)
            ),
            (task,),
            output_root / "expert",
        )
        for task in tasks
    )
    missing = [
        str(target.checkpoint) for target in targets if not target.checkpoint.exists()
    ]
    if missing:
        raise FileNotFoundError("missing expert checkpoints: " + ", ".join(missing))
    return targets


def _eval_clip(
    config_path: Path,
    targets: Sequence[EvalTarget],
) -> None:
    by_task: dict[str, list[ClipRequest]] = {}
    output_roots: set[Path] = set()
    for target in targets:
        for task in target.tasks:
            output = target.output / f"{task}.json"
            if not runtime.json_file_complete(output):
                output.parent.mkdir(parents=True, exist_ok=True)
                by_task.setdefault(task, []).append(
                    ClipRequest(target.checkpoint, output)
                )
                output_roots.add(output.parent.parent)
    if len(output_roots) > 1:
        raise ValueError("CLIP evaluation targets must share one output root")
    jobs: queue.Queue[ClipJob] = queue.Queue()
    if output_roots:
        request_root = output_roots.pop() / ".clip_requests"
        for task, requests in by_task.items():
            jobs.put(
                ClipJob(
                    task=task,
                    requests=tuple(requests),
                    manifest=request_root / f"{task}.json",
                )
            )
    gpus = tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if value.strip()
    )
    if not gpus:
        raise ValueError("CUDA_VISIBLE_DEVICES must name at least one GPU")
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                runtime.atomic_write_json(
                    job.manifest,
                    {
                        "task": job.task,
                        "requests": [
                            {
                                "checkpoint": str(request.checkpoint),
                                "output": str(request.output),
                            }
                            for request in job.requests
                        ],
                    },
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--clip-worker",
                        str(config_path),
                        str(job.manifest),
                    ],
                    cwd=REPO,
                    env=environment,
                    check=True,
                )
            except BaseException as error:
                with lock:
                    errors.append(error)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def _run_clip_worker(config_path: Path, request_path: Path) -> None:
    """Run one task's checkpoint requests inside one reusable CLIP process."""

    config = load_config(config_path)
    if str(config["experiment"]) != "clip8_vit":
        raise ValueError("--clip-worker only supports clip8_vit")
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CLIP request manifest must be a JSON object")
    task = payload.get("task")
    raw_requests = payload.get("requests")
    if not isinstance(task, str) or task not in runtime.task_names(config):
        raise ValueError("CLIP request manifest has an unknown task")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("CLIP request manifest has no requests")
    requests: list[tuple[Path, Path]] = []
    for item in raw_requests:
        if not isinstance(item, dict):
            raise ValueError("CLIP request must be a JSON object")
        checkpoint = item.get("checkpoint")
        output = item.get("output")
        if not isinstance(checkpoint, str) or not isinstance(output, str):
            raise ValueError("CLIP request needs checkpoint and output paths")
        requests.append((Path(checkpoint), Path(output)))
    module = runtime.experiment_module(config)
    evaluate_many = runtime.require(module, "evaluate_many")
    evaluate_many(
        config, task, requests, config.get("evaluation", {}).get("max_samples")
    )


def _eval_trace(
    config_path: Path,
    config: dict[str, Any],
    targets: Sequence[EvalTarget],
    log_root: Path,
    repeats: int,
    max_samples: int | None,
    port_base: int,
) -> None:
    if importlib.util.find_spec("sglang") is None:
        raise RuntimeError(
            "SGLang is unavailable; install the evaluation environment with python -m pip install -r requirements/eval.txt"
        )
    from smat.eval.trace_server import start, stop, stop_all

    jobs: queue.Queue[EvalTarget] = queue.Queue()
    for target in targets:
        complete = all(
            runtime.json_file_complete(
                target.output / f"repeat_{repeat}" / f"{task}.json"
            )
            for repeat in range(1, repeats + 1)
            for task in target.tasks
        )
        if not complete:
            jobs.put(target)
    gpus = tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if value.strip()
    )
    if not gpus:
        raise ValueError("CUDA_VISIBLE_DEVICES must name at least one GPU")
    errors: list[BaseException] = []
    lock = threading.Lock()
    concurrency = int(config["evaluation"]["concurrency"])
    base_seed = int(config["seed"])

    def worker(index: int, gpu: str) -> None:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                target = jobs.get_nowait()
            except queue.Empty:
                return
            process = None
            try:
                process, endpoint = start(
                    target.checkpoint,
                    port_base + index,
                    concurrency,
                    log_root / f"eval-ta-{config['name']}-{target.label}.log",
                    environment,
                )
                for repeat in range(1, repeats + 1):
                    for task in target.tasks:
                        output = target.output / f"repeat_{repeat}" / f"{task}.json"
                        if runtime.json_file_complete(output):
                            continue
                        command = [
                            sys.executable,
                            "-m",
                            "smat",
                            "eval",
                            str(config_path),
                            str(target.checkpoint),
                            task,
                            str(output),
                            "--endpoint",
                            endpoint,
                            "--concurrency",
                            str(concurrency),
                            "--seed",
                            str(base_seed + repeat - 1),
                        ]
                        if max_samples is not None:
                            command.extend(("--max-samples", str(max_samples)))
                        subprocess.run(command, cwd=REPO, env=environment, check=True)
            except BaseException as error:
                with lock:
                    errors.append(error)
            finally:
                if process is not None:
                    stop(process)

    threads = [
        threading.Thread(target=worker, args=(index, gpu))
        for index, gpu in enumerate(gpus)
    ]
    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {value: signal.getsignal(value) for value in handled_signals}

    def terminate(signum: int, _frame: object) -> None:
        stop_all()
        raise SystemExit(128 + signum)

    for value in handled_signals:
        signal.signal(value, terminate)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        stop_all()
        for value, handler in previous_handlers.items():
            signal.signal(value, handler)
    if errors:
        raise errors[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal CLIP evaluation worker")
    parser.add_argument("--clip-worker", action="store_true", required=True)
    parser.add_argument("config", type=Path)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    _run_clip_worker(args.config, args.request)
