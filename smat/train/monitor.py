"""Low-frequency training telemetry; no per-step CUDA synchronization."""

import json
import time
from pathlib import Path

from ..runtime import atomic_write_json


class TrainingMonitor:
    def __init__(self, directory: Path, *, task: str, total_steps: int, started: float):
        self.directory = directory
        self.task, self.total_steps, self.started = task, total_steps, started
        self.previous_time, self.previous_step = started, 0
        self.latest = {}

    def record(
        self,
        step: int,
        *,
        loss: float,
        lr: float,
        peak_bytes: int | None,
        smat: dict | None = None,
    ) -> dict:
        now = time.monotonic()
        elapsed = now - self.started
        window_steps = step - self.previous_step
        window_seconds = now - self.previous_time
        seconds_per_step = window_seconds / window_steps if window_steps else None
        row = dict(
            state="training",
            updated=time.time(),
            task=self.task,
            step=step,
            total_steps=self.total_steps,
            loss=loss,
            lr=lr,
            elapsed_seconds=elapsed,
            window_steps=window_steps,
            window_seconds=window_seconds,
            seconds_per_step=seconds_per_step,
            eta_seconds=seconds_per_step * (self.total_steps - step)
            if seconds_per_step
            else None,
            peak_memory_allocated_bytes=peak_bytes,
            smat=smat,
        )
        # Wall time includes loading batches and initial warmup; not kernel cost.
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        atomic_write_json(self.directory / "progress.json", row)
        self.previous_time, self.previous_step, self.latest = now, step, row
        return row

    def finish(self, *, loop_seconds: float, save_seconds: float) -> dict:
        row = dict(
            self.latest,
            state="complete",
            updated=time.time(),
            eta_seconds=0.0,
            loop_seconds=loop_seconds,
            checkpoint_save_seconds=save_seconds,
            timing_scope="training loop wall time including batch loading, warmup and logging; "
            "checkpoint saving reported separately; not an isolated kernel benchmark",
        )
        atomic_write_json(self.directory / "progress.json", row)
        return row
