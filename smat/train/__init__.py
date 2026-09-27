"""Common distributed expert-training loop and its update rules."""

from __future__ import annotations

import atexit
import gc
import math
import os
import platform
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .. import runtime
from .optimizers import build_optimizer
from .updates import build_stepper, validate_settings


@dataclass(frozen=True)
class DistributedContext:
    """The small amount of process state the common trainer needs for DDP."""

    local_rank: int = 0
    rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def _distributed_environment(values: Mapping[str, str]) -> DistributedContext:
    """Parse torchrun's environment without touching CUDA or process groups."""

    world_size = int(values.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext()
    local_rank = int(values["LOCAL_RANK"])
    return DistributedContext(
        local_rank=local_rank,
        rank=int(values.get("RANK", str(local_rank))),
        world_size=world_size,
    )


def _distributed_context() -> DistributedContext:
    """Initialize NCCL only for a multi-process ``torchrun`` invocation."""

    context = _distributed_environment(os.environ)
    if not context.enabled:
        return context
    if not torch.cuda.is_available():
        raise RuntimeError(
            "WORLD_SIZE>1 requires CUDA/NCCL; launch this command with torchrun on GPUs"
        )
    torch.cuda.set_device(context.local_rank)
    if not distributed.is_initialized():
        distributed.init_process_group(backend="nccl")
        atexit.register(_shutdown_distributed)
    return DistributedContext(
        local_rank=context.local_rank,
        rank=distributed.get_rank(),
        world_size=distributed.get_world_size(),
    )


def _shutdown_distributed() -> None:
    if distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()


def is_rank_zero() -> bool:
    """Whether this process should emit user-facing train output."""

    return not distributed.is_initialized() or distributed.get_rank() == 0


def _device(config: Mapping[str, Any]) -> str:
    configured = config.get("device")
    if configured is not None:
        return str(configured)
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _rank_config(
    config: Mapping[str, Any], context: DistributedContext
) -> Mapping[str, Any]:
    """Pin every torchrun process to its local device without changing v1 configs."""

    if not context.enabled:
        return config
    resolved = dict(config)
    resolved["device"] = f"cuda:{context.local_rank}"
    return resolved


def _distributed_loader(
    loader: DataLoader, context: DistributedContext, seed: int
) -> tuple[DataLoader, DistributedSampler | None]:
    """Replace only the train sampler while preserving adapter loader settings."""

    if not context.enabled:
        return loader, None
    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=seed,
        drop_last=loader.drop_last,
    )
    arguments: dict[str, Any] = {
        "batch_size": loader.batch_size,
        "sampler": sampler,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "generator": torch.Generator().manual_seed(seed + context.rank),
    }
    if loader.num_workers > 0:
        arguments["persistent_workers"] = loader.persistent_workers
        if loader.prefetch_factor is not None:
            arguments["prefetch_factor"] = loader.prefetch_factor
        if loader.multiprocessing_context is not None:
            arguments["multiprocessing_context"] = loader.multiprocessing_context
    if loader.pin_memory_device:
        arguments["pin_memory_device"] = loader.pin_memory_device
    result = DataLoader(loader.dataset, **arguments)
    for name, value in vars(loader).items():
        if name.startswith("smat_"):
            setattr(result, name, value)
    return result, sampler


def _ddp_model(model: Any, context: DistributedContext):
    if not context.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("train")
    if not isinstance(settings, Mapping) or not settings:
        raise ValueError("config needs a train object")
    validate_settings(settings)
    return settings


def _schedule(settings: Mapping[str, Any], total_steps: int):
    kind = settings.get("schedule")
    if not isinstance(kind, str):
        raise ValueError("train.schedule must be a string")
    warmup = int(settings.get("warmup_steps", 0))
    if warmup < 0:
        raise ValueError("warmup_steps must not be negative")

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        if kind == "linear":
            return max(0.0, 1.0 - progress)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        raise ValueError(f"unsupported learning-rate schedule: {kind}")

    return torch.optim.lr_scheduler.LambdaLR, factor


def _total_steps(
    module: Any,
    config: Mapping[str, Any],
    task: str,
    loader: Any,
    settings: Mapping[str, Any],
) -> tuple[int, int | None]:
    explicit = settings.get("max_steps")
    if explicit is not None:
        steps = int(explicit)
        if steps <= 0:
            raise ValueError("train.max_steps must be positive")
        return steps, None

    task_epochs = getattr(module, "task_epochs", None)
    if callable(task_epochs):
        epochs = int(task_epochs(config, task))
    else:
        overrides = settings.get("epochs_by_task", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("train.epochs_by_task must be an object")
        epochs = int(overrides.get(task, settings.get("epochs", 1)))
    if epochs <= 0:
        raise ValueError("training epochs must be positive")
    try:
        batches = len(loader)
    except TypeError as error:
        raise ValueError("an iterable training loader needs train.max_steps") from error
    if batches <= 0:
        raise ValueError(f"training loader is empty for task {task}")
    return batches * epochs, epochs


def _named_parameters(module: Any, config: Mapping[str, Any], model: Any):
    selector = getattr(module, "named_parameters", None)
    if callable(selector):
        return list(selector(config, model))
    return list(model.named_parameters())


def train(config: Mapping[str, Any], task: str, max_samples: int | None = None) -> Any:
    """Train exactly one expert with the common SMAT update loop."""
    if "expert_overrides" in config:
        raise ValueError("expert_overrides are read-only evaluation inputs")

    context = _distributed_context()
    run_config = _rank_config(config, context)
    module = runtime.experiment_module(run_config)
    if task not in runtime.task_names(run_config):
        raise ValueError(f"task {task!r} is not in this experiment")
    settings = _train_settings(run_config)
    output_dir = runtime.expert_checkpoint(config, task)
    if runtime.checkpoint_exists(output_dir):
        raise FileExistsError(f"checkpoint exists: {output_dir}")
    partial_dir = runtime.partial_directory(output_dir)
    partial_output = runtime.checkpoint_path(module, config, partial_dir)
    # Reserve output before loading model weights or starting training.
    if context.is_rank_zero:
        partial_dir.mkdir(parents=True)

    seed = int(config.get("seed", 42))
    _seed(seed + context.rank)
    model_first = bool(getattr(module, "MODEL_BEFORE_LOADER", False))
    if model_first:
        model = runtime.require(module, "build_model")(run_config, None, True)
        loader = runtime.require(module, "build_loader")(
            run_config, task, "train", max_samples
        )
    else:
        loader = runtime.require(module, "build_loader")(
            run_config, task, "train", max_samples
        )
        model = runtime.require(module, "build_model")(run_config, None, True)
    register_loader = getattr(module, "register_loader", None)
    if callable(register_loader):
        register_loader(model, loader)
    loader, sampler = _distributed_loader(loader, context, seed)
    total_steps, configured_epochs = _total_steps(
        module, run_config, task, loader, settings
    )
    model.train()
    training_model = _ddp_model(model, context)
    named_parameters = _named_parameters(module, run_config, model)
    loss_function = runtime.require(module, "loss")
    save_model = runtime.require(module, "save_model")
    optimizer = build_optimizer(settings, named_parameters)
    from .noise import block_linear_parameter_names, embedding_parameter_names

    stepper = build_stepper(
        settings,
        named_parameters,
        optimizer,
        seed,
        block_linear_names=block_linear_parameter_names(model, named_parameters),
        embedding_names=embedding_parameter_names(model, named_parameters),
    )
    scheduler_type, schedule_factor = _schedule(settings, total_steps)
    scheduler = scheduler_type(optimizer, schedule_factor)
    parameter_devices = sorted({p.device for _, p in named_parameters}, key=str)
    device = _device(run_config)
    use_cuda = torch.device(device).type == "cuda"
    max_grad_norm = settings.get("max_grad_norm")
    max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
    log_every = max(1, int(settings.get("log_every", 50)))
    if use_cuda:
        for parameter_device in parameter_devices:
            torch.cuda.synchronize(parameter_device)
        for parameter_device in parameter_devices:
            torch.cuda.reset_peak_memory_stats(parameter_device)
    started = time.monotonic()
    from .monitor import TrainingMonitor

    monitor = (
        TrainingMonitor(
            runtime.work_root(config) / "monitor" / runtime.run_name(config) / task,
            task=task,
            total_steps=total_steps,
            started=started,
        )
        if context.is_rank_zero and settings.get("monitor", True)
        else None
    )
    if sampler is not None:
        sampler.set_epoch(0)
    iterator = iter(loader)
    completed_epochs = 0
    for step in range(1, total_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            completed_epochs += 1
            if sampler is not None:
                sampler.set_epoch(completed_epochs)
            iterator = iter(loader)
            batch = next(iterator)
        loss = stepper.step(
            lambda: loss_function(training_model, batch, device),
            max_grad_norm=max_grad_norm,
        )
        scheduler.step()
        if context.is_rank_zero and (
            step == 1 or step % log_every == 0 or step == total_steps
        ):
            epoch = completed_epochs + 1
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite training loss at step {step}: {value}")
            lr = scheduler.get_last_lr()[0]
            row = (
                monitor.record(
                    step,
                    loss=value,
                    lr=lr,
                    peak_bytes=torch.cuda.max_memory_allocated(device)
                    if use_cuda
                    else None,
                    smat=stepper.smat.metadata() if hasattr(stepper, "smat") else None,
                )
                if monitor
                else None
            )
            print(
                f"task={task} epoch={epoch} step={step}/{total_steps} "
                f"loss={value:.5f} lr={lr:.3e}"
                + (
                    f" sec/step={row['seconds_per_step']:.4f} eta={row['eta_seconds']:.1f}s"
                    if row
                    else ""
                ),
                flush=True,
            )

    if context.enabled:
        distributed.barrier()
    if use_cuda:
        for parameter_device in parameter_devices:
            torch.cuda.synchronize(parameter_device)
    elapsed_seconds = time.monotonic() - started
    loop_elapsed_seconds = elapsed_seconds
    saved = runtime.checkpoint_path(module, config, output_dir)
    if context.is_rank_zero:
        save_started = time.monotonic()
        save_model(model, partial_output)
        save_seconds = time.monotonic() - save_started
        metadata: dict[str, Any] = {
            "experiment": runtime.experiment_name(config),
            "name": runtime.run_name(config),
            "task": task,
            "seed": seed,
            "steps": total_steps,
            "epochs": configured_epochs,
            "elapsed_seconds": elapsed_seconds,
            "loop_elapsed_seconds": loop_elapsed_seconds,
            "checkpoint_save_seconds": save_seconds,
            "framework": "smat",
            "device": device,
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
            },
            "train": dict(settings),
        }
        if use_cuda:
            metadata["gpu"] = torch.cuda.get_device_name(device)
            metadata["parameter_devices"] = [str(d) for d in parameter_devices]
            metadata["peak_memory_allocated_bytes_by_device"] = {
                str(d): torch.cuda.max_memory_allocated(d) for d in parameter_devices
            }
            metadata["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            metadata["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
        if hasattr(stepper, "smat"):
            metadata["smat"] = stepper.smat.metadata()
        if getattr(stepper, "fused_smat", None) is not None:
            metadata["fused_smat"] = {
                "steps": stepper.fused_smat.steps,
                "rng": "triton" if stepper.fused_smat.seeded else "torch",
                "base_bytes": sum(
                    p.numel() * p.element_size()
                    for p in stepper.self_path.base.values()
                ),
                "base_cuda_bytes": sum(
                    p.numel() * p.element_size()
                    for p in stepper.self_path.base.values()
                    if p.is_cuda
                ),
            }
        if context.enabled:
            metadata["distributed"] = {
                "world_size": context.world_size,
                "rank": context.rank,
                "per_rank_batch_size": settings.get("batch_size"),
            }
        runtime.write_metadata(partial_dir, metadata)
        partial_dir.rename(output_dir)
        if monitor:
            monitor.finish(loop_seconds=loop_elapsed_seconds, save_seconds=save_seconds)
    if context.enabled:
        distributed.barrier()
    return saved


def train_suite(
    config: Mapping[str, Any],
    tasks: Sequence[str] | None = None,
    max_samples: int | None = None,
) -> list[Any]:
    """Train the requested task experts sequentially on the current process/GPU."""

    selected = tuple(tasks) if tasks else runtime.task_names(config)
    available = set(runtime.task_names(config))
    unknown = [task for task in selected if task not in available]
    if unknown:
        raise ValueError(f"unknown experiment tasks: {', '.join(unknown)}")
    checkpoints = []
    for task in selected:
        checkpoints.append(train(config, task, max_samples))
        # The fused backend holds a cycle with its stepper. Release its tensors
        # before initializing the next expert, especially for two-GPU Llama.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return checkpoints
