"""FusionBench-aligned WA and Task-Arithmetic streaming merge kernels.

The experiment adapter owns the checkpoint scope. These kernels preserve the
source tensor dtype, copy non-floating tensors from the base/template, and do
not depend on a model framework.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack

import torch

from .contracts import CheckpointAdapter, CheckpointRef, TensorStore


def _is_float(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point()


def _copy(output: TensorStore, key: str, tensor: torch.Tensor) -> None:
    output.put_tensor(key, tensor.clone())


def weight_average(
    adapter: CheckpointAdapter,
    experts: Sequence[CheckpointRef],
    output: CheckpointRef,
) -> None:
    """FusionBench simple average: source-dtype sum, then one division."""

    with ExitStack() as stack:
        stores = [stack.enter_context(adapter.read(expert)) for expert in experts]
        destination = stack.enter_context(adapter.write(output, template=experts[0]))
        for key in stores[0].keys():
            tensors = [store.get_tensor(key) for store in stores]
            if not _is_float(tensors[0]):
                _copy(destination, key, tensors[0])
                continue
            result = tensors[0].clone()
            for tensor in tensors[1:]:
                result.add_(tensor)
            result.div_(len(tensors))
            destination.put_tensor(key, result)


def task_arithmetic(
    adapter: CheckpointAdapter,
    base: CheckpointRef,
    experts: Sequence[CheckpointRef],
    output: CheckpointRef,
    *,
    scale: float,
) -> None:
    """FusionBench task arithmetic with an explicit task-vector scale."""
    with ExitStack() as stack:
        base_store = stack.enter_context(adapter.read(base))
        stores = [stack.enter_context(adapter.read(expert)) for expert in experts]
        destination = stack.enter_context(adapter.write(output, template=base))
        for key in base_store.keys():
            base_tensor = base_store.get_tensor(key)
            if not _is_float(base_tensor):
                _copy(destination, key, base_tensor)
                continue
            total = torch.zeros_like(base_tensor)
            for store in stores:
                total.add_(store.get_tensor(key) - base_tensor)
            destination.put_tensor(key, base_tensor + float(scale) * total)
