from __future__ import annotations

import unittest
from contextlib import contextmanager

import torch

from smat.merge import task_arithmetic, weight_average


class MemoryStore:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        self.tensors[key] = tensor


class MemoryAdapter:
    def __init__(self, checkpoints: dict[str, dict[str, torch.Tensor]]):
        self.checkpoints = checkpoints

    @contextmanager
    def read(self, checkpoint: str):
        yield MemoryStore(self.checkpoints[checkpoint])

    @contextmanager
    def write(self, output: str, *, template: str):
        tensors = {
            key: tensor.clone() for key, tensor in self.checkpoints[template].items()
        }
        yield MemoryStore(tensors)
        self.checkpoints[output] = tensors


class MergeTest(unittest.TestCase):
    def test_weight_average_is_native_dtype_sum_then_divide(self):
        one = torch.tensor([1.0, 0.333984375], dtype=torch.bfloat16)
        two = torch.tensor([2.0, 0.66796875], dtype=torch.bfloat16)
        adapter = MemoryAdapter({"one": {"weight": one}, "two": {"weight": two}})

        weight_average(adapter, ("one", "two"), "merged")

        expected = one.clone()
        expected.add_(two)
        expected.div_(2)
        self.assertEqual(adapter.checkpoints["merged"]["weight"].dtype, torch.bfloat16)
        torch.testing.assert_close(adapter.checkpoints["merged"]["weight"], expected)

    def test_task_arithmetic_uses_native_dtype_and_fixed_scale(self):
        base = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        one = torch.tensor([2.0, 5.0], dtype=torch.bfloat16)
        two = torch.tensor([4.0, 3.0], dtype=torch.bfloat16)
        adapter = MemoryAdapter(
            {"base": {"weight": base}, "one": {"weight": one}, "two": {"weight": two}}
        )

        task_arithmetic(adapter, "base", ("one", "two"), "merged", scale=0.3)

        expected = base.clone()
        total = torch.zeros_like(base)
        total.add_(one - base)
        total.add_(two - base)
        expected.add_(0.3 * total)
        torch.testing.assert_close(adapter.checkpoints["merged"]["weight"], expected)
