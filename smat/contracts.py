"""Small checkpoint I/O boundary used by the merge kernels.

The merge implementations deliberately know nothing about Hugging Face,
``safetensors`` or a compact vision state dict. An experiment backend supplies this adapter for
its checkpoint format, while the kernels below only ever hold one named tensor
from each input checkpoint at a time.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

import torch


CheckpointRef: TypeAlias = str | Path


@runtime_checkable
class TensorStore(Protocol):
    """A named-tensor view of one checkpoint.

    ``keys`` and ``get_tensor`` should stream from the underlying format where
    possible.  ``put_tensor`` is only required for output stores.
    """

    def keys(self) -> Iterable[str]: ...

    def get_tensor(self, key: str) -> torch.Tensor: ...

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None: ...


@runtime_checkable
class CheckpointAdapter(Protocol):
    """Opens checkpoint tensor stores and creates an output from a template.

    The template lets a backend preserve non-tensor assets and checkpoint
    metadata without making the merge kernels framework-specific.
    """

    def read(
        self, checkpoint: CheckpointRef
    ) -> AbstractContextManager[TensorStore]: ...

    def write(
        self, output: CheckpointRef, *, template: CheckpointRef
    ) -> AbstractContextManager[TensorStore]: ...
