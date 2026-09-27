"""Temporary parameter perturbations with private device RNG streams."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
import math

import torch


NamedParameters = Iterable[tuple[str, torch.nn.Parameter]]
Offset = Callable[[str, torch.nn.Parameter], torch.Tensor]
_STREAM_OFFSETS = {"noise": 0, "coefficient": 1}


def sample_noise(
    distribution: str,
    reference: torch.Tensor,
    *,
    rms: float,
    generator: torch.Generator,
    direct_uniform: bool = False,
) -> torch.Tensor:
    """Sample one float32 parameter-space noise tensor with target RMS."""

    if distribution == "uniform":
        if direct_uniform:
            bound = math.sqrt(3.0) * rms
            return torch.empty_like(reference, dtype=torch.float32).uniform_(
                -bound,
                bound,
                generator=generator,
            )
        return (
            torch.rand(
                reference.shape,
                device=reference.device,
                dtype=torch.float32,
                generator=generator,
            )
            .sub_(0.5)
            .mul_(2 * math.sqrt(3.0) * rms)
        )
    raise ValueError(f"unsupported noise distribution: {distribution}")


class Perturbation:
    """Apply offsets for one forward/backward pass, then restore parameters."""

    def __init__(self, parameters: NamedParameters, *, seed: int):
        self.parameters = [
            (name, parameter)
            for name, parameter in parameters
            if parameter.requires_grad
        ]
        self.backups = [parameter.detach().clone() for _, parameter in self.parameters]
        self.active: list[int] = []
        self.seed = int(seed)
        self.multi_device = len({p.device for _, p in self.parameters}) > 1
        self.generators: dict[tuple[str, str], torch.Generator] = {}

    def generator(self, device: torch.device, stream: str) -> torch.Generator:
        key = (stream, str(device))
        generator = self.generators.get(key)
        if generator is None:
            try:
                offset = _STREAM_OFFSETS[stream]
            except KeyError as error:
                raise ValueError(f"unsupported RNG stream: {stream}") from error
            generator = torch.Generator(device=device)
            # Noise streams are independent across model-parallel devices.
            if self.multi_device and stream == "noise":
                offset += 104729 * (device.index or 0)
            generator.manual_seed(self.seed + offset)
            self.generators[key] = generator
        return generator

    @torch.no_grad()
    def apply(self, offset: Offset, *, replace: bool = False) -> None:
        self.active.clear()
        try:
            for index, ((name, parameter), backup) in enumerate(
                zip(self.parameters, self.backups)
            ):
                value = offset(name, parameter)
                backup.copy_(parameter)
                self.active.append(index)
                if replace:
                    parameter.copy_(value)
                else:
                    parameter.add_(value.to(parameter.dtype))
        except Exception:
            self.restore()
            raise

    @torch.no_grad()
    def restore(self) -> None:
        for index in self.active:
            _, parameter = self.parameters[index]
            backup = self.backups[index]
            parameter.copy_(backup)
        self.active.clear()

    @contextmanager
    def applied(self, offset: Offset, *, replace: bool = False) -> Iterator[None]:
        self.apply(offset, replace=replace)
        try:
            yield
        finally:
            self.restore()
