"""Optimizer construction shared by the common expert-training loop."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import wraps
from itertools import zip_longest
from typing import Any

import torch

NamedParameters = Iterable[tuple[str, torch.nn.Parameter]]


def _interleave_device_steps(optimizer: torch.optim.Optimizer) -> None:
    """Overlap native matrix updates across devices; preserve checkpoint ordering."""
    if len({p.device for g in optimizer.param_groups for p in g["params"]}) < 2:
        return
    native_step = optimizer.step

    @wraps(native_step)
    def step(*args, **kwargs):
        layouts = []
        for group in optimizer.param_groups:
            devices = defaultdict(list)
            for parameter in group["params"]:
                devices[parameter.device].append(parameter)
            if len(devices) > 1:
                order = [
                    p
                    for row in zip_longest(*devices.values())
                    for p in row
                    if p is not None
                ]
                layouts.append((group, group["params"]))
                group["params"] = order
        try:
            return native_step(*args, **kwargs)
        finally:
            for group, original in layouts:
                group["params"] = original

    optimizer.step = step


class CombinedOptimizer(torch.optim.Optimizer):
    """Present native optimizers as one optimizer to the training loop.

    The children retain their own state and parameter groups.  Reusing those
    group dictionaries means schedulers attached to this wrapper update the
    learning rates read by each native optimizer.
    """

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one optimizer")
        parameters = [
            parameter
            for optimizer in optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if not parameters:
            raise ValueError("CombinedOptimizer needs at least one parameter")
        super().__init__(parameters, {})
        self.optimizers = tuple(optimizers)
        self._refresh_param_groups()

    def _refresh_param_groups(self) -> None:
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Save the native optimizer states without flattening their formats."""

        return {
            "format": "smat.combined-optimizer.v1",
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        states = state_dict.get("optimizers")
        if not isinstance(states, list) or len(states) != len(self.optimizers):
            raise ValueError(
                "combined optimizer state does not match its child optimizers"
            )
        for optimizer, state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(state)
        self._refresh_param_groups()


_TRANSFORMER_LAYER_WEIGHT = re.compile(
    r"(?:^|\.)(?:encoder|model)\.layers\.\d+\..+\.weight$"
)


def _adam_arguments(settings: Mapping[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "lr": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
    }
    fused = settings.get("fused")
    if fused is not None:
        arguments["fused"] = bool(fused)
    return arguments


def _muon_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split only hidden transformer matrix weights into Muon and AdamW."""

    muon: list[torch.nn.Parameter] = []
    adamw: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if id(parameter) in seen:
            raise ValueError(f"duplicate optimizer parameter: {name}")
        seen.add(id(parameter))
        if parameter.ndim == 2 and _TRANSFORMER_LAYER_WEIGHT.search(name):
            muon.append(parameter)
        else:
            adamw.append(parameter)
    return muon, adamw


def _muon_arguments(settings: Mapping[str, Any]) -> dict[str, Any]:
    specification = settings.get("muon")
    if not isinstance(specification, Mapping):
        raise ValueError("train.muon must be an object for optimizer muon-adamw")
    try:
        learning_rate = float(specification["learning_rate"])
    except KeyError as error:
        raise ValueError("train.muon.learning_rate is required") from error
    return {
        "lr": learning_rate,
        "weight_decay": float(settings["weight_decay"]),
        "momentum": float(specification.get("momentum", 0.95)),
        "nesterov": True,
        "ns_steps": int(specification.get("ns_steps", 5)),
        "adjust_lr_fn": str(specification.get("adjust_lr_fn", "original")),
    }


def build_optimizer(
    settings: Mapping[str, Any], named_parameters: NamedParameters
) -> torch.optim.Optimizer:
    """Build the configured optimizer from named trainable model parameters."""

    optimizer_name = settings.get("optimizer")
    if not isinstance(optimizer_name, str):
        raise ValueError("train.optimizer must be a string")
    values = list(named_parameters)
    parameters = [parameter for _, parameter in values]
    arguments = _adam_arguments(settings)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, **arguments)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, **arguments)
    if optimizer_name != "muon-adamw":
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    muon_parameters, adamw_parameters = _muon_parameters(values)
    optimizers: list[torch.optim.Optimizer] = []
    if muon_parameters:
        muon_optimizer = torch.optim.Muon(muon_parameters, **_muon_arguments(settings))
        _interleave_device_steps(muon_optimizer)
        optimizers.append(muon_optimizer)
    if adamw_parameters:
        optimizers.append(torch.optim.AdamW(adamw_parameters, **arguments))
    return CombinedOptimizer(optimizers)
