"""One-forward/one-backward FT and SMAT updates.

SMAT differentiates q = base + a * mask/(1-p) * (theta-base) + noise
with respect to the original expert theta, then restores theta before clipping
and stepping its optimizer. Optimizer state never belongs to temporary q.
"""

import math
from collections.abc import Mapping
import torch
from .noise import ScaleMask
from .perturb import Perturbation, sample_noise
from .smat import SmatSchedule


class TrainingStepper:
    def __init__(self, parameters, optimizer):
        self.named_parameters = [(n, p) for n, p in parameters if p.requires_grad]
        self.parameters = [p for _, p in self.named_parameters]
        self.optimizer = optimizer
        self.step_index = 0

    def finish(self, max_grad_norm):
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_index += 1


class FTStepper(TrainingStepper):
    def step(self, closure, max_grad_norm=None):
        self.optimizer.zero_grad(set_to_none=True)
        loss = closure()
        loss.backward()
        self.finish(max_grad_norm)
        return loss.detach()


class SMATStepper(TrainingStepper):
    def __init__(self, parameters, optimizer, settings, seed, block_linear_names):
        super().__init__(parameters, optimizer)
        scale, mask, perturb = (
            settings.get(k, {}) for k in ("scale", "mask", "perturb")
        )
        for label, values, allowed in (
            ("scale", scale, {"alpha_min", "scope"}),
            ("mask", mask, {"probability", "scope", "embedding_probability"}),
            ("perturb", perturb, {"rms"}),
        ):
            if not isinstance(values, Mapping) or set(values) - allowed:
                raise ValueError(f"unsupported {label} settings")
        if scale.get("scope", "global") != "global":
            raise ValueError("the paper uses global Scale")
        if (
            mask.get("scope", "block-linear") != "block-linear"
            or mask.get("embedding_probability", 0) != 0
        ):
            raise ValueError("the paper masks Attention/MLP linear weights only")
        self.rms = float(perturb["rms"])
        if not math.isfinite(self.rms) or self.rms < 0:
            raise ValueError("perturb.rms must be finite and nonnegative")
        backend = settings.get("backend", "fast")
        if backend not in {"eager", "fused", "fast"}:
            raise ValueError("backend must be eager, fused or fast")
        offload = settings.get("offload_base", False)
        if not isinstance(offload, bool) or (offload and backend != "fast"):
            raise ValueError("offload_base must be boolean and requires backend=fast")
        self.self_path = ScaleMask(
            self.named_parameters,
            alpha_min=float(scale.get("alpha_min", 0.2)),
            probability=float(mask.get("probability", 0.5)),
            seed=seed,
            block_linear_names=block_linear_names,
            compact_base=backend != "eager",
        )
        self.perturbation = Perturbation(self.named_parameters, seed=seed)
        self.smat = SmatSchedule(
            settings.get("smat"), available=("scale", "mask", "perturb"), seed=seed
        )
        self.distribution = "uniform"
        self.direct_uniform = True
        self.noise_scales = dict.fromkeys((n for n, _ in self.named_parameters), 1.0)
        self.task_scale = 1.0
        self.task_noise_rms = self.rms
        self.fast_smat = backend == "fast"
        self.fused_smat = None
        if backend == "fast":
            from .fast_smat import FastSmat

            self.fused_smat = FastSmat(self, offload_base=offload)
        elif backend == "fused":
            from .fused_smat import FusedSmat

            self.fused_smat = FusedSmat(self)

    def offset(self, name, parameter):
        value = self.self_path.value(name, parameter, self.task_scale)
        if self.task_noise_rms:
            value.add_(
                sample_noise(
                    "uniform",
                    parameter,
                    rms=self.task_noise_rms,
                    generator=self.perturbation.generator(parameter.device, "noise"),
                    direct_uniform=True,
                )
            )
        return value

    def step(self, closure, max_grad_norm=None):
        self.optimizer.zero_grad(set_to_none=True)
        active = self.smat.select(self.step_index)
        if not active:
            loss = closure()
            if self.fast_smat:
                self.fused_smat.prefetch_base()
            loss.backward()
            self.finish(max_grad_norm)
            return loss.detach()
        self.task_scale = self.self_path.sample() if "scale" in active else 1.0
        self.self_path.begin_mask_step("mask" in active)
        self.task_noise_rms = self.rms if "perturb" in active else 0.0
        fused = self.fused_smat
        context = (
            fused.applied()
            if fused
            else self.perturbation.applied(self.offset, replace=True)
        )
        clipped = False
        with context:
            loss = closure()
            # Apply global Scale in backward, then the coordinate Mask; this
            # preserves the original low-precision order of arithmetic.
            anchor = self.task_scale if self.task_scale else 1.0
            (loss * anchor).backward()
            if fused:
                if self.fast_smat:
                    clipped = fused.gradients(
                        anchor, restore=True, max_grad_norm=max_grad_norm
                    )
                else:
                    fused.gradients(anchor, restore=True)
            else:
                for name, parameter in self.named_parameters:
                    if parameter.grad is None:
                        continue
                    factor = self.task_scale / anchor
                    if self.self_path.has_mask(name):
                        parameter.grad.mul_(self.self_path.masks[name])
                        factor /= 1 - self.self_path.mask_probabilities[name]
                    if factor != 1.0:
                        parameter.grad.mul_(factor)
        self.finish(None if clipped else max_grad_norm)
        return loss.detach()


def validate_settings(settings):
    """Reject misspelled or removed options before expensive model loading."""
    common = {
        "method",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "fused",
        "muon",
        "batch_size",
        "max_steps",
        "epochs",
        "epochs_by_task",
        "schedule",
        "warmup_steps",
        "max_grad_norm",
        "log_every",
        "monitor",
        "num_workers",
        "max_length",
        "padding",
        "append_eos",
        "gradient_checkpointing",
        "device_map",
    }
    operators = {"scale", "mask", "perturb", "backend", "offload_base", "smat"}
    method = settings.get("method")
    if method not in {"ft", "smat"}:
        raise ValueError("train.method must be ft or smat")
    unknown = set(settings) - common - (operators if method == "smat" else set())
    if unknown:
        raise ValueError(f"unsupported training options: {sorted(unknown)}")


def build_stepper(
    settings,
    named_parameters,
    optimizer,
    seed,
    *,
    block_linear_names=None,
    embedding_names=None,
):
    """Construct only the two released methods; reject unsupported recipes."""
    validate_settings(settings)
    method = settings.get("method")
    if method == "ft":
        return FTStepper(named_parameters, optimizer)
    if method == "smat":
        return SMATStepper(
            named_parameters, optimizer, settings, seed, block_linear_names or set()
        )
    raise ValueError("train.method must be ft or smat")
