"""Two SMAT kernels with either original samples or optional stateless RNG."""

from contextlib import contextmanager
import math

import torch
import triton
import triton.language as tl

from .perturb import sample_noise


@triton.jit(do_not_specialize=["Scale", "NoiseScale", "NoiseSeed", "MaskSeed", "Drop"])
def _apply(
    Weight,
    Base,
    Backup,
    Noise,
    Mask,
    N,
    Scale,
    NoiseScale,
    NoiseSeed,
    MaskSeed,
    Drop,
    MASKED: tl.constexpr,
    ADD_NOISE: tl.constexpr,
    RAW_NOISE: tl.constexpr,
    SEEDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    original = tl.load(Weight + i, valid, other=0)
    weight = original.to(tl.float32)
    base = tl.load(Base + i, valid, other=0).to(tl.float32)
    delta = weight - base
    if MASKED:
        mask = (
            (tl.rand(MaskSeed, i) >= Drop)
            if SEEDED
            else tl.load(Mask + i, valid, other=0)
        ).to(tl.float32)
        value = ((delta * mask) * Scale) + base
    else:
        # Match ATen's numerically stable lerp, including its fused multiply-add.
        value = tl.where(
            tl.abs(Scale) < 0.5,
            tl.fma(Scale, delta, base),
            tl.fma(-(1.0 - Scale), delta, weight),
        )
    if ADD_NOISE:
        noise = tl.rand(NoiseSeed, i) if SEEDED else tl.load(Noise + i, valid, other=0)
        if RAW_NOISE or SEEDED:
            noise = (noise - 0.5) * NoiseScale
        value = value + noise
    tl.store(Backup + i, original, valid)
    tl.store(Weight + i, value, valid)


@triton.jit(do_not_specialize=["Scale", "MaskSeed", "Drop"])
def _gradient_restore(
    Weight,
    Backup,
    Grad,
    Mask,
    N,
    Scale,
    MaskSeed,
    Drop,
    MASKED: tl.constexpr,
    HAS_GRAD: tl.constexpr,
    RESTORE: tl.constexpr,
    SEEDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    if HAS_GRAD:
        grad = tl.load(Grad + i, valid, other=0)
        if MASKED:
            mask = (
                (tl.rand(MaskSeed, i) >= Drop)
                if SEEDED
                else tl.load(Mask + i, valid, other=0)
            ).to(tl.float32)
            grad = (grad.to(tl.float32) * mask).to(grad.dtype)
        tl.store(Grad + i, grad.to(tl.float32) * Scale, valid)
    if RESTORE:
        tl.store(Weight + i, tl.load(Backup + i, valid, other=0), valid)


class FusedSmat:
    def __init__(self, stepper, *, seeded: bool = False):
        self.stepper = stepper
        self.steps = 0
        self.seeded = seeded
        self.generator = (
            torch.Generator().manual_seed(stepper.perturbation.seed + 1024)
            if seeded
            else None
        )
        self.seeds = [(0, 0)] * len(stepper.named_parameters)
        path = stepper.self_path
        if path is not None and not path.mask_rescale:
            raise ValueError(
                "mask_rescale=False is supported only by the eager reference backend"
            )
        if (
            stepper.distribution != "uniform"
            or path is None
            or path.noise_coupling != "constant"
            or path.mask_kind != "bernoulli"
            or path.mask_granularity != "coordinate"
        ):
            raise ValueError(
                "fused_smat needs Uniform, constant-coupling self and coordinate Bernoulli mask"
            )
        if any(
            not p.is_cuda
            or not p.is_contiguous()
            or p.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            for _, p in stepper.named_parameters
        ):
            raise ValueError(
                "fused_smat needs contiguous CUDA FP32/BF16/FP16 parameters"
            )

    @contextmanager
    def applied(self):
        stepper = self.stepper
        path, perturb = stepper.self_path, stepper.perturbation
        perturb.active.clear()
        self.steps += 1
        if self.seeded:
            # Separate 63-bit keys for noise/mask, independent of enabled scopes.
            self.seeds = torch.randint(
                0, 2**63 - 1, (len(self.seeds), 2), generator=self.generator
            ).tolist()
        try:
            for index, ((name, parameter), backup) in enumerate(
                zip(stepper.named_parameters, perturb.backups)
            ):
                rms = stepper.task_noise_rms * stepper.noise_scales[name]
                noise = parameter
                if rms and not self.seeded:
                    generator = perturb.generator(parameter.device, "noise")
                    noise = (
                        sample_noise(
                            "uniform",
                            parameter,
                            rms=rms,
                            generator=generator,
                            direct_uniform=True,
                        )
                        if stepper.direct_uniform
                        else torch.rand(
                            parameter.shape,
                            device=parameter.device,
                            dtype=torch.float32,
                            generator=generator,
                        )
                    )
                masked = path.has_mask(name)
                mask = (
                    path.sample_mask(name, parameter)
                    if masked and not self.seeded
                    else parameter
                )
                drop = path.mask_probabilities[name] if masked else 0.0
                scale = path.scale(name, stepper.task_scale) / (1 - drop)
                _apply[(triton.cdiv(parameter.numel(), 1024),)](
                    parameter,
                    path.base[name],
                    backup,
                    noise,
                    mask,
                    parameter.numel(),
                    scale,
                    2 * math.sqrt(3.0) * rms,
                    *self.seeds[index],
                    drop,
                    masked,
                    bool(rms),
                    not stepper.direct_uniform,
                    self.seeded,
                    1024,
                    enable_fp_fusion=False,
                )
                perturb.active.append(index)
                torch.autograd.graph.increment_version(parameter)
            yield
        finally:
            # Also restores completed writes if forward/backward raises.
            perturb.restore()

    @torch.no_grad()
    def gradients(self, anchor_scale: float, *, restore: bool):
        stepper = self.stepper
        path, perturb = stepper.self_path, stepper.perturbation
        for index in perturb.active:
            name, parameter = stepper.named_parameters[index]
            original_grad = parameter.grad
            grad = original_grad.contiguous() if original_grad is not None else None
            masked = path.has_mask(name)
            mask = path.masks[name] if masked and not self.seeded else parameter
            drop = path.mask_probabilities[name] if masked else 0.0
            scale = path.scale(name, stepper.task_scale) / anchor_scale / (1 - drop)
            _gradient_restore[(triton.cdiv(parameter.numel(), 1024),)](
                parameter,
                perturb.backups[index],
                grad if grad is not None else parameter,
                mask,
                parameter.numel(),
                scale,
                self.seeds[index][1],
                drop,
                masked,
                grad is not None,
                restore,
                self.seeded,
                1024,
                enable_fp_fusion=False,
            )
            if grad is not original_grad:
                original_grad.copy_(grad)
            if restore:
                torch.autograd.graph.increment_version(parameter)
        if restore:
            perturb.active.clear()
