"""Low-overhead Uniform + self + mask training path for CUDA parameters."""

from collections import defaultdict
from contextlib import contextmanager
import math

import torch
import triton
import triton.language as tl

from .fused_smat import FusedSmat


# The reduction benefits from larger tiles than the two full-tensor writes.
_APPLY_BLOCK = 2048
_NORM_BLOCK = 4096
_GRAD_BLOCK = 2048


@triton.jit
def _rand4(seed, BLOCK: tl.constexpr):
    """Return one Philox uniform for each contiguous coordinate in this program."""
    tl.static_assert(BLOCK % 4 == 0)
    start = tl.program_id(0) * BLOCK
    r0, r1, r2, r3 = tl.rand4x(seed, start // 4 + tl.arange(0, BLOCK // 4))
    # Coordinate 4*j + lane takes lane from tl.rand4x(seed, j).
    return tl.reshape(tl.join(tl.join(r0, r2), tl.join(r1, r3)), [BLOCK])


@triton.jit
def _apply_temporary(
    Pointers,
    Values,
    Seeds,
    N: tl.constexpr,
    DTYPE: tl.constexpr,
    MASKED: tl.constexpr,
    NOISE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1) * 4
    index = tl.load(Pointers + row)
    weight = tl.load(Pointers + row + 1).to(tl.pointer_type(DTYPE))
    base = tl.load(Pointers + row + 2).to(tl.pointer_type(DTYPE))
    temporary = tl.load(Pointers + row + 3).to(tl.pointer_type(DTYPE))
    scale = tl.load(Values + index * 3)
    noise_scale = tl.load(Values + index * 3 + 1)
    drop = tl.load(Values + index * 3 + 2)
    noise_seed = tl.load(Seeds + index * 2)
    mask_seed = tl.load(Seeds + index * 2 + 1)

    coordinate = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = coordinate < N
    original = tl.load(weight + coordinate, valid, other=0)
    base_value = tl.load(base + coordinate, valid, other=0).to(tl.float32)
    delta = original.to(tl.float32) - base_value
    if MASKED:
        mask = (_rand4(mask_seed, BLOCK) >= drop).to(tl.float32)
        value = (delta * mask) * scale + base_value
    else:
        value = tl.where(
            tl.abs(scale) < 0.5,
            tl.fma(scale, delta, base_value),
            tl.fma(-(1.0 - scale), delta, original.to(tl.float32)),
        )
    if NOISE:
        value += (_rand4(noise_seed, BLOCK) - 0.5) * noise_scale
    tl.store(temporary + coordinate, value, valid)


@triton.jit
def _scale_gradients(
    Pointers,
    GradPointers,
    Scales,
    Values,
    Seeds,
    N: tl.constexpr,
    DTYPE: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1) * 4
    index = tl.load(Pointers + row)
    address = tl.load(GradPointers + index)
    gradient = address.to(tl.pointer_type(DTYPE))
    scale = tl.load(Scales + index)
    drop = tl.load(Values + index * 3 + 2)
    mask_seed = tl.load(Seeds + index * 2 + 1)

    coordinate = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = (coordinate < N) & (address != 0)
    value = tl.load(gradient + coordinate, valid, other=0)
    if MASKED:
        mask = (_rand4(mask_seed, BLOCK) >= drop).to(tl.float32)
        value = (value.to(tl.float32) * mask).to(value.dtype)
    tl.store(gradient + coordinate, value.to(tl.float32) * scale, valid)


@triton.jit
def _deferred_gradients(
    Pointers,
    GradPointers,
    Scales,
    Values,
    Seeds,
    Partials,
    Clip,
    N: tl.constexpr,
    DTYPE: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCKS: tl.constexpr,
    WRITE: tl.constexpr,
    PARTIALS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Collect post-chain norm partials or regenerate and write clipped gradients."""
    row = tl.program_id(1) * 4
    index = tl.load(Pointers + row)
    address = tl.load(GradPointers + index)
    gradient = address.to(tl.pointer_type(DTYPE))
    scale = tl.load(Scales + index)
    drop = tl.load(Values + index * 3 + 2)
    mask_seed = tl.load(Seeds + index * 2 + 1)

    block = tl.program_id(0)
    coordinate = block * BLOCK + tl.arange(0, BLOCK)
    valid = (coordinate < N) & (address != 0)
    value = tl.load(gradient + coordinate, valid, other=0)
    if MASKED:
        mask = (_rand4(mask_seed, BLOCK) >= drop).to(tl.float32)
        value = (value.to(tl.float32) * mask).to(value.dtype)
    # Norms must see the low-precision value that the normal chain rule would
    # have written before PyTorch clips it.
    value = (value.to(tl.float32) * scale).to(DTYPE)
    if PARTIALS:
        square = value.to(tl.float32) * value.to(tl.float32)
        tl.store(
            Partials + tl.program_id(1) * BLOCKS + block,
            tl.sum(tl.where(valid, square, 0.0), axis=0),
        )
    if WRITE:
        coefficient = tl.load(Clip).to(tl.float32)
        tl.store(
            gradient + coordinate, (value.to(tl.float32) * coefficient).to(DTYPE), valid
        )


class FastSmat(FusedSmat):
    """Grouped stateless SMAT with temporary-weight storage swaps.

    Forward and its complete backward must run inside ``applied``; temporary
    weights may be overwritten as soon as that context exits.
    """

    def __init__(self, stepper, *, offload_base=False):
        super().__init__(stepper, seeded=True)
        self.originals = [
            parameter.detach() for _, parameter in stepper.named_parameters
        ]
        self.devices = {parameter.device for _, parameter in stepper.named_parameters}
        self.base_streams = {}
        self.base_batches = {}
        self.prefetch_count = 0
        self.prefetch_index = 0
        if offload_base:
            schedule = stepper.smat
            if (
                schedule.mode != "joint"
                or schedule.interval < 2
                or not set(schedule.components) & {"scale", "mask"}
            ):
                raise ValueError(
                    "offload_base requires joint SMAT with interval >= 2 and scale or mask"
                )
            # Between SMAT steps, the temporary weights double as a base cache.
            # Keep the immutable source pinned so refill overlaps clean steps.
            for name, base in stepper.self_path.base.items():
                host = torch.empty_like(base, device="cpu", pin_memory=True)
                host.copy_(base)
                stepper.self_path.base[name] = host
            self.base_streams = {
                device: torch.cuda.Stream(device=device) for device in self.devices
            }
            self.prefetch_count = schedule.interval - 1
            self.base_batches = self._base_batches(self.prefetch_count)
            self.prefetch_index = (
                self.prefetch_count
            )  # Initial clones already contain the base.
        buckets = defaultdict(list)
        path = stepper.self_path
        for index, (name, parameter) in enumerate(stepper.named_parameters):
            key = (
                parameter.device,
                parameter.dtype,
                parameter.numel(),
                bool(path.mask_probabilities[name]),
                bool(stepper.noise_scales[name]),
            )
            buckets[key].append(index)
        self.groups = []
        for (device, dtype, size, masked, noise), indices in buckets.items():
            rows = []
            for index in indices:
                name, parameter = stepper.named_parameters[index]
                temporary = stepper.perturbation.backups[index]
                base = temporary if offload_base else path.base[name]
                rows.append(
                    [index, parameter.data_ptr(), base.data_ptr(), temporary.data_ptr()]
                )
            pointers = torch.tensor(rows, dtype=torch.int64, device=device)
            self.groups.append((indices, pointers, dtype, size, masked, noise))
        self.partial_buffers = [
            torch.empty(
                (len(indices), math.ceil(size / _NORM_BLOCK)),
                device=pointers.device,
                dtype=torch.float32,
            )
            for indices, pointers, _, size, _, _ in self.groups
        ]

    @staticmethod
    def _device_tables(*host_tensors, devices):
        return {
            device: tuple(
                tensor.to(device, non_blocking=True) for tensor in host_tensors
            )
            for device in devices
        }

    def _base_batches(self, count):
        """Partition existing views by bytes, with one refill per clean step."""
        stepper = self.stepper
        batches = {}
        for device in self.devices:
            pairs = [
                (stepper.self_path.base[name].view(-1), temporary.view(-1))
                for (name, parameter), temporary in zip(
                    stepper.named_parameters, stepper.perturbation.backups, strict=True
                )
                if parameter.device == device
            ]
            total = sum(host.numel() * host.element_size() for host, _ in pairs)
            batches[device] = []
            for index in range(count):
                low, high = total * index // count, total * (index + 1) // count
                offset, batch = 0, []
                for host, temporary in pairs:
                    width = host.element_size()
                    start = max(0, (low - offset + width - 1) // width)
                    end = min(host.numel(), (high - offset + width - 1) // width)
                    if end > start:
                        batch.append((host[start:end], temporary[start:end]))
                    offset += host.numel() * width
                batches[device].append(batch)
        return batches

    def _path_tables(self):
        stepper, path = self.stepper, self.stepper.self_path
        seeds = torch.randint(
            0, 2**63 - 1, (len(self.seeds), 2), generator=self.generator
        )
        values = []
        for name, _ in stepper.named_parameters:
            drop = path.mask_probabilities[name] if path.has_mask(name) else 0.0
            values.append(
                [
                    path.scale(name, stepper.task_scale) / (1 - drop),
                    2
                    * math.sqrt(3.0)
                    * (stepper.task_noise_rms * stepper.noise_scales[name]),
                    drop,
                ]
            )
        return self._device_tables(
            torch.tensor(values, dtype=torch.float32, pin_memory=True),
            seeds.pin_memory(),
            devices=self.devices,
        )

    @contextmanager
    def applied(self):
        stepper, path, perturb = (
            self.stepper,
            self.stepper.self_path,
            self.stepper.perturbation,
        )
        perturb.active.clear()
        self.steps += 1
        self.tables = self._path_tables()
        try:
            # Normally clean steps have filled every batch. Also handle a
            # direct context call or retry after an interrupted training step.
            while self.prefetch_index < self.prefetch_count:
                self.prefetch_base()
            for device, stream in self.base_streams.items():
                torch.cuda.current_stream(device).wait_stream(stream)
            for indices, pointers, dtype, size, masked, noise in self.groups:
                with torch.cuda.device(pointers.device):
                    _apply_temporary[(triton.cdiv(size, _APPLY_BLOCK), len(indices))](
                        pointers,
                        *self.tables[pointers.device],
                        size,
                        getattr(tl, str(dtype).split(".")[-1]),
                        masked and path.mask_active,
                        noise and bool(stepper.task_noise_rms),
                        _APPLY_BLOCK,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                perturb.active.extend(indices)
            for index in perturb.active:
                stepper.named_parameters[index][1].data = perturb.backups[index]
            torch.autograd.graph.increment_version(stepper.parameters)
            yield
        finally:
            self._restore()
            self.prefetch_index = 0

    def prefetch_base(self):
        """Refill a slice after clean forward, overlapping its backward pass.

        Queuing the full base at once can delay the next input's H2D transfer,
        even on a separate stream. Clean steps spread those copies out.
        """
        if self.prefetch_index >= self.prefetch_count:
            return
        for device, stream in self.base_streams.items():
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for host, temporary in self.base_batches[device][self.prefetch_index]:
                    temporary.copy_(host, non_blocking=True)
                    temporary.record_stream(stream)
        self.prefetch_index += 1

    def _gradient_tables(self, anchor_scale, gradients):
        stepper, path = self.stepper, self.stepper.self_path
        pointers = torch.tensor(
            [
                gradient.data_ptr() if gradient is not None else 0
                for gradient in gradients
            ],
            dtype=torch.int64,
            pin_memory=True,
        )
        scales = torch.tensor(
            [
                path.scale(name, stepper.task_scale)
                / anchor_scale
                / (1 - path.mask_probabilities[name] if path.has_mask(name) else 1)
                for name, _ in stepper.named_parameters
            ],
            dtype=torch.float32,
            pin_memory=True,
        )
        return self._device_tables(pointers, scales, devices=self.devices)

    @torch.no_grad()
    def _write_gradients(self, anchor_scale, *, restore):
        path = self.stepper.self_path
        originals = [parameter.grad for _, parameter in self.stepper.named_parameters]
        gradients = [
            gradient.contiguous() if gradient is not None else None
            for gradient in originals
        ]
        tables = self._gradient_tables(anchor_scale, gradients)
        for _, parameter_pointers, dtype, size, masked, _ in self.groups:
            with torch.cuda.device(parameter_pointers.device):
                _scale_gradients[
                    (triton.cdiv(size, 1024), parameter_pointers.shape[0])
                ](
                    parameter_pointers,
                    *tables[parameter_pointers.device],
                    *self.tables[parameter_pointers.device],
                    size,
                    getattr(tl, str(dtype).split(".")[-1]),
                    masked and path.mask_active,
                    1024,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        for original, gradient in zip(originals, gradients):
            if gradient is not original:
                original.copy_(gradient)
        if restore:
            self._restore()

    @staticmethod
    def _total_norm(per_parameter_norms):
        first_device = per_parameter_norms[0].device
        values = list(per_parameter_norms)
        remote = defaultdict(list)
        for index, norm in enumerate(values):
            if norm.device != first_device:
                remote[norm.device, norm.dtype].append((index, norm))
        # Transfer a packed vector per remote device/dtype, retaining the
        # original scalar order and precision for the final norm.
        for entries in remote.values():
            packed = torch.stack([norm for _, norm in entries]).to(first_device)
            for (index, _), norm in zip(entries, packed.unbind(), strict=True):
                values[index] = norm
        return torch.linalg.vector_norm(torch.stack(values), 2.0)

    @torch.no_grad()
    def _deferred_gradients(self, anchor_scale, max_grad_norm, gradients):
        path = self.stepper.self_path
        tables = self._gradient_tables(anchor_scale, gradients)
        per_parameter = [None] * len(gradients)
        for group, partials in zip(self.groups, self.partial_buffers, strict=True):
            indices, parameter_pointers, dtype, size, masked, _ = group
            blocks = partials.shape[1]
            with torch.cuda.device(parameter_pointers.device):
                _deferred_gradients[(blocks, len(indices))](
                    parameter_pointers,
                    *tables[parameter_pointers.device],
                    *self.tables[parameter_pointers.device],
                    partials,
                    tables[parameter_pointers.device][1],
                    size,
                    getattr(tl, str(dtype).split(".")[-1]),
                    masked and path.mask_active,
                    blocks,
                    False,
                    True,
                    _NORM_BLOCK,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            norms = torch.sqrt(partials.sum(dim=1)).to(dtype)
            for index, norm in zip(indices, norms, strict=True):
                if gradients[index] is not None:
                    per_parameter[index] = norm

        norms = [norm for norm in per_parameter if norm is not None]
        if not norms:
            self._restore()
            return False
        total_norm = self._total_norm(norms)
        coefficient = torch.clamp(max_grad_norm / (total_norm + 1e-6), max=1.0)
        coefficients = {device: coefficient.to(device) for device in self.devices}
        for group, partials in zip(self.groups, self.partial_buffers, strict=True):
            _, parameter_pointers, dtype, size, masked, _ = group
            with torch.cuda.device(parameter_pointers.device):
                _deferred_gradients[
                    (triton.cdiv(size, _GRAD_BLOCK), parameter_pointers.shape[0])
                ](
                    parameter_pointers,
                    *tables[parameter_pointers.device],
                    *self.tables[parameter_pointers.device],
                    partials,
                    coefficients[parameter_pointers.device],
                    size,
                    getattr(tl, str(dtype).split(".")[-1]),
                    masked and path.mask_active,
                    partials.shape[1],
                    True,
                    False,
                    _GRAD_BLOCK,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        self._restore()
        return True

    @torch.no_grad()
    def gradients(self, anchor_scale, *, restore, max_grad_norm=None):
        current = [parameter.grad for _, parameter in self.stepper.named_parameters]
        if (
            max_grad_norm is not None
            and restore
            and all(
                gradient is None or gradient.is_contiguous() for gradient in current
            )
        ):
            return self._deferred_gradients(anchor_scale, max_grad_norm, current)
        self._write_gradients(anchor_scale, restore=restore)
        return False

    @torch.no_grad()
    def _restore(self):
        active = self.stepper.perturbation.active
        if active:
            for index in active:
                self.stepper.named_parameters[index][1].data = self.originals[index]
            torch.autograd.graph.increment_version(self.stepper.parameters)
            active.clear()
