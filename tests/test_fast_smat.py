import copy
from contextlib import contextmanager
import math
import unittest

import torch

from smat.train.noise import block_linear_parameter_names, embedding_parameter_names
from smat.train.updates import build_stepper

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _oracle_rand4(Output, N, Seed, BLOCK: tl.constexpr):
        """Independently select the rand4x lane for each logical coordinate."""

        coordinate = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = coordinate < N
        r0, r1, r2, r3 = tl.rand4x(Seed, coordinate // 4)
        lane = coordinate % 4
        value = tl.where(
            lane == 0, r0, tl.where(lane == 1, r1, tl.where(lane == 2, r2, r3))
        )
        tl.store(Output + coordinate, value, valid)


def _rand4(seed: int, parameter: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(parameter, dtype=torch.float32)
    with torch.cuda.device(parameter.device):
        _oracle_rand4[(triton.cdiv(parameter.numel(), 1024),)](
            output, parameter.numel(), seed, 1024
        )
    return output


class _Block(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = torch.nn.Linear(16, 16)
        self.mlp_up = torch.nn.Linear(16, hidden)
        self.mlp_down = torch.nn.Linear(hidden, 16)
        self.norm = torch.nn.LayerNorm(16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value + self.attn(value)
        return self.norm(
            residual + self.mlp_down(torch.nn.functional.gelu(self.mlp_up(residual)))
        )


class _MixedBucketModel(torch.nn.Module):
    """A tied embedding and varied matrix/vector shapes, independent of artifacts."""

    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.randn(7))
        self.embedding = torch.nn.Embedding(19, 16)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block(24), _Block(12)])
        self.head = torch.nn.Linear(16, 19, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.embedding(tokens)
        for block in self.model.layers:
            value = block(value)
        return self.head(value)


class _RaggedBlock(torch.nn.Module):
    """Two equal large matrices form one deferred-norm bucket."""

    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(73, 65, bias=False)
        self.second = torch.nn.Linear(73, 65, bias=False)


class _RaggedGradientModel(torch.nn.Module):
    """No forward is needed: this fixture directly supplies fixed gradients."""

    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.randn(7))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_RaggedBlock()])


class OracleBackend:
    """Small, explicit rand4x/Torch reference for FastSmat's public protocol."""

    def __init__(self, stepper):
        self.stepper = stepper
        self.steps = 0
        self.generator = torch.Generator().manual_seed(stepper.perturbation.seed + 1024)
        self.seeds = [(0, 0)] * len(stepper.named_parameters)

    @contextmanager
    def applied(self):
        stepper = self.stepper
        path, perturb = stepper.self_path, stepper.perturbation
        perturb.active.clear()
        self.steps += 1
        self.seeds = torch.randint(
            0,
            2**63 - 1,
            (len(self.seeds), 2),
            generator=self.generator,
        ).tolist()
        try:
            with torch.no_grad():
                for index, ((name, parameter), backup) in enumerate(
                    zip(stepper.named_parameters, perturb.backups)
                ):
                    masked = path.has_mask(name)
                    drop = path.mask_probabilities[name] if masked else 0.0
                    scale = path.scale(name, stepper.task_scale) / (1 - drop)
                    base = path.base[name].float()
                    delta = parameter.float() - base
                    if masked:
                        mask = (_rand4(self.seeds[index][1], parameter) >= drop).to(
                            torch.float32
                        )
                        # Deliberately keep the affine operations separate: no FMA shortcut.
                        value = delta * mask
                        value = value * scale
                        value = value + base
                    else:
                        value = torch.lerp(base, parameter.float(), scale)
                    rms = stepper.task_noise_rms * stepper.noise_scales[name]
                    if rms:
                        noise = _rand4(self.seeds[index][0], parameter)
                        noise = noise - 0.5
                        noise = noise * (2 * math.sqrt(3.0) * rms)
                        value = value + noise
                    backup.copy_(parameter)
                    parameter.copy_(value.to(parameter.dtype))
                    perturb.active.append(index)
                    torch.autograd.graph.increment_version(parameter)
            yield
        finally:
            self.restore()

    @torch.no_grad()
    def restore(self):
        perturb = self.stepper.perturbation
        for index in perturb.active:
            _, parameter = self.stepper.named_parameters[index]
            parameter.copy_(perturb.backups[index])
            torch.autograd.graph.increment_version(parameter)
        perturb.active.clear()

    @torch.no_grad()
    def gradients(self, anchor_scale, *, restore, max_grad_norm=None):
        del max_grad_norm
        stepper = self.stepper
        path, perturb = stepper.self_path, stepper.perturbation
        for index in perturb.active:
            name, parameter = stepper.named_parameters[index]
            original = parameter.grad
            if original is not None:
                gradient = original.contiguous()
                if path.has_mask(name):
                    drop = path.mask_probabilities[name]
                    mask = (_rand4(self.seeds[index][1], parameter) >= drop).to(
                        torch.float32
                    )
                    # Match the BF16 mask writeback before applying the chain scale.
                    gradient = (gradient.float() * mask).to(gradient.dtype)
                scale = path.scale(name, stepper.task_scale) / anchor_scale
                if path.has_mask(name):
                    scale /= 1 - path.mask_probabilities[name]
                gradient = (gradient.float() * scale).to(gradient.dtype)
                if gradient is not original:
                    original.copy_(gradient)
            if restore:
                parameter.copy_(perturb.backups[index])
                torch.autograd.graph.increment_version(parameter)
        if restore:
            perturb.active.clear()
        return False


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA FastSmat test")
class FastSmatTest(unittest.TestCase):
    @staticmethod
    def settings(interval=1):
        return {
            "method": "smat",
            "backend": "fast",
            "perturb": {"rms": 0.001},
            "scale": {"scope": "global", "alpha_min": 0.2},
            "mask": {"probability": 0.5, "scope": "block-linear"},
            "smat": {"interval": interval},
        }

    def make_pair(self, dtype, interval=1):
        torch.manual_seed(42)
        source = _MixedBucketModel().cuda().to(dtype=dtype)
        tokens = torch.arange(24, device="cuda").reshape(3, 8).remainder(19)
        records = []
        for fast in (False, True):
            model = copy.deepcopy(source)
            named = list(model.named_parameters())
            parameters = [parameter for _, parameter in named]
            optimizer = torch.optim.AdamW(
                parameters, lr=0.003, weight_decay=0.001, fused=True
            )
            settings = self.settings(interval)
            settings["backend"] = "fast" if fast else "fused"
            stepper = build_stepper(
                settings,
                named,
                optimizer,
                seed=42,
                block_linear_names=block_linear_parameter_names(model, named),
                embedding_names=embedding_parameter_names(model, named),
            )
            with torch.no_grad():
                for parameter in parameters:
                    parameter.add_(0.02)
            if fast:
                # Exercise the public factory route, not a manual replacement.
                from smat.train.fast_smat import FastSmat

                self.assertIsInstance(stepper.fused_smat, FastSmat)
            else:
                stepper.fused_smat = OracleBackend(stepper)
            records.append(
                (
                    stepper,
                    model,
                    named,
                    tokens,
                    [parameter.data_ptr() for _, parameter in named],
                )
            )
        return records

    def make_ragged_pair(self, dtype):
        torch.manual_seed(42)
        source = _RaggedGradientModel().cuda().to(dtype=dtype)
        records = []
        for fast in (False, True):
            model = copy.deepcopy(source)
            named = list(model.named_parameters())
            parameters = [parameter for _, parameter in named]
            settings = self.settings()
            settings["backend"] = "fast" if fast else "fused"
            # This fixture has no embedding, so its two Linear weights alone
            # exercise the block-linear mask bucket.
            optimizer = torch.optim.AdamW(
                parameters, lr=0.003, weight_decay=0.001, fused=True
            )
            stepper = build_stepper(
                settings,
                named,
                optimizer,
                seed=42,
                block_linear_names=block_linear_parameter_names(model, named),
                embedding_names=embedding_parameter_names(model, named),
            )
            with torch.no_grad():
                for parameter in parameters:
                    parameter.add_(0.02)
            if fast:
                from smat.train.fast_smat import FastSmat

                self.assertIsInstance(stepper.fused_smat, FastSmat)
            else:
                stepper.fused_smat = OracleBackend(stepper)
            records.append(
                (stepper, named, [parameter.data_ptr() for _, parameter in named])
            )
        return records

    def assert_exact_pairs(self, left, right):
        for (left_name, left_value), (right_name, right_value) in zip(left, right):
            self.assertEqual(left_name, right_name)
            self.assertTrue(torch.equal(left_value, right_value), left_name)

    def assert_optimizer_state(self, left, right):
        for index, (left_parameter, right_parameter) in enumerate(
            zip(left.parameters, right.parameters)
        ):
            left_state, right_state = (
                left.optimizer.state[left_parameter],
                right.optimizer.state[right_parameter],
            )
            self.assertEqual(set(left_state), set(right_state), index)
            for key in left_state:
                left_value, right_value = left_state[key], right_state[key]
                if isinstance(left_value, torch.Tensor):
                    self.assertTrue(torch.equal(left_value, right_value), (index, key))
                else:
                    self.assertEqual(left_value, right_value, (index, key))

    def assert_close_pairs(self, left, right, *, rtol=1e-5, atol=1e-6):
        self.assertEqual(len(left), len(right))
        for (left_name, left_value), (right_name, right_value) in zip(left, right):
            self.assertEqual(left_name, right_name)
            torch.testing.assert_close(
                left_value, right_value, rtol=rtol, atol=atol, msg=left_name
            )

    def assert_optimizer_close(self, left, right, *, rtol=1e-5, atol=1e-6):
        for index, (left_parameter, right_parameter) in enumerate(
            zip(left.parameters, right.parameters)
        ):
            left_state, right_state = (
                left.optimizer.state[left_parameter],
                right.optimizer.state[right_parameter],
            )
            self.assertEqual(set(left_state), set(right_state), index)
            for key in left_state:
                left_value, right_value = left_state[key], right_state[key]
                if isinstance(left_value, torch.Tensor):
                    torch.testing.assert_close(
                        left_value,
                        right_value,
                        rtol=rtol,
                        atol=atol,
                        msg=f"{index}:{key}",
                    )
                else:
                    self.assertEqual(left_value, right_value, (index, key))

    @staticmethod
    def total_grad_norm(parameters):
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        norms = torch._foreach_norm(gradients, 2.0)
        return torch.linalg.vector_norm(torch.stack(norms), 2.0)

    @staticmethod
    def total_tensor_norm(named_tensors):
        norms = torch._foreach_norm([value for _, value in named_tensors], 2.0)
        return torch.linalg.vector_norm(torch.stack(norms), 2.0)

    def record_preclip_norms(self, stepper):
        """Observe the explicit Torch-oracle norm immediately before clipping."""

        original, norms = stepper.finish, []

        def finish(max_grad_norm):
            if max_grad_norm is not None:
                norms.append(self.total_grad_norm(stepper.parameters).detach().clone())
            return original(max_grad_norm)

        stepper.finish = finish
        return norms

    def test_fp32_and_bf16_temporary_weights_gradients_and_adamw_state(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                reference, fast = self.make_pair(dtype)
                reference_stepper, reference_model, reference_named, tokens, _ = (
                    reference
                )
                fast_stepper, fast_model, fast_named, _, fast_pointers = fast
                self.assertIs(fast_model.head.weight, fast_model.embedding.weight)
                self.assertNotIn("head.weight", dict(fast_named))
                captures = [[], []]
                for target, capture in zip((reference_stepper, fast_stepper), captures):
                    target.optimizer.register_step_pre_hook(
                        lambda optimizer, arguments, keywords, out=capture, stepper=target: (
                            out.append(
                                [
                                    (name, parameter.grad.detach().clone())
                                    for name, parameter in stepper.named_parameters
                                ]
                            )
                        )
                    )
                for _ in range(16):
                    temporary, losses = [[], []], []
                    for index, (stepper, model) in enumerate(
                        (
                            (reference_stepper, reference_model),
                            (fast_stepper, fast_model),
                        )
                    ):

                        def closure(
                            current=model,
                            out=temporary[index],
                            parameters=stepper.named_parameters,
                        ):
                            out.extend(
                                (name, parameter.detach().clone())
                                for name, parameter in parameters
                            )
                            output = current(tokens)
                            return (
                                output.float().square().mean()
                                + current.unused.float().square().mean()
                            )

                        # The no-clip path must remain a strict seeded reference.
                        losses.append(stepper.step(closure, max_grad_norm=None))
                        self.assertFalse(stepper.perturbation.active)
                    self.assertTrue(torch.equal(losses[0], losses[1]))
                    self.assert_exact_pairs(temporary[0], temporary[1])
                    self.assert_exact_pairs(captures[0][-1], captures[1][-1])
                self.assert_exact_pairs(reference_named, fast_named)
                self.assert_optimizer_state(reference_stepper, fast_stepper)
                self.assertEqual(
                    fast_pointers, [parameter.data_ptr() for _, parameter in fast_named]
                )
                self.assertIs(fast_model.head.weight, fast_model.embedding.weight)

    def test_clipped_fp32_and_bf16_match_explicit_torch_oracle_within_float_precision(
        self,
    ):
        """Deferred clipping may round differently, but must track the Torch oracle."""

        max_grad_norm = 0.1
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                reference, fast = self.make_pair(dtype)
                reference_stepper, reference_model, reference_named, tokens, _ = (
                    reference
                )
                fast_stepper, fast_model, fast_named, _, _ = fast
                preclip_norms = self.record_preclip_norms(reference_stepper)
                # BF16's spacing is 2**-7 relative; FP32 needs only epsilon slack.
                clipped_norm_slack = (
                    1e-5 if dtype == torch.float32 else max_grad_norm / 128 + 1e-6
                )
                captures = [[], []]
                for target, capture in zip((reference_stepper, fast_stepper), captures):
                    target.optimizer.register_step_pre_hook(
                        lambda optimizer, arguments, keywords, out=capture, stepper=target: (
                            out.append(
                                [
                                    (name, parameter.grad.detach().clone())
                                    for name, parameter in stepper.named_parameters
                                ]
                            )
                        )
                    )
                for step in range(16):
                    temporary, losses = [[], []], []
                    for index, (stepper, model) in enumerate(
                        (
                            (reference_stepper, reference_model),
                            (fast_stepper, fast_model),
                        )
                    ):

                        def closure(
                            current=model,
                            out=temporary[index],
                            parameters=stepper.named_parameters,
                        ):
                            out.extend(
                                (name, parameter.detach().clone())
                                for name, parameter in parameters
                            )
                            output = current(tokens)
                            return (
                                output.float().square().mean()
                                + current.unused.float().square().mean()
                            )

                        losses.append(
                            stepper.step(closure, max_grad_norm=max_grad_norm)
                        )
                    self.assertEqual(len(preclip_norms), step + 1)
                    self.assertGreater(float(preclip_norms[-1]), max_grad_norm)
                    self.assertLessEqual(
                        float(self.total_tensor_norm(captures[0][-1])),
                        max_grad_norm + clipped_norm_slack,
                    )
                    self.assertLessEqual(
                        float(self.total_tensor_norm(captures[1][-1])),
                        max_grad_norm + clipped_norm_slack,
                    )
                    self.assert_close_pairs(temporary[0], temporary[1])
                    self.assert_close_pairs(captures[0][-1], captures[1][-1])
                    torch.testing.assert_close(
                        losses[0], losses[1], rtol=1e-5, atol=1e-6
                    )
                self.assert_close_pairs(reference_named, fast_named)
                self.assert_optimizer_close(reference_stepper, fast_stepper)

    def test_t4_active_then_clean_does_not_reapply_deferred_clip(self):
        """A clean step after an active t=4 step takes ordinary one-pass clipping."""

        max_grad_norm = 0.1
        reference, fast = self.make_pair(torch.bfloat16, interval=4)
        reference_stepper, reference_model, reference_named, tokens, _ = reference
        fast_stepper, fast_model, fast_named, _, _ = fast
        for step in range(4):
            for stepper, model in (
                (reference_stepper, reference_model),
                (fast_stepper, fast_model),
            ):
                stepper.step(
                    lambda current=model: (
                        current(tokens).float().square().mean()
                        + current.unused.float().square().mean()
                    ),
                    max_grad_norm=max_grad_norm,
                )
            self.assertEqual(fast_stepper.fused_smat.steps, int(step == 3))
        before = fast_stepper.fused_smat.generator.get_state().clone()
        for stepper, model in (
            (reference_stepper, reference_model),
            (fast_stepper, fast_model),
        ):
            stepper.step(
                lambda current=model: (
                    current(tokens).float().square().mean()
                    + current.unused.float().square().mean()
                ),
                max_grad_norm=max_grad_norm,
            )
        self.assertTrue(
            torch.equal(before, fast_stepper.fused_smat.generator.get_state())
        )
        self.assertEqual(fast_stepper.fused_smat.steps, 1)
        self.assert_close_pairs(reference_named, fast_named)
        self.assert_optimizer_close(reference_stepper, fast_stepper)

    def test_deferred_clip_fixed_ragged_multi_partial_gradients(self):
        """Two 65x73 matrices exercise multiple norm partials and a ragged tail."""

        max_grad_norm = 0.1
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                reference, fast = self.make_ragged_pair(dtype)
                recorded_gradients = []
                for stepper, named, pointers in (reference, fast):
                    path = stepper.self_path
                    stepper.task_scale = path.sample()
                    path.begin_mask_step(enabled=True)
                    stepper.task_noise_rms = stepper.rms
                    clean = [
                        (name, parameter.detach().clone()) for name, parameter in named
                    ]
                    with stepper.fused_smat.applied():
                        for index, (name, parameter) in enumerate(named):
                            if name == "unused":
                                parameter.grad = None
                                continue
                            parameter.grad = (
                                torch.arange(
                                    parameter.numel(),
                                    device="cuda",
                                    dtype=torch.float32,
                                )
                                .add_(index + 0.25)
                                .reshape(parameter.shape)
                                .to(parameter.dtype)
                            )
                        if stepper is fast[0]:
                            self.assertTrue(
                                stepper.fused_smat.gradients(
                                    stepper.task_scale,
                                    restore=True,
                                    max_grad_norm=max_grad_norm,
                                )
                            )
                        else:
                            self.assertFalse(
                                stepper.fused_smat.gradients(
                                    stepper.task_scale, restore=True
                                )
                            )
                            torch.nn.utils.clip_grad_norm_(
                                stepper.parameters, max_grad_norm
                            )
                    self.assert_exact_pairs(clean, named)
                    self.assertEqual(
                        pointers, [parameter.data_ptr() for _, parameter in named]
                    )
                    self.assertFalse(stepper.perturbation.active)
                    self.assertIsNone(dict(named)["unused"].grad)
                    recorded_gradients.append(
                        [
                            (name, parameter.grad.detach().clone())
                            for name, parameter in named
                            if parameter.grad is not None
                        ]
                    )
                self.assertTrue(
                    any(
                        buffer.shape[0] == 2 and buffer.shape[1] > 1
                        for buffer in fast[0].fused_smat.partial_buffers
                    )
                )
                self.assert_close_pairs(*recorded_gradients)

    @unittest.skipUnless(
        CUDA_AVAILABLE and torch.cuda.device_count() >= 2, "two CUDA devices required"
    )
    def test_deferred_clip_multi_gpu_ragged_gradients_restore_storage(self):
        """Temporary swaps and deferred clipping must guard every parameter device."""

        torch.manual_seed(42)
        source = _RaggedGradientModel()
        source.unused = torch.nn.Parameter(source.unused.detach().cuda(0))
        source.model.layers[0].first.cuda(0)
        source.model.layers[0].second.cuda(1)
        records = []
        for fast in (False, True):
            model = copy.deepcopy(source)
            named = list(model.named_parameters())
            settings = self.settings()
            settings["backend"] = "fast" if fast else "fused"
            stepper = build_stepper(
                settings,
                named,
                torch.optim.SGD([parameter for _, parameter in named], lr=0.001),
                42,
                block_linear_names=block_linear_parameter_names(model, named),
                embedding_names=embedding_parameter_names(model, named),
            )
            with torch.no_grad():
                for _, parameter in named:
                    parameter.add_(0.02)
            if not fast:
                stepper.fused_smat = OracleBackend(stepper)
            records.append(
                (stepper, named, [parameter.data_ptr() for _, parameter in named])
            )

        gradients = []
        for is_fast, (stepper, named, pointers) in zip((False, True), records):
            stepper.task_scale = stepper.self_path.sample()
            stepper.self_path.begin_mask_step(enabled=True)
            stepper.task_noise_rms = stepper.rms
            clean = [(name, parameter.detach().clone()) for name, parameter in named]
            with stepper.fused_smat.applied():
                for index, (name, parameter) in enumerate(named):
                    parameter.grad = (
                        None
                        if name == "unused"
                        else torch.arange(
                            parameter.numel(),
                            device=parameter.device,
                            dtype=torch.float32,
                        )
                        .add_(index + 0.25)
                        .reshape(parameter.shape)
                        .to(parameter.dtype)
                    )
                if is_fast:
                    self.assertTrue(
                        stepper.fused_smat.gradients(
                            stepper.task_scale, restore=True, max_grad_norm=0.1
                        )
                    )
                else:
                    stepper.fused_smat.gradients(stepper.task_scale, restore=True)
                    torch.nn.utils.clip_grad_norm_(stepper.parameters, 0.1)
            self.assert_exact_pairs(clean, named)
            self.assertEqual(pointers, [parameter.data_ptr() for _, parameter in named])
            self.assertFalse(stepper.perturbation.active)
            self.assertIsNone(dict(named)["unused"].grad)
            gradients.append(
                [
                    (name, parameter.grad.detach().clone())
                    for name, parameter in named
                    if parameter.grad is not None
                ]
            )
        self.assert_close_pairs(*gradients)

    def test_none_noncontiguous_and_exception_restore(self):
        reference, fast = self.make_pair(torch.bfloat16)
        for stepper, _, named, _, pointers in (reference, fast):
            stepper.task_scale = stepper.self_path.sample()
            stepper.self_path.begin_mask_step(enabled=True)
            stepper.task_noise_rms = stepper.rms
            clean = [parameter.detach().clone() for _, parameter in named]
            with stepper.fused_smat.applied():
                for name, parameter in named:
                    if name == "unused":
                        parameter.grad = None
                        continue
                    value = torch.arange(
                        parameter.numel(), device="cuda", dtype=torch.float32
                    )
                    value = value.reshape(parameter.shape).to(parameter.dtype)
                    if name == "model.layers.0.attn.weight":
                        value = value.t().contiguous().t()
                        self.assertFalse(value.is_contiguous())
                    parameter.grad = value
                # A strided gradient must retain FastSmat's exact native-clip fallback.
                deferred_clip = stepper.fused_smat.gradients(
                    stepper.task_scale, restore=True, max_grad_norm=0.1
                )
                self.assertFalse(deferred_clip)
                torch.nn.utils.clip_grad_norm_(stepper.parameters, 0.1)
            self.assert_exact_pairs(
                list(zip((name for name, _ in named), clean)), named
            )
            self.assertEqual(pointers, [parameter.data_ptr() for _, parameter in named])
            self.assertFalse(stepper.perturbation.active)
        for (reference_name, reference_parameter), (fast_name, fast_parameter) in zip(
            reference[2], fast[2]
        ):
            self.assertEqual(reference_name, fast_name)
            if reference_name == "unused":
                self.assertIsNone(reference_parameter.grad)
                self.assertIsNone(fast_parameter.grad)
            else:
                self.assertEqual(
                    reference_parameter.grad.stride(), fast_parameter.grad.stride()
                )
                self.assertTrue(
                    torch.equal(reference_parameter.grad, fast_parameter.grad),
                    reference_name,
                )

        _, (stepper, model, named, _, pointers) = self.make_pair(torch.bfloat16)
        clean = [parameter.detach().clone() for _, parameter in named]
        with self.assertRaisesRegex(RuntimeError, "expected closure failure"):
            stepper.step(
                lambda: (_ for _ in ()).throw(RuntimeError("expected closure failure"))
            )
        self.assert_exact_pairs(list(zip((name for name, _ in named), clean)), named)
        self.assertEqual(pointers, [parameter.data_ptr() for _, parameter in named])
        self.assertFalse(stepper.perturbation.active)
        self.assertIs(model.head.weight, model.embedding.weight)

    def test_t4_clean_steps_do_not_advance_stateless_seed_generator(self):
        reference, fast = self.make_pair(torch.bfloat16, interval=4)
        for stepper, model, _, tokens, _ in (reference, fast):
            before = stepper.fused_smat.generator.get_state().clone()
            for _ in range(3):
                stepper.step(
                    lambda current=model: (
                        current(tokens).float().square().mean()
                        + current.unused.float().square().mean()
                    )
                )
            self.assertTrue(
                torch.equal(before, stepper.fused_smat.generator.get_state())
            )
            self.assertEqual(stepper.fused_smat.steps, 0)
            self.assertFalse(stepper.perturbation.active)
            stepper.step(
                lambda current=model: (
                    current(tokens).float().square().mean()
                    + current.unused.float().square().mean()
                )
            )
            self.assertFalse(
                torch.equal(before, stepper.fused_smat.generator.get_state())
            )
            self.assertEqual(stepper.fused_smat.steps, 1)


if __name__ == "__main__":
    unittest.main()
