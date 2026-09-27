"""CUDA regression tests for Fast SMAT's pinned-base offload path."""

import copy
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from smat.train.noise import block_linear_parameter_names, embedding_parameter_names
from smat.train.updates import build_stepper

try:  # unittest discovery puts ``tests`` itself on sys.path.
    from test_fast_smat import _MixedBucketModel, _RaggedGradientModel
except ModuleNotFoundError:  # pragma: no cover - supports module-style discovery too.
    from tests.test_fast_smat import _MixedBucketModel, _RaggedGradientModel


class _RaggedDtypeParameters(torch.nn.Module):
    """Standalone parameters whose byte ranges cross dtype boundaries."""

    def __init__(self):
        super().__init__()
        self.float_prefix = torch.nn.Parameter(torch.arange(5, dtype=torch.float32))
        self.float16_middle = torch.nn.Parameter(torch.arange(7, dtype=torch.float16))
        self.bfloat16_middle = torch.nn.Parameter(
            torch.arange(11, dtype=torch.bfloat16)
        )
        self.float_suffix = torch.nn.Parameter(torch.arange(3, dtype=torch.float32))


CUDA_AVAILABLE = torch.cuda.is_available()


def _settings(*, interval=4, mode="joint", components=None, offload_base=False):
    smat = {"interval": interval, "mode": mode}
    if components is not None:
        smat["components"] = list(components)
    return {
        "method": "smat",
        "backend": "fast",
        "offload_base": offload_base,
        "perturb": {"rms": 0.001},
        "scale": {"scope": "global", "alpha_min": 0.2},
        "mask": {"probability": 0.5, "scope": "block-linear"},
        "smat": smat,
    }


def _named_rows(path):
    return [(name, value.detach().clone()) for name, value in path]


def _assert_exact_named(test, expected, actual):
    test.assertEqual(len(expected), len(actual))
    for (expected_name, expected_value), (actual_name, actual_value) in zip(
        expected, actual
    ):
        test.assertEqual(expected_name, actual_name)
        test.assertTrue(torch.equal(expected_value, actual_value), expected_name)


def _assert_exact_optimizer(test, expected, actual):
    for index, (expected_parameter, actual_parameter) in enumerate(
        zip(expected.parameters, actual.parameters)
    ):
        # ``optimizer.state`` is a defaultdict: indexing an unused parameter
        # would create exactly the empty entry this regression guards against.
        expected_state = expected.optimizer.state.get(expected_parameter, {})
        actual_state = actual.optimizer.state.get(actual_parameter, {})
        test.assertEqual(set(expected_state), set(actual_state), index)
        for key in expected_state:
            left, right = expected_state[key], actual_state[key]
            if isinstance(left, torch.Tensor):
                test.assertTrue(torch.equal(left, right), (index, key))
            else:
                test.assertEqual(left, right, (index, key))


def _cuda_storage_bytes(tensors):
    """Count each live CUDA storage once, including aliases/views only once."""

    seen, total = set(), 0
    for tensor in tensors:
        if tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def _assert_finite_loss(test, loss, *, dtype, step, backend):
    test.assertTrue(
        bool(torch.isfinite(loss).item()),
        f"dtype={dtype}, step={step}, backend={backend}, loss={float(loss.detach())!r}",
    )


class OffloadBaseValidationTest(unittest.TestCase):
    def test_offload_base_requires_a_boolean_fast_backend(self):
        model = torch.nn.Linear(2, 2)
        named = list(model.named_parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        non_boolean = _settings(offload_base=True)
        non_boolean["offload_base"] = 1
        with self.assertRaises(ValueError):
            build_stepper(non_boolean, named, optimizer, seed=1)

        non_fast = _settings(offload_base=True)
        non_fast["backend"] = "fused"
        with self.assertRaises(ValueError):
            build_stepper(non_fast, named, optimizer, seed=1)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA base-offload tests")
class FastSmatBaseOffloadTest(unittest.TestCase):
    def make_pair(self, dtype, *, interval=4):
        torch.manual_seed(731)
        source = _MixedBucketModel().cuda().to(dtype=dtype)
        tokens = torch.arange(24, device="cuda").reshape(3, 8).remainder(19)
        records = []
        for offload_base in (False, True):
            model = copy.deepcopy(source)
            named = list(model.named_parameters())
            # FP16 fused AdamW needs an epsilon representable at its update
            # precision; keep the fixture stable without relaxing equality.
            optimizer = torch.optim.AdamW(
                [parameter for _, parameter in named],
                lr=0.001,
                weight_decay=0.001,
                eps=1e-4,
                fused=True,
            )
            stepper = build_stepper(
                _settings(interval=interval, offload_base=offload_base),
                named,
                optimizer,
                seed=731,
                block_linear_names=block_linear_parameter_names(model, named),
                embedding_names=embedding_parameter_names(model, named),
            )
            with torch.no_grad():
                for _, parameter in named:
                    parameter.add_(0.02)
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

    def make_ragged_dtype_pair(self):
        """Create byte-ragged FP32/FP16/BF16 bases without model-name special cases."""

        torch.manual_seed(811)
        source = _RaggedDtypeParameters().cuda()
        records = []
        for offload_base in (False, True):
            model = copy.deepcopy(source)
            named = list(model.named_parameters())
            settings = _settings(
                interval=4, components=("scale", "mask"), offload_base=offload_base
            )
            settings["mask"]["probability"] = 0.0
            stepper = build_stepper(
                settings,
                named,
                torch.optim.SGD(model.parameters(), lr=0.001),
                seed=811,
                block_linear_names=set(),
                embedding_names=set(),
            )
            with torch.no_grad():
                for _, parameter in named:
                    parameter.add_(0.02)
            records.append((stepper, model, named))
        return records

    @staticmethod
    def _loss(model, tokens, capture=None):
        if capture is not None:
            capture.extend(_named_rows(model.named_parameters()))
        output = model(tokens)
        return output.float().square().mean()

    @staticmethod
    def _checkpoint_loss(model, tokens, *, use_reentrant):
        hidden = model.embedding(tokens)

        def tail(value):
            for layer in model.model.layers:
                value = layer(value)
            return model.head(value)

        output = checkpoint(tail, hidden, use_reentrant=use_reentrant)
        return output.float().square().mean()

    @staticmethod
    def _synchronize_devices():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

    def _assert_offloaded_base(self, stepper):
        path = stepper.self_path
        self.assertTrue(all(base.device.type == "cpu" for base in path.base.values()))
        self.assertTrue(all(base.is_pinned() for base in path.base.values()))

    def _assert_complete_base_cache(self, stepper):
        """Check a fully queued refill only after every byte batch was issued."""

        self._synchronize_devices()
        for (name, parameter), temporary in zip(
            stepper.named_parameters, stepper.perturbation.backups, strict=True
        ):
            expected = stepper.self_path.base[name].to(parameter.device)
            self.assertTrue(torch.equal(temporary, expected), name)

    def _prefetch_all_base(self, stepper):
        """Flush outstanding batches without observing or mutating optimizer state."""

        fast = stepper.fused_smat
        self.assertGreater(fast.prefetch_count, 0)
        while fast.prefetch_index < fast.prefetch_count:
            fast.prefetch_base()
        self.assertEqual(fast.prefetch_index, fast.prefetch_count)
        self._assert_complete_base_cache(stepper)

    def _assert_byte_batch_coverage(self, stepper):
        """Each pinned-base element must occur in exactly one queued copy slice."""

        fast = stepper.fused_smat
        for device, batches in fast.base_batches.items():
            roots = []
            for (name, parameter), temporary in zip(
                stepper.named_parameters, stepper.perturbation.backups, strict=True
            ):
                if parameter.device != device:
                    continue
                host = stepper.self_path.base[name].view(-1)
                roots.append(
                    (
                        name,
                        host,
                        temporary.view(-1),
                        torch.zeros(host.numel(), dtype=torch.int64),
                    )
                )

            for batch in batches:
                for host_slice, temporary_slice in batch:
                    matching = [
                        root
                        for root in roots
                        if root[1].data_ptr()
                        <= host_slice.data_ptr()
                        < root[1].data_ptr() + root[1].numel() * root[1].element_size()
                    ]
                    self.assertEqual(len(matching), 1)
                    name, host, temporary, coverage = matching[0]
                    byte_offset = host_slice.data_ptr() - host.data_ptr()
                    self.assertEqual(byte_offset % host.element_size(), 0, name)
                    start = byte_offset // host.element_size()
                    end = start + host_slice.numel()
                    self.assertEqual(
                        temporary_slice.data_ptr() - temporary.data_ptr(),
                        byte_offset,
                        name,
                    )
                    self.assertEqual(temporary_slice.numel(), host_slice.numel(), name)
                    coverage[start:end].add_(1)

            for name, _, _, coverage in roots:
                self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)), name)

    def _capture_applied_weights(self, stepper, named):
        path = stepper.self_path
        stepper.task_scale = path.sample()
        path.begin_mask_step(enabled=True)
        stepper.task_noise_rms = stepper.rms
        with stepper.fused_smat.applied():
            return _named_rows(named)

    def _assert_pair_equal(self, resident, offloaded):
        resident_stepper, resident_model, resident_named, _, resident_pointers = (
            resident
        )
        offloaded_stepper, offloaded_model, offloaded_named, _, offloaded_pointers = (
            offloaded
        )
        self._synchronize_devices()
        _assert_exact_named(
            self, _named_rows(resident_named), _named_rows(offloaded_named)
        )
        _assert_exact_optimizer(self, resident_stepper, offloaded_stepper)
        self.assertEqual(
            resident_pointers, [parameter.data_ptr() for _, parameter in resident_named]
        )
        self.assertEqual(
            offloaded_pointers,
            [parameter.data_ptr() for _, parameter in offloaded_named],
        )
        self.assertIs(resident_model.head.weight, resident_model.embedding.weight)
        self.assertIs(offloaded_model.head.weight, offloaded_model.embedding.weight)
        resident_unused = dict(resident_named)["unused"]
        offloaded_unused = dict(offloaded_named)["unused"]
        self.assertIsNone(resident_unused.grad)
        self.assertIsNone(offloaded_unused.grad)
        self.assertNotIn(resident_unused, resident_stepper.optimizer.state)
        self.assertNotIn(offloaded_unused, offloaded_stepper.optimizer.state)

    def test_requires_joint_interval_and_a_base_dependent_component(self):
        source = _MixedBucketModel().cuda()
        named = list(source.named_parameters())
        blocks = block_linear_parameter_names(source, named)
        embeddings = embedding_parameter_names(source, named)

        for specification in (
            dict(interval=1),
            dict(mode="cycle"),
            dict(mode="independent"),
            dict(components=("perturb",)),
            dict(components=()),
        ):
            with self.subTest(specification=specification):
                settings = _settings(offload_base=True, **specification)
                optimizer = torch.optim.AdamW(
                    source.parameters(), lr=0.001, eps=1e-4, fused=True
                )
                with self.assertRaises(ValueError):
                    build_stepper(
                        settings,
                        named,
                        optimizer,
                        seed=731,
                        block_linear_names=blocks,
                        embedding_names=embeddings,
                    )

        for components in (("scale",), ("mask",), ("perturb", "scale", "mask")):
            with self.subTest(components=components):
                settings = _settings(offload_base=True, components=components)
                optimizer = torch.optim.AdamW(
                    source.parameters(), lr=0.001, eps=1e-4, fused=True
                )
                stepper = build_stepper(
                    settings,
                    named,
                    optimizer,
                    seed=731,
                    block_linear_names=blocks,
                    embedding_names=embeddings,
                )
                self._assert_offloaded_base(stepper)

        settings = _settings()
        settings.pop("offload_base")
        optimizer = torch.optim.AdamW(
            source.parameters(), lr=0.001, eps=1e-4, fused=True
        )
        resident = build_stepper(
            settings,
            named,
            optimizer,
            seed=731,
            block_linear_names=blocks,
            embedding_names=embeddings,
        )
        self.assertTrue(all(base.is_cuda for base in resident.self_path.base.values()))

    def test_t4_offload_matches_resident_fast_for_three_precisions_with_clip(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                resident, offloaded = self.make_pair(dtype)
                resident_stepper, resident_model, resident_named, tokens, _ = resident
                offloaded_stepper, offloaded_model, offloaded_named, _, _ = offloaded
                self._assert_offloaded_base(offloaded_stepper)
                for step in range(12):
                    temporary, losses = [[], []], []
                    for stepper, model, named, capture in (
                        (
                            resident_stepper,
                            resident_model,
                            resident_named,
                            temporary[0],
                        ),
                        (
                            offloaded_stepper,
                            offloaded_model,
                            offloaded_named,
                            temporary[1],
                        ),
                    ):
                        losses.append(
                            stepper.step(
                                lambda current=model, out=capture: self._loss(
                                    current, tokens, out
                                ),
                                max_grad_norm=0.1,
                            )
                        )
                    _assert_finite_loss(
                        self, losses[0], dtype=dtype, step=step, backend="resident"
                    )
                    _assert_finite_loss(
                        self, losses[1], dtype=dtype, step=step, backend="offload"
                    )
                    self.assertTrue(torch.equal(losses[0], losses[1]), step)
                    _assert_exact_named(self, temporary[0], temporary[1])
                    self.assertIsNone(dict(resident_named)["unused"].grad)
                    self.assertIsNone(dict(offloaded_named)["unused"].grad)
                    self.assertNotIn(
                        dict(resident_named)["unused"], resident_stepper.optimizer.state
                    )
                    self.assertNotIn(
                        dict(offloaded_named)["unused"],
                        offloaded_stepper.optimizer.state,
                    )
                    self.assertEqual(
                        offloaded_stepper.fused_smat.steps, (step + 1) // 4
                    )
                    if step >= 4:
                        self.assertEqual(
                            offloaded_stepper.fused_smat.prefetch_index,
                            (step + 1) % 4,
                            f"step {step + 1} should refill one batch or reset after active SMAT",
                        )
                self._assert_pair_equal(resident, offloaded)
                self._prefetch_all_base(offloaded_stepper)

    def test_joint_intervals_t2_t3_and_t5_match_resident_fast(self):
        """Two full clean/active cycles cover every supported prefetch count."""

        for interval in (2, 3, 5):
            with self.subTest(interval=interval):
                resident, offloaded = self.make_pair(torch.bfloat16, interval=interval)
                resident_stepper, resident_model, resident_named, tokens, _ = resident
                offloaded_stepper, offloaded_model, offloaded_named, _, _ = offloaded
                for step in range(2 * interval):
                    losses = []
                    for stepper, model in (
                        (resident_stepper, resident_model),
                        (offloaded_stepper, offloaded_model),
                    ):
                        losses.append(
                            stepper.step(
                                lambda current=model: self._loss(current, tokens),
                                max_grad_norm=0.1,
                            )
                        )
                    _assert_finite_loss(
                        self,
                        losses[0],
                        dtype=torch.bfloat16,
                        step=step,
                        backend=f"resident-t{interval}",
                    )
                    _assert_finite_loss(
                        self,
                        losses[1],
                        dtype=torch.bfloat16,
                        step=step,
                        backend=f"offload-t{interval}",
                    )
                    self.assertTrue(torch.equal(losses[0], losses[1]), (interval, step))
                    _assert_exact_named(
                        self, _named_rows(resident_named), _named_rows(offloaded_named)
                    )
                self._assert_pair_equal(resident, offloaded)
                self._prefetch_all_base(offloaded_stepper)

    def test_offload_reuses_backups_and_reduces_cuda_state_storage(self):
        resident, offloaded = self.make_pair(torch.bfloat16)
        resident_stepper = resident[0]
        offloaded_stepper = offloaded[0]
        resident_base_bytes = _cuda_storage_bytes(
            resident_stepper.self_path.base.values()
        )
        offloaded_base_bytes = _cuda_storage_bytes(
            offloaded_stepper.self_path.base.values()
        )
        resident_state_bytes = _cuda_storage_bytes(
            [
                *resident_stepper.self_path.base.values(),
                *resident_stepper.perturbation.backups,
            ]
        )
        offloaded_state_bytes = _cuda_storage_bytes(
            [
                *offloaded_stepper.self_path.base.values(),
                *offloaded_stepper.perturbation.backups,
            ]
        )
        host_base_bytes = sum(
            base.untyped_storage().nbytes()
            for base in offloaded_stepper.self_path.base.values()
        )

        self._assert_offloaded_base(offloaded_stepper)
        self.assertGreater(resident_base_bytes, 0)
        self.assertEqual(offloaded_base_bytes, 0)
        self.assertEqual(host_base_bytes, resident_base_bytes)
        self.assertEqual(
            resident_state_bytes - offloaded_state_bytes, resident_base_bytes
        )

    def test_byte_partition_refills_every_ragged_mixed_dtype_base_exactly_once(self):
        resident, offloaded = self.make_ragged_dtype_pair()
        resident_stepper, _, resident_named = resident
        offloaded_stepper, _, offloaded_named = offloaded

        # The initial backup clones are valid for the first active call. This
        # checks that reading the temporary base is bit-exact across dtypes.
        resident_applied = self._capture_applied_weights(
            resident_stepper, resident_named
        )
        offloaded_applied = self._capture_applied_weights(
            offloaded_stepper, offloaded_named
        )
        _assert_exact_named(self, resident_applied, offloaded_applied)
        self.assertEqual(offloaded_stepper.fused_smat.prefetch_index, 0)

        # With every temporary poisoned, a complete refill proves that each
        # byte-partition reaches the right tensor element. The independent
        # coverage accounting rejects both holes and duplicate slices.
        for temporary in offloaded_stepper.perturbation.backups:
            temporary.fill_(-123)
        self._assert_byte_batch_coverage(offloaded_stepper)
        self._prefetch_all_base(offloaded_stepper)

        # A second active call consumes the refilled cache rather than the
        # original clones, while preserving the resident Fast output exactly.
        resident_applied = self._capture_applied_weights(
            resident_stepper, resident_named
        )
        offloaded_applied = self._capture_applied_weights(
            offloaded_stepper, offloaded_named
        )
        _assert_exact_named(self, resident_applied, offloaded_applied)

    def test_exception_restores_then_a_later_active_step_waits_for_prefetch(self):
        resident, offloaded = self.make_pair(torch.bfloat16)
        for _ in range(3):
            for stepper, model, _, tokens, _ in (resident, offloaded):
                stepper.step(lambda current=model: self._loss(current, tokens))

        before = [_named_rows(record[2]) for record in (resident, offloaded)]
        for stepper, _, _, _, _ in (resident, offloaded):
            with self.assertRaisesRegex(RuntimeError, "expected closure failure"):
                stepper.step(
                    lambda: (_ for _ in ()).throw(
                        RuntimeError("expected closure failure")
                    )
                )
            self.assertFalse(stepper.perturbation.active)
        self._synchronize_devices()
        _assert_exact_named(self, before[0], _named_rows(resident[2]))
        _assert_exact_named(self, before[1], _named_rows(offloaded[2]))
        self._prefetch_all_base(offloaded[0])

        # The failed t=4 step does not advance the optimizer step index, so this
        # is another active step and must wait for the copy scheduled above.
        losses = []
        for stepper, model, _, tokens, _ in (resident, offloaded):
            losses.append(
                stepper.step(lambda current=model: self._loss(current, tokens))
            )
        _assert_finite_loss(
            self, losses[0], dtype=torch.bfloat16, step=4, backend="resident"
        )
        _assert_finite_loss(
            self, losses[1], dtype=torch.bfloat16, step=4, backend="offload"
        )
        self.assertTrue(torch.equal(losses[0], losses[1]))
        self._assert_pair_equal(resident, offloaded)

    def test_checkpoint_reentrant_and_nonreentrant_match_resident_fast(self):
        for use_reentrant in (True, False):
            with self.subTest(use_reentrant=use_reentrant):
                resident, offloaded = self.make_pair(torch.bfloat16)
                for step in range(8):
                    losses = []
                    for stepper, model, _, tokens, _ in (resident, offloaded):
                        losses.append(
                            stepper.step(
                                lambda current=model: self._checkpoint_loss(
                                    current, tokens, use_reentrant=use_reentrant
                                ),
                                max_grad_norm=0.1,
                            )
                        )
                    _assert_finite_loss(
                        self,
                        losses[0],
                        dtype=torch.bfloat16,
                        step=step,
                        backend="resident",
                    )
                    _assert_finite_loss(
                        self,
                        losses[1],
                        dtype=torch.bfloat16,
                        step=step,
                        backend="offload",
                    )
                    self.assertTrue(torch.equal(losses[0], losses[1]), step)
                self._assert_pair_equal(resident, offloaded)
                self._prefetch_all_base(offloaded[0])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_multi_gpu_prefetch_and_gradients_match_resident_fast(self):
        torch.manual_seed(919)
        source = _RaggedGradientModel()
        source.unused = torch.nn.Parameter(source.unused.detach().cuda(0))
        source.model.layers[0].first.cuda(0)
        source.model.layers[0].second.cuda(1)
        records = []
        for offload_base in (False, True):
            model = copy.deepcopy(source)
            named = list(model.named_parameters())
            settings = _settings(offload_base=offload_base)
            stepper = build_stepper(
                settings,
                named,
                torch.optim.SGD([parameter for _, parameter in named], lr=0.001),
                919,
                block_linear_names=block_linear_parameter_names(model, named),
                embedding_names=embedding_parameter_names(model, named),
            )
            with torch.no_grad():
                for _, parameter in named:
                    parameter.add_(0.02)
            records.append(
                (stepper, named, [parameter.data_ptr() for _, parameter in named])
            )

        gradients = []
        for _ in range(
            2
        ):  # The second active pass consumes the async prefetch from the first.
            for stepper, named, pointers in records:
                path = stepper.self_path
                stepper.task_scale = path.sample()
                path.begin_mask_step(enabled=True)
                stepper.task_noise_rms = stepper.rms
                clean = _named_rows(named)
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
                    self.assertTrue(
                        stepper.fused_smat.gradients(
                            stepper.task_scale, restore=True, max_grad_norm=0.1
                        )
                    )
                self._synchronize_devices()
                _assert_exact_named(self, clean, _named_rows(named))
                self.assertEqual(
                    pointers, [parameter.data_ptr() for _, parameter in named]
                )
                self.assertFalse(stepper.perturbation.active)
                gradients.append(
                    [
                        (name, parameter.grad.detach().clone())
                        for name, parameter in named
                        if parameter.grad is not None
                    ]
                )
        for left, right in zip(gradients[::2], gradients[1::2], strict=True):
            _assert_exact_named(self, left, right)

        offloaded_stepper = records[1][0]
        self._assert_offloaded_base(offloaded_stepper)
        self._prefetch_all_base(offloaded_stepper)


if __name__ == "__main__":
    unittest.main()
