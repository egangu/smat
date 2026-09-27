"""Scientific behavior at the FT/SMAT update boundary."""

import unittest

import torch

from smat.train.updates import build_stepper


def smat_settings(**overrides):
    settings = {
        "method": "smat",
        "backend": "eager",
        "scale": {"alpha_min": 0.2, "scope": "global"},
        "mask": {"probability": 0.0, "scope": "block-linear"},
        "perturb": {"rms": 0.0},
        "smat": {"interval": 1},
    }
    settings.update(overrides)
    return settings


class UpdatesTest(unittest.TestCase):
    def test_ft_is_one_ordinary_optimizer_step(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        q = torch.nn.Parameter(p.detach().clone())
        optimizer = torch.optim.AdamW([p], lr=0.01)
        reference = torch.optim.AdamW([q], lr=0.01)
        stepper = build_stepper({"method": "ft"}, [("weight", p)], optimizer, 42)
        stepper.step(lambda: p.square().sum())
        q.square().sum().backward()
        reference.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_scale_gradient_and_restore_before_optimizer(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([p], lr=0.1)
        stepper = build_stepper(smat_settings(), [("weight", p)], optimizer, 42)
        with torch.no_grad():
            p.fill_(3.0)
        seen = []
        optimizer.register_step_pre_hook(
            lambda *_: seen.append((p.detach().clone(), p.grad.detach().clone()))
        )
        simulated = []

        def closure():
            simulated.append(p.detach().clone())
            return p.square().sum()

        stepper.step(closure)
        # q=1+a*(3-1); the gradient at theta is a*2*q, not just 2*q.
        q = simulated[0]
        a = (q - 1.0) / 2.0
        torch.testing.assert_close(seen[0][0], torch.tensor([3.0]))
        torch.testing.assert_close(seen[0][1], a * 2 * q)
        torch.testing.assert_close(p, 3.0 - 0.1 * a * 2 * q)

    def test_mask_rescales_only_selected_update_coordinates(self):
        p = torch.nn.Parameter(torch.ones(256))
        optimizer = torch.optim.SGD([p], lr=0.0)
        settings = smat_settings(
            scale={"alpha_min": 1.0, "scope": "global"},
            mask={"probability": 0.5, "scope": "block-linear"},
        )
        stepper = build_stepper(
            settings, [("weight", p)], optimizer, 7, block_linear_names={"weight"}
        )
        with torch.no_grad():
            p.fill_(2.0)
        seen = []
        gradients = []
        optimizer.register_step_pre_hook(lambda *_: gradients.append(p.grad.clone()))

        def closure():
            seen.append(p.detach().clone())
            return p.sum()

        stepper.step(closure)
        self.assertEqual(set(seen[0].tolist()), {1.0, 3.0})
        torch.testing.assert_close(gradients[0], seen[0] - 1.0)
        torch.testing.assert_close(p, torch.full_like(p, 2.0))

    def test_clean_steps_do_not_consume_operator_randomness(self):
        snapshots = []
        for interval in (1, 4):
            p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            optimizer = torch.optim.SGD([p], lr=0.0)
            settings = smat_settings(smat={"interval": interval}, perturb={"rms": 0.01})
            stepper = build_stepper(settings, [("weight", p)], optimizer, 42)
            with torch.no_grad():
                p.add_(0.5)
            seen = []

            def closure():
                seen.append(p.detach().clone())
                return p.square().sum()

            for _ in range(interval):
                stepper.step(closure)
            snapshots.append(seen[-1])
        torch.testing.assert_close(*snapshots, rtol=0, atol=0)

    def test_failed_forward_restores_expert(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        stepper = build_stepper(
            smat_settings(perturb={"rms": 0.1}),
            [("w", p)],
            torch.optim.SGD([p], lr=0.1),
            42,
        )
        with torch.no_grad():
            p.fill_(2.0)

        def fail():
            raise RuntimeError("test forward failure")

        with self.assertRaisesRegex(RuntimeError, "test forward failure"):
            stepper.step(fail)
        torch.testing.assert_close(p, torch.tensor([2.0]), rtol=0, atol=0)

    def test_unknown_method_is_rejected(self):
        p = torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            build_stepper(
                {"method": "unsupported"}, [("w", p)], torch.optim.SGD([p], lr=0.1), 42
            )


if __name__ == "__main__":
    unittest.main()
