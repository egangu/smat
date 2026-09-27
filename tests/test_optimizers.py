import copy
import unittest

import torch

from smat.train.optimizers import CombinedOptimizer, build_optimizer


def _settings(optimizer: str) -> dict:
    return {
        "optimizer": optimizer,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "muon": {
            "learning_rate": 0.1,
            "momentum": 0.95,
            "ns_steps": 5,
            "adjust_lr_fn": "original",
        },
    }


class OptimizerFactoryTest(unittest.TestCase):
    def test_adam_and_adamw_keep_their_existing_defaults(self):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        for name, expected in (
            ("adam", torch.optim.Adam),
            ("adamw", torch.optim.AdamW),
        ):
            with self.subTest(optimizer=name):
                optimizer = build_optimizer(_settings(name), [("weight", parameter)])
                self.assertIsInstance(optimizer, expected)
                self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)

    def test_muon_uses_only_hidden_transformer_matrix_weights(self):
        clip_layer = torch.nn.Parameter(torch.ones(2, 3))
        trace_layer = torch.nn.Parameter(torch.ones(3, 2))
        position_embedding = torch.nn.Parameter(torch.ones(4, 3))
        lm_head = torch.nn.Parameter(torch.ones(3, 4))
        bias = torch.nn.Parameter(torch.ones(3))
        optimizer = build_optimizer(
            _settings("muon-adamw"),
            [
                ("vision_model.encoder.layers.0.mlp.fc1.weight", clip_layer),
                ("model.layers.1.self_attn.q_proj.weight", trace_layer),
                (
                    "vision_model.embeddings.position_embedding.weight",
                    position_embedding,
                ),
                ("lm_head.weight", lm_head),
                ("model.layers.1.self_attn.q_proj.bias", bias),
            ],
        )

        self.assertIsInstance(optimizer, CombinedOptimizer)
        muon, adamw = optimizer.optimizers
        self.assertIsInstance(muon, torch.optim.Muon)
        self.assertIsInstance(adamw, torch.optim.AdamW)
        self.assertEqual(muon.param_groups[0]["params"], [clip_layer, trace_layer])
        self.assertEqual(
            adamw.param_groups[0]["params"], [position_embedding, lm_head, bias]
        )
        grouped = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(grouped), len(set(map(id, grouped))))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {
                id(parameter)
                for parameter in (
                    clip_layer,
                    trace_layer,
                    position_embedding,
                    lm_head,
                    bias,
                )
            },
        )

    def test_muon_and_adamw_both_update_and_zero_grad(self):
        matrix = torch.nn.Parameter(torch.randn(3, 2))
        other = torch.nn.Parameter(torch.randn(3))
        optimizer = build_optimizer(
            _settings("muon-adamw"),
            [
                ("model.layers.0.mlp.down_proj.weight", matrix),
                ("model.norm.weight", other),
            ],
        )
        before_matrix, before_other = matrix.detach().clone(), other.detach().clone()
        matrix.grad = torch.randn_like(matrix)
        other.grad = torch.randn_like(other)

        optimizer.step()

        self.assertFalse(torch.equal(matrix, before_matrix))
        self.assertFalse(torch.equal(other, before_other))
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(matrix.grad)
        self.assertIsNone(other.grad)

    def test_one_scheduler_scales_both_native_learning_rates(self):
        matrix = torch.nn.Parameter(torch.ones(2, 3))
        other = torch.nn.Parameter(torch.ones(3))
        optimizer = build_optimizer(
            _settings("muon-adamw"),
            [
                ("model.layers.0.mlp.gate_proj.weight", matrix),
                ("model.norm.weight", other),
            ],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 1.0 if step == 0 else 0.25,
        )

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.1, 0.01])
        matrix.grad = torch.ones_like(matrix)
        other.grad = torch.ones_like(other)
        optimizer.step()
        scheduler.step()

        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups], [0.025, 0.0025]
        )
        self.assertEqual(
            [
                group["lr"]
                for child in optimizer.optimizers
                for group in child.param_groups
            ],
            [0.025, 0.0025],
        )

    def test_combined_optimizer_state_round_trip(self):
        matrix = torch.nn.Parameter(torch.ones(2, 3))
        other = torch.nn.Parameter(torch.ones(3))
        named = [
            ("model.layers.0.mlp.gate_proj.weight", matrix),
            ("model.norm.weight", other),
        ]
        optimizer = build_optimizer(_settings("muon-adamw"), named)
        matrix.grad = torch.ones_like(matrix)
        other.grad = torch.ones_like(other)
        optimizer.step()
        state = optimizer.state_dict()

        restored_matrix = torch.nn.Parameter(matrix.detach().clone())
        restored_other = torch.nn.Parameter(other.detach().clone())
        restored = build_optimizer(
            _settings("muon-adamw"),
            [
                ("model.layers.0.mlp.gate_proj.weight", restored_matrix),
                ("model.norm.weight", restored_other),
            ],
        )
        restored.load_state_dict(state)

        self.assertEqual(state["format"], "smat.combined-optimizer.v1")
        self.assertIn("momentum_buffer", restored.optimizers[0].state[restored_matrix])
        self.assertIn("exp_avg", restored.optimizers[1].state[restored_other])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            restored,
            lambda step: 1.0 if step == 0 else 0.5,
        )
        restored_matrix.grad = torch.ones_like(restored_matrix)
        restored_other.grad = torch.ones_like(restored_other)
        restored.step()
        scheduler.step()
        self.assertEqual(
            [
                group["lr"]
                for child in restored.optimizers
                for group in child.param_groups
            ],
            [0.05, 0.005],
        )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class MultiDeviceMuonTest(unittest.TestCase):
    def test_native_updates_and_checkpoints_match_across_resume(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                reference = [
                    torch.nn.Parameter(
                        torch.randn(shape, device=f"cuda:{device}", dtype=dtype)
                    )
                    for device, shape in [
                        (0, (32, 96)),
                        (0, (96, 32)),
                        (0, (24, 48)),
                        (1, (32, 96)),
                        (1, (48, 24)),
                    ]
                ]
                candidate = [torch.nn.Parameter(p.detach().clone()) for p in reference]
                names = [f"model.layers.{i}.mlp.weight" for i in range(len(reference))]
                settings = _settings("muon-adamw")
                baseline = torch.optim.Muon(
                    reference, lr=0.1, weight_decay=0.0, adjust_lr_fn="original"
                )
                optimizer = build_optimizer(settings, zip(names, candidate))
                native = optimizer.optimizers[0]
                for step in range(5):
                    baseline.param_groups[0]["lr"] = 0.1 / (step + 1)
                    native.param_groups[0]["lr"] = 0.1 / (step + 1)
                    for p, q in zip(reference, candidate):
                        gradient = torch.randn_like(p)
                        p.grad, q.grad = gradient.clone(), gradient.clone()
                    baseline.step()
                    optimizer.step()
                    self.assertEqual(
                        list(map(id, native.param_groups[0]["params"])),
                        list(map(id, candidate)),
                    )
                    for p, q in zip(reference, candidate):
                        self.assertTrue(torch.equal(p, q))
                        self.assertTrue(
                            torch.equal(
                                baseline.state[p]["momentum_buffer"],
                                native.state[q]["momentum_buffer"],
                            )
                        )
                    expected, actual = baseline.state_dict(), native.state_dict()
                    self.assertEqual(expected["param_groups"], actual["param_groups"])
                    for key in expected["state"]:
                        self.assertTrue(
                            torch.equal(
                                expected["state"][key]["momentum_buffer"],
                                actual["state"][key]["momentum_buffer"],
                            )
                        )
                    if step == 2:
                        # Load old native ordering into the new wrapper, including new group dicts.
                        optimizer.load_state_dict(
                            {
                                "format": "smat.combined-optimizer.v1",
                                "optimizers": [copy.deepcopy(expected)],
                            }
                        )

    def test_failed_step_restores_checkpoint_parameter_order(self):
        parameters = [
            torch.nn.Parameter(torch.ones(2, 3, device=f"cuda:{d}")) for d in (0, 0, 1)
        ]
        settings = _settings("muon-adamw")
        settings["muon"]["ns_steps"] = 100
        optimizer = build_optimizer(
            settings,
            [(f"model.layers.{i}.mlp.weight", p) for i, p in enumerate(parameters)],
        )
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        with self.assertRaises(ValueError):
            optimizer.step()
        self.assertEqual(
            list(map(id, optimizer.param_groups[0]["params"])),
            list(map(id, parameters)),
        )


if __name__ == "__main__":
    unittest.main()
