import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from smat.merge_sparse import sparse_merge, ties_update
from smat.suite import merge_settings, method_settings, merge_suite
from tests.test_merge import MemoryAdapter


class SuiteTest(unittest.TestCase):
    def test_merger_settings_reject_invalid_grid_before_model_reads(self):
        self.assertEqual(merge_settings({})["scale"], 0.3)
        self.assertEqual(method_settings("wa", merge_settings({})), {})
        for values in [
            dict(scale=float("nan")),
            dict(ties_retention=0),
            dict(della_drop=0.05, della_window=0.14),
            dict(retension=0.5),
            dict(ties_scale=0),
            dict(dare_scale=-1),
            dict(della_scale=float("nan")),
        ]:
            with self.assertRaises(ValueError):
                merge_settings({"merging": values})

    def test_per_method_scales_reach_merger_and_receipt_without_changing_ta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(
                experiment="clip8_vit",
                name="fixture",
                work_root=str(root),
                base_model="base",
                tasks=["one"],
                merging=dict(
                    scale=0.3, ties_scale=0.45, dare_scale=0.2, della_scale=0.6
                ),
            )
            expert = root / "experts" / "fixture" / "one"
            expert.mkdir(parents=True)
            (expert / "metadata.json").write_text("{}")

            def write_checkpoint(adapter, base, experts, output, *, scale, **kwargs):
                Path(output).mkdir(parents=True)
                (Path(output) / "weights").write_text(str(scale))

            with (
                patch("smat.runtime.experiment_module", return_value=object()),
                patch("smat.runtime.checkpoint_adapter", return_value=object()),
                patch("smat.runtime.task_names", return_value=("one",)),
                patch("smat.runtime.checkpoint_path", side_effect=lambda m, c, p: p),
                patch("smat.merge.task_arithmetic", side_effect=write_checkpoint),
                patch("smat.merge_sparse.sparse_merge", side_effect=write_checkpoint),
            ):
                merge_suite(config, root / "result", ["ta", "ties", "dare", "della"])
                for method, scale in [
                    ("ta", 0.3),
                    ("ties", 0.45),
                    ("dare", 0.2),
                    ("della", 0.6),
                ]:
                    self.assertEqual(
                        float(
                            (root / "result/merged" / method / "weights").read_text()
                        ),
                        scale,
                    )
                    receipt = json.loads(
                        (root / "result" / f"{method}-merge.json").read_text()
                    )
                    self.assertEqual(receipt["identity"]["parameters"]["scale"], scale)
                config["merging"]["ties_scale"] = 0.6
                merge_suite(config, root / "result", ["ta", "dare"])
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    merge_suite(config, root / "result", ["ties"])

    def test_ties_matches_full_tied_state_dict_and_writes_canonical_keys(self):
        with tempfile.TemporaryDirectory() as root:
            base = str(Path(root) / "base")
            Path(base).mkdir()
            (Path(base) / "config.json").write_text(
                json.dumps({"tie_word_embeddings": True})
            )
            a = torch.tensor([1.0, 2.0, 3.0, 4.0])
            b = torch.tensor([4.0, -3.0, 2.0, -1.0])
            zeros = {"model.embed_tokens.weight": torch.zeros(4), "w": torch.zeros(4)}
            adapter = MemoryAdapter(
                {
                    base: zeros,
                    "a": {"model.embed_tokens.weight": a, "w": a.flip(0)},
                    "b": {"model.embed_tokens.weight": b, "w": -b},
                }
            )
            sparse_merge(
                adapter,
                base,
                ["a", "b"],
                "out",
                method="ties",
                scale=0.3,
                ties_retention=0.5,
            )
            expanded = torch.stack(
                [torch.cat([a, a, a.flip(0)]), torch.cat([b, b, -b])]
            )
            expected = 0.3 * ties_update(expanded, threshold=0.5)
            torch.testing.assert_close(
                adapter.checkpoints["out"]["model.embed_tokens.weight"], expected[4:8]
            )
            torch.testing.assert_close(adapter.checkpoints["out"]["w"], expected[8:])
            self.assertNotIn("lm_head.weight", adapter.checkpoints["out"])

    def test_merger_parameter_changes_require_new_destination(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            config = dict(
                experiment="clip8_vit",
                name="fixture",
                work_root=str(root),
                base_model="base",
                tasks=["one"],
                merging={"scale": 0.3},
            )
            expert = root / "experts" / "fixture" / "one"
            expert.mkdir(parents=True)
            (expert / "metadata.json").write_text("{}")

            def write_checkpoint(adapter, base, experts, output, *, scale):
                Path(output).mkdir(parents=True)
                (Path(output) / "weights").write_text(str(scale))

            with (
                patch("smat.runtime.experiment_module", return_value=object()),
                patch("smat.runtime.checkpoint_adapter", return_value=object()),
                patch("smat.runtime.task_names", return_value=("one",)),
                patch("smat.runtime.checkpoint_path", side_effect=lambda m, c, p: p),
                patch(
                    "smat.merge.task_arithmetic", side_effect=write_checkpoint
                ) as merge,
            ):
                merge_suite(config, root / "result", ["ta"])
                merge_suite(config, root / "result", ["ta"])
                self.assertEqual(merge.call_count, 1)
                config["merging"]["scale"] = 0.2
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    merge_suite(config, root / "result", ["ta"])


if __name__ == "__main__":
    unittest.main()
