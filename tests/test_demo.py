"""Common initialization, frozen heads, external SMAT and merging contracts."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import demo_experiment as demo


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Linear(192, 192), nn.Tanh())
        self.norm = nn.LayerNorm(192)


class DemoTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)
        self.base = demo.TwoTaskViT(SmallEncoder()).eval()
        self.data = {task: {"train": (torch.randn(20, 2, 192), torch.arange(20) % 10)} for task in demo.TASKS}

    def test_head_calibration_leaves_hf_backbone_unchanged(self):
        initial = deepcopy(self.base.backbone.state_dict())
        demo.fit_heads(self.base, self.data)
        for name, value in initial.items():
            torch.testing.assert_close(value, self.base.backbone.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in self.base.backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.base.heads.parameters()))

    def test_disabled_smat_equals_ft_and_preserves_base_and_heads(self):
        self.base.heads.requires_grad_(False)
        initial = deepcopy(self.base.state_dict())
        settings = deepcopy(demo.SMAT_SETTINGS)
        settings["smat"]["components"] = []
        ft = demo.train_expert(self.base, self.data, "cifar10", "FT")
        with patch.object(demo, "SMAT_SETTINGS", settings):
            smat = demo.train_expert(self.base, self.data, "cifar10", "SMAT")
        for name, value in initial.items():
            torch.testing.assert_close(self.base.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(ft.state_dict()[name], smat.state_dict()[name], rtol=1e-5, atol=1e-6)
            if name.startswith("heads."):
                torch.testing.assert_close(ft.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(demo.SMATStepper.__module__, "smat.train.updates")

    def test_merging_math_and_task_heads(self):
        experts = [deepcopy(self.base), deepcopy(self.base)]
        with torch.no_grad():
            for index, expert in enumerate(experts):
                for p in expert.backbone.parameters():
                    p.add_(index + 1)
                for p in expert.heads.parameters():
                    p.add_(100)
        for coefficient in (0, .5, .75):
            merged = demo.merge_experts(self.base, experts, coefficient)
            for name, initial in self.base.state_dict().items():
                expected = initial if name.startswith("heads.") else initial + 3 * coefficient
                torch.testing.assert_close(merged.state_dict()[name], expected)

    def test_unknown_training_method_is_rejected(self):
        with self.assertRaises(ValueError):
            demo.train_expert(self.base, self.data, "cifar10", "other")

    def test_notebook_core_matches_executable_source(self):
        root = Path(__file__).resolve().parents[1]
        source = ast.parse((root / "examples/demo_experiment.py").read_text())
        wanted = {n.name: ast.dump(n, include_attributes=False) for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != "fit_heads"}
        notebook = json.loads((root / "examples/smat_image_demo.ipynb").read_text())
        found = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
                    found[node.name] = ast.dump(node, include_attributes=False)
        self.assertEqual(found, wanted)


if __name__ == "__main__":
    unittest.main()
