"""Protect the demo's common-base, frozen-head and data-isolation contracts."""
import copy
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from demo_utils import TASKS, TinyMLP, split_indices
from demo_experiment import SETTINGS, merge, train_expert

class DemoTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)
        self.base = TinyMLP()
        self.data = {task: {'train': (torch.rand(40, 28, 28), torch.arange(40) % 10)} for task in TASKS}
        self.device = torch.device('cpu')

    def test_disjoint_balanced_reproducible_splits(self):
        labels = torch.arange(1000) % 10
        first = split_indices(labels, (200, 500, 100))
        second = split_indices(labels, (200, 500, 100))
        self.assertEqual(len(set(torch.cat(first).tolist())), 800)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
            self.assertEqual(torch.bincount(labels[a]).unique().numel(), 1)

    def test_disabled_smat_is_ft_and_heads_and_base_do_not_change(self):
        initial = copy.deepcopy(self.base.state_dict())
        settings = copy.deepcopy(SETTINGS)
        settings['smat']['components'] = []
        ft, _ = train_expert(self.base, TASKS[0], 'ft', self.data, self.device, steps=8)
        smat, _ = train_expert(self.base, TASKS[0], 'smat', self.data, self.device, steps=8, settings=settings)
        for name, value in initial.items():
            torch.testing.assert_close(value, self.base.state_dict()[name], rtol=0, atol=0)
            # CPU BLAS may round differently for separately allocated tensors.
            torch.testing.assert_close(ft.state_dict()[name], smat.state_dict()[name], rtol=1e-5, atol=1e-6)
            if name.startswith('heads.'):
                torch.testing.assert_close(value, ft.state_dict()[name], rtol=0, atol=0)

    def test_merge_only_backbone_and_endpoints(self):
        experts = {task: copy.deepcopy(self.base) for task in TASKS}
        with torch.no_grad():
            for index, model in enumerate(experts.values()):
                for p in model.backbone.parameters(): p.add_(index + 1)
                for p in model.heads.parameters(): p.add_(100)
        merged = merge(self.base, experts)
        for name, value in self.base.state_dict().items():
            expected = value if name.startswith('heads.') else value + 1.5
            torch.testing.assert_close(merged.state_dict()[name], expected)
        for weight, task in ((1, TASKS[0]), (0, TASKS[1])):
            result = merge(self.base, experts, weight)
            for name, value in result.backbone.state_dict().items():
                torch.testing.assert_close(value, experts[task].backbone.state_dict()[name])

if __name__ == '__main__':
    unittest.main()
