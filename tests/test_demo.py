"""Protect the demo's common-base, frozen-head and data-isolation contracts."""
import copy
import json
import hashlib
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from demo_utils import TASKS, TinyMLP, split_indices
from demo_experiment import SETTINGS, merge, train_expert, task_arithmetic, smat_logits

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

    def test_released_base_and_rotated_domains_do_not_leak_images(self):
        assets = Path(__file__).resolve().parents[1] / 'examples' / 'assets'
        metadata = json.loads((assets / 'shared_base.json').read_text())
        checkpoint = assets / 'shared_base.pt'
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), metadata['sha256'])
        splits = metadata['splits']
        self.assertEqual(splits[TASKS[0]]['base'], splits[TASKS[1]]['base'])
        buckets = [splits[TASKS[0]]['base']]
        buckets += [splits[task][split] for task in TASKS for split in ('train', 'dev')]
        self.assertEqual(sum(map(len, buckets)), 14000)
        self.assertEqual(len(set(index for bucket in buckets for index in bucket)), 14000)
        self.assertTrue(set(splits[TASKS[0]]['test']).isdisjoint(splits[TASKS[1]]['test']))
        state = torch.load(checkpoint, weights_only=True)
        for name in ('weight', 'bias'):
            torch.testing.assert_close(state[f'heads.{TASKS[0]}.{name}'],
                                       state[f'heads.{TASKS[1]}.{name}'], rtol=0, atol=0)

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

    def test_task_arithmetic_half_equals_average_and_zero_equals_base(self):
        experts = {task: copy.deepcopy(self.base) for task in TASKS}
        with torch.no_grad():
            for model in experts.values():
                for parameter in model.backbone.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.01)
        for coefficient, expected in ((0.5, merge(self.base, experts)), (0.0, self.base)):
            result = task_arithmetic(self.base, experts, coefficient)
            for name, value in result.state_dict().items():
                torch.testing.assert_close(value, expected.state_dict()[name])

    def test_inline_smat_matches_released_eager_stepper(self):
        from smat.train.updates import SMATStepper
        from torch.nn import functional as F
        reference = copy.deepcopy(self.base)
        reference.heads.requires_grad_(False)
        model = copy.deepcopy(reference)
        anchor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        generators = (torch.Generator().manual_seed(1), torch.Generator().manual_seed(3), torch.Generator().manual_seed(0))
        optimizer = torch.optim.Adam(model.backbone.parameters(), lr=0.001)
        ref_optimizer = torch.optim.Adam(reference.backbone.parameters(), lr=0.001)
        stepper = SMATStepper(list(reference.named_parameters()), ref_optimizer, SETTINGS, 0,
                              {'backbone.1.weight', 'backbone.3.weight'})
        images, labels = self.data[TASKS[0]]['train']
        for step in range(12):
            ref_loss = stepper.step(lambda: F.cross_entropy(reference(images, TASKS[0]), labels))
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            optimizer.zero_grad(set_to_none=True)
            logits = smat_logits(model, anchor, images, TASKS[0], generators, SETTINGS) if (step+1)%4 == 0 else model(images, TASKS[0])
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            optimizer.step()
            torch.testing.assert_close(loss.detach(), ref_loss, atol=2e-6, rtol=1e-5)
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=1e-4)

if __name__ == '__main__':
    unittest.main()
