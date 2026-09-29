"""Executable experiment, also used to construct the visible notebook cells."""
import copy
import time
import torch
from torch.nn import functional as F
from smat.train.updates import FTStepper, SMATStepper
from demo_utils import TinyMLP, TASKS, score

SETTINGS = {'backend': 'eager', 'scale': {'alpha_min': 0.5}, 'mask': {'probability': 0.1}, 'perturb': {'rms': 0.005}, 'smat': {'interval': 4}}
STEPS = 300
LR = 1e-3
BATCH_SIZE = 128

def train_expert(base, task, method, data, device, seed=0, steps=STEPS, lr=LR, settings=None):
    model = copy.deepcopy(base).to(device)
    model.heads.requires_grad_(False)
    model.train()
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr)
    parameters = list(model.named_parameters())
    if method == 'ft':
        stepper = FTStepper(parameters, optimizer)
    else:
        stepper = SMATStepper(parameters, optimizer, settings or SETTINGS, seed=seed,
                              block_linear_names={'backbone.1.weight', 'backbone.3.weight'})
    # Data RNG is local and independent of all SMAT random streams.
    generator = torch.Generator().manual_seed(seed + 100)
    images, labels = (x.to(device) for x in data[task]['train'])
    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(len(labels), (BATCH_SIZE,), generator=generator).to(device)
        stepper.step(lambda: F.cross_entropy(model(images[indices], task), labels[indices]))
    if device.type == 'cuda': torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    for name, value in base.heads.state_dict().items():
        assert torch.equal(value.cpu(), model.heads.state_dict()[name].cpu())
    return model, elapsed

def merge(base, experts, weight=0.5):
    """Fixed task heads; interpolate ONLY the two compatible backbones."""
    model = copy.deepcopy(base)
    left = experts[TASKS[0]].backbone.state_dict()
    right = experts[TASKS[1]].backbone.state_dict()
    model.backbone.load_state_dict({name: weight * left[name] + (1-weight) * right[name] for name in left})
    return model

def run(base, data, device, seed=0, split='dev', steps=STEPS, lr=LR, settings=None):
    rows = {'Shared base': score(dict.fromkeys(TASKS, base), data, split, device)}
    times = {}
    for method in ('ft', 'smat'):
        experts = {}
        times[method] = 0
        for task in TASKS:
            experts[task], elapsed = train_expert(base, task, method, data, device, seed, steps, lr, settings)
            times[method] += elapsed
        rows[f'{method.upper()} separate experts'] = score(experts, data, split, device)
        merged = merge(base, experts)
        rows[f'{method.upper()} merged'] = score(dict.fromkeys(TASKS, merged), data, split, device)
    return {'seed': seed, 'split': split, 'rows': rows, 'train_seconds': times}
