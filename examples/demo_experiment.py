"""Executable experiment, also used to construct the visible notebook cells."""
import copy
import time
import torch
from torch.nn import functional as F
from torch.func import functional_call
from demo_utils import TinyMLP, TASKS, score

SETTINGS = {'backend': 'eager', 'scale': {'alpha_min': 0.1}, 'mask': {'probability': 0.8}, 'perturb': {'rms': 0.01}, 'smat': {'interval': 4}}
STEPS = 1200
LR = 1e-3
BATCH_SIZE = 128

def smat_logits(model, anchor, images, task, generators, settings):
    """Differentiable simulated weights; original Parameters are never overwritten."""
    scale_rng, mask_rng, noise_rng = generators
    components = settings.get('smat', {}).get('components', ('scale', 'mask', 'perturb'))
    alpha_min = settings['scale']['alpha_min']
    alpha = alpha_min + (1-alpha_min) * torch.rand((), generator=scale_rng).item() if 'scale' in components else 1.0
    probability = settings['mask']['probability'] if 'mask' in components else 0.0
    rms = settings['perturb']['rms'] if 'perturb' in components else 0.0
    simulated = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue  # Frozen task heads are not transformed.
        if probability and name in {'backbone.1.weight', 'backbone.3.weight'}:
            mask = torch.empty_like(parameter, dtype=torch.bool).bernoulli_(1-probability, generator=mask_rng)
            value = anchor[name] + (parameter-anchor[name]) * mask * (alpha/(1-probability))
        else:
            value = torch.lerp(anchor[name], parameter, alpha)
        if rms:
            bound = (3.0 ** 0.5) * rms
            value = value + torch.empty_like(parameter).uniform_(-bound, bound, generator=noise_rng)
        simulated[name] = value
    # Autograd differentiates through Scale and Mask to the expert Parameters.
    return functional_call(model, simulated, (images, task))


def train_expert(base, task, method, data, device, seed=0, steps=STEPS, lr=LR, settings=None):
    model = copy.deepcopy(base).to(device)
    model.heads.requires_grad_(False)
    model.train()
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr)
    settings = settings or SETTINGS
    anchor = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    generators = (torch.Generator().manual_seed(seed+1),
                  torch.Generator(device=device).manual_seed(seed+3),
                  torch.Generator(device=device).manual_seed(seed))
    data_rng = torch.Generator().manual_seed(seed+100)
    images, labels = (x.to(device) for x in data[task]['train'])
    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(len(labels), (BATCH_SIZE,), generator=data_rng).to(device)
        optimizer.zero_grad(set_to_none=True)
        active = method == 'smat' and (step+1) % settings['smat']['interval'] == 0
        if active and settings['smat'].get('components', ('scale', 'mask', 'perturb')):
            logits = smat_logits(model, anchor, images[indices], task, generators, settings)
        else:
            logits = model(images[indices], task)
        F.cross_entropy(logits, labels[indices]).backward()
        optimizer.step()
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

def task_arithmetic(base, experts, coefficient=1.0):
    """Add the sum of task vectors to the shared base; heads remain fixed."""
    model = copy.deepcopy(base)
    origin = base.backbone.state_dict()
    states = [experts[task].backbone.state_dict() for task in TASKS]
    model.backbone.load_state_dict({
        name: origin[name] + coefficient * sum(state[name]-origin[name] for state in states)
        for name in origin
    })
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
        rows[f'{method.upper()} AVG'] = score(dict.fromkeys(TASKS, merged), data, split, device)
        arithmetic = task_arithmetic(base, experts)
        rows[f'{method.upper()} TA'] = score(dict.fromkeys(TASKS, arithmetic), data, split, device)
    return {'seed': seed, 'split': split, 'rows': rows, 'train_seconds': times}
