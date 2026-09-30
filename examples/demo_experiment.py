"""The small training/merging functions shown in the notebook."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from smat.train.updates import FTStepper, SMATStepper

TASKS = ("cifar10", "svhn")
SMAT_SETTINGS = {
    "backend": "eager",
    "scale": {"alpha_min": 0.1},
    "mask": {"probability": 0.8},
    "perturb": {"rms": 0.01},
    "smat": {"interval": 4},
}


class TwoTaskViT(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = nn.Sequential(deepcopy(backbone.blocks), deepcopy(backbone.norm))
        self.heads = nn.ModuleDict({task: nn.Linear(192, 10) for task in TASKS})

    def features(self, tokens):
        return self.backbone(tokens)[:, 0]

    def forward(self, tokens, task):
        return self.heads[task](self.features(tokens))


def fit_heads(base, data):
    """Fit task heads on training data; leave the HF encoder unchanged."""
    base.requires_grad_(False)
    base.heads.requires_grad_(True)
    with torch.no_grad():
        features = {
            task: torch.cat([base.features(x) for x in data[task]["train"][0].split(128)])
            for task in TASKS
        }
    optimizer = torch.optim.Adam(base.heads.parameters(), lr=0.01)
    rng = torch.Generator().manual_seed(1729)
    for step in range(800):
        task = TASKS[step % 2]
        x, y = features[task], data[task]["train"][1]
        ids = torch.randint(len(y), (128,), generator=rng).to(y.device)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(base.heads[task](x[ids]), y[ids]).backward()
        optimizer.step()
    base.requires_grad_(False)
    base.backbone.requires_grad_(True)


def train_expert(base, data, task, method, seed=0):
    """Same initialization, Adam, batches and 600 updates for FT and SMAT."""
    if method not in {"FT", "SMAT"}:
        raise ValueError("method must be FT or SMAT")
    expert = deepcopy(base)
    parameters = list(expert.named_parameters())
    optimizer = torch.optim.Adam([p for _, p in parameters if p.requires_grad], lr=1e-4)
    linear_weights = {n for n, p in parameters if p.requires_grad and p.ndim == 2}
    stepper = (
        FTStepper(parameters, optimizer) if method == "FT" else
        SMATStepper(parameters, optimizer, SMAT_SETTINGS, seed, linear_weights)
    )
    x, y = data[task]["train"]
    rng = torch.Generator().manual_seed(seed + 100)
    for _ in range(600):
        ids = torch.randint(len(y), (32,), generator=rng).to(y.device)
        stepper.step(lambda: F.cross_entropy(expert(x[ids], task), y[ids]))
    return expert.eval()


def merge_experts(base, experts, coefficient):
    """AVG: 0.5. TA: 0.75. Keep both task-specific heads unchanged."""
    merged = deepcopy(base)
    anchor = base.backbone.state_dict()
    states = [expert.backbone.state_dict() for expert in experts]
    merged.backbone.load_state_dict({
        name: value + coefficient * sum(state[name] - value for state in states)
        for name, value in anchor.items()
    })
    return merged.eval()


@torch.no_grad()
def evaluate(model, data):
    """Evaluate one shared model, or a mapping of task to its own expert."""
    scores = {}
    for task in TASKS:
        task_model = model[task] if isinstance(model, dict) else model
        x, y = data[task]["test"]
        predictions = torch.cat([task_model(batch, task).argmax(1) for batch in x.split(128)])
        scores[task] = 100 * (predictions == y).sum().item() / len(y)
    scores["Mean"] = sum(scores.values()) / len(TASKS)
    return scores
