"""Small, explicit development-only SMAT grid after the common LR selection."""
import copy
import json
from pathlib import Path
import torch
from demo_utils import TASKS, TinyMLP, load_data, score
from demo_experiment import SETTINGS, LR, train_expert, merge

torch.set_num_threads(2)
device = torch.device('cuda')
base = TinyMLP().to(device)
base.load_state_dict(torch.load('examples/assets/shared_base.pt', map_location=device, weights_only=True))
data, _ = load_data('examples/.cache')
results = []
# Candidate list fixed before executing this script; no test feedback.
for alpha, mask, rms in [(0.1, 0.5, 0.001), (0.1, 0.1, 0.001), (0.5, 0.3, 0.001), (0.1, 0.5, 0.005), (0.5, 0.1, 0.005)]:
    settings = copy.deepcopy(SETTINGS)
    settings.update(scale={'alpha_min': alpha}, mask={'probability': mask}, perturb={'rms': rms})
    experts = {task: train_expert(base, task, 'smat', data, device, seed=0, settings=settings)[0] for task in TASKS}
    merged = merge(base, experts)
    result = {'settings': settings, 'lr': LR, 'seed': 0, 'split': 'dev', 'separate': score(experts, data, 'dev', device), 'merged': score(dict.fromkeys(TASKS, merged), data, 'dev', device)}
    results.append(result)
    print(json.dumps(result), flush=True)
Path('examples/results/dev-settings.json').write_text(json.dumps(results, indent=2))
