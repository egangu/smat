"""Development selection first; frozen five-seed test only in --phase test."""
import argparse
import json
from pathlib import Path
import torch
from demo_utils import TinyMLP, load_data, SEEDS
from demo_experiment import run, SETTINGS

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='examples/.cache')
parser.add_argument('--device', default='cpu')
parser.add_argument('--phase', choices=['dev', 'test'], default='dev')
args = parser.parse_args()
torch.set_num_threads(2)
device = torch.device(args.device)
base = TinyMLP().to(device)
base.load_state_dict(torch.load('examples/assets/shared_base.pt', weights_only=True, map_location=device))
data, _ = load_data(args.data_root)
results = []
import platform
metadata = {'torch': torch.__version__, 'python': platform.python_version(), 'device': str(device), 'device_name': torch.cuda.get_device_name() if device.type == 'cuda' else platform.processor(), 'threads': torch.get_num_threads(), 'steps': 300, 'lr': 0.001, 'batch_size': 128}
if args.phase == 'dev':
    for lr in (1e-4, 3e-4, 1e-3):
        paper_settings = {'backend': 'eager', 'scale': {'alpha_min': 0.1}, 'mask': {'probability': 0.5}, 'perturb': {'rms': 0.001}, 'smat': {'interval': 4}}
        result = run(base, data, device, seed=0, lr=lr, settings=paper_settings)
        result['lr'] = lr
        results.append(result)
        print(json.dumps(result), flush=True)
else:
    for seed in SEEDS:
        result = run(base, data, device, seed=seed, split='test')
        results.append(result)
        print(json.dumps(result), flush=True)
Path(f'examples/results/{args.phase}-{args.device}.json').write_text(json.dumps({'settings': paper_settings if args.phase == 'dev' else SETTINGS, 'metadata': metadata, 'results': results}, indent=2))
