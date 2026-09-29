"""Run the frozen notebook recipe for five predeclared seeds, outside Jupyter."""
import argparse
import json
import platform
import statistics
from pathlib import Path
import torch
from demo_utils import TinyMLP, load_data, SEEDS
from demo_experiment import run, SETTINGS, STEPS, LR, BATCH_SIZE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='examples/.cache')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--phase', choices=['dev', 'test'], default='test')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    base = TinyMLP().to(device)
    base.load_state_dict(torch.load('examples/assets/shared_base.pt', weights_only=True, map_location=device))
    data, _ = load_data(args.data_root)
    results = []
    for seed in SEEDS:
        result = run(base, data, device, seed=seed, split=args.phase)
        results.append(result)
        print(json.dumps(result), flush=True)
    def moments(values):
        return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
    summary = {name: moments([r['rows'][name]['Mean'] for r in results]) for name in results[0]['rows']}
    gains = {merger: moments([r['rows'][f'SMAT {merger}']['Mean']-r['rows'][f'FT {merger}']['Mean'] for r in results]) for merger in ('AVG', 'TA')}
    report = {'settings': SETTINGS, 'metadata': {'torch': torch.__version__, 'python': platform.python_version(),
        'device': str(device), 'device_name': torch.cuda.get_device_name() if device.type == 'cuda' else platform.machine(),
        'threads': 2, 'steps': STEPS, 'lr': LR, 'batch_size': BATCH_SIZE, 'parameters': sum(p.numel() for p in base.parameters())},
        'results': results, 'summary': summary, 'paired_gain_pp': gains}
    Path('examples/results').mkdir(exist_ok=True)
    Path(f'examples/results/{args.phase}-{args.device}.json').write_text(json.dumps(report, indent=2)+'\n')
