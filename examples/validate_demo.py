"""Run the frozen notebook recipe for five predeclared seeds, outside Jupyter."""
import argparse
import json
import platform
import statistics
from pathlib import Path
import torch
from demo_utils import TinyMLP, load_data, SEEDS, TASKS, raw_data, accuracy
from demo_experiment import run, SETTINGS, STEPS, LR, BATCH_SIZE, TA_COEFFICIENT

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
    data, manifest = load_data(args.data_root)
    results = []
    for seed in SEEDS:
        result = run(base, data, device, seed=seed, split=args.phase)
        results.append(result)
        print(json.dumps(result), flush=True)
    def moments(values):
        return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
    summary = {name: moments([r['rows'][name]['Mean'] for r in results]) for name in results[0]['rows']}
    gains = {merger: moments([r['rows'][f'SMAT {merger}']['Mean']-r['rows'][f'FT {merger}']['Mean'] for r in results]) for merger in ('AVG', 'TA')}
    upright_score = None
    if args.phase == 'test':
        _, _, upright_x, upright_y = raw_data(args.data_root, 'MNIST')
        indices = torch.tensor(manifest[TASKS[0]]['test'])
        upright_score = accuracy(base, TASKS[0], (upright_x[indices].float()/255, upright_y[indices].long()), device)
    above_base = all(r['rows'][f'{method} {merger}'][task] > r['rows']['Shared base'][task]
                     for r in results for method in ('FT', 'SMAT') for merger in ('AVG', 'TA')
                     for task in (*TASKS, 'Mean'))
    report = {'settings': SETTINGS, 'all_merged_above_base_each_task': above_base,
              'upright_base_test_accuracy': upright_score, 'metadata': {'torch': torch.__version__, 'python': platform.python_version(),
        'device': str(device), 'device_name': torch.cuda.get_device_name() if device.type == 'cuda' else platform.machine(),
        'threads': 2, 'steps': STEPS, 'lr': LR, 'batch_size': BATCH_SIZE, 'ta_coefficient': TA_COEFFICIENT, 'parameters': sum(p.numel() for p in base.parameters())},
        'results': results, 'summary': summary, 'paired_gain_pp': gains}
    Path('examples/results').mkdir(exist_ok=True)
    Path(f'examples/results/{args.phase}-{args.device}.json').write_text(json.dumps(report, indent=2)+'\n')

    assert above_base, 'A merged result failed the shared-base acceptance criterion'
