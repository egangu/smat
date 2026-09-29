"""Regenerate the shared start (CPU, fixed seed, no dev/test optimization)."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from demo_utils import BASE_SEED, SPLIT_SEED, TASKS, TinyMLP, load_data

def prepare(root, output):
    torch.set_num_threads(2)
    torch.manual_seed(BASE_SEED)
    data, splits = load_data(root)
    model = TinyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(BASE_SEED)
    for step in range(400):
        task = TASKS[step % 2]
        images, labels = data[task]['base']
        indices = torch.randint(len(labels), (128,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(images[indices], task), labels[indices]).backward()
        optimizer.step()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output / 'shared_base.pt'
    torch.save(model.state_dict(), path)
    metadata = {'base_seed': BASE_SEED, 'split_seed': SPLIT_SEED, 'steps': 400, 'batch_size': 128,
                'lr': 0.001, 'optimizer': 'Adam', 'torch': torch.__version__, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'splits': splits}
    (output / 'shared_base.json').write_text(json.dumps(metadata))
    print(json.dumps({k:v for k,v in metadata.items() if k != 'splits'}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='examples/.cache')
    parser.add_argument('--output', default='examples/assets')
    args = parser.parse_args()
    prepare(args.data_root, args.output)
