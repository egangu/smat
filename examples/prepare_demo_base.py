"""Pretrain the tiny base on upright MNIST only; target domains are unseen."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from demo_utils import BASE_SEED, SPLIT_SEED, TASKS, ANGLES, TinyMLP, load_data


def prepare(root, output, device='cpu'):
    torch.set_num_threads(2)
    torch.manual_seed(BASE_SEED)
    data, splits = load_data(root)
    model = TinyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(BASE_SEED)
    images, labels = (value.to(device) for value in data[TASKS[0]]['base'])
    for step in range(400):
        indices = torch.randint(len(labels), (128,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(images[indices], TASKS[0]), labels[indices]).backward()
        optimizer.step()
    # Both domains have the same digit labels and the same fixed classifier.
    model.heads[TASKS[1]].load_state_dict(model.heads[TASKS[0]].state_dict())
    model.heads.requires_grad_(False)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output / 'shared_base.pt'
    torch.save(model.cpu().state_dict(), path)
    metadata = {
        'pretraining': 'upright MNIST only', 'base_seed': BASE_SEED,
        'split_seed': SPLIT_SEED, 'angles': ANGLES, 'steps': 400,
        'batch_size': 128, 'backbone_widths': [32, 16],
        'parameters': sum(p.numel() for p in model.parameters()),
        'lr': 0.001, 'optimizer': 'Adam', 'torch': torch.__version__,
        'device': str(device), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'splits': splits,
    }
    (output / 'shared_base.json').write_text(json.dumps(metadata) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'splits'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='examples/.cache')
    parser.add_argument('--output', default='examples/assets')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    prepare(args.data_root, args.output, args.device)
