"""Data and presentation helpers; training/merging live in the notebook."""
from pathlib import Path
import gzip
import math
import hashlib
import urllib.request
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TASKS = ('Right', 'Left')
ANGLES = {'Right': -30, 'Left': 30}
SPLIT_SEED = 20260929
BASE_SEED = 1729
SEEDS = (0, 1, 2, 3, 4)  # Declared before development experiments.

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.heads = nn.ModuleDict({task: nn.Linear(16, 10) for task in TASKS})

    def forward(self, images, task):
        return self.heads[task](self.backbone(images))

# Official dataset mirrors and published torchvision MD5 checksums.
SOURCES = {
    'MNIST': ('https://ossci-datasets.s3.amazonaws.com/mnist/', (
        'f68b3c2dcbeaaa9fbdd348bbdeb94873', 'd53e105ee54ea40749a09fcbcd1e9432',
        '9fb629c4189551a2d022fa330f9573f3', 'ec29112dd5afa0611ce80d1b7f02629c')),

}
FILES = ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

def raw_data(root, task):
    folder = Path(root) / task
    folder.mkdir(parents=True, exist_ok=True)
    url, checksums = SOURCES[task]
    arrays = []
    for name, checksum in zip(FILES, checksums):
        path = folder / name
        if not path.exists():
            print(f'Downloading {task}/{name}', flush=True)
            urllib.request.urlretrieve(url + name, path)
        if hashlib.md5(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Checksum mismatch: {path}; remove and download again')
        contents = gzip.decompress(path.read_bytes())
        offset = 16 if 'images' in name else 8
        array = np.frombuffer(contents, dtype=np.uint8, offset=offset).copy()
        arrays.append(torch.from_numpy(array.reshape(-1, 28, 28) if offset == 16 else array))
    return arrays

def split_indices(labels, sizes, seed=SPLIT_SEED):
    """Disjoint, class-balanced subsets from a fixed per-class permutation."""
    generator = torch.Generator().manual_seed(seed)
    buckets = [[] for _ in sizes]
    for label in range(10):
        ids = torch.where(labels == label)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        start = 0
        for bucket, size in zip(buckets, sizes):
            count = size // 10
            bucket.append(ids[start:start + count])
            start += count
    return [torch.cat(bucket) for bucket in buckets]

def rotate_images(images, degrees):
    """Fixed domain shift: bilinear rotation, zero padding, no random augmentation."""
    angle = math.radians(degrees)
    matrix = images.new_tensor([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle), math.cos(angle), 0]])
    batches = []
    for start in range(0, len(images), 512):
        batch = images[start:start+512, None]
        grid = F.affine_grid(matrix[None].expand(len(batch), -1, -1), batch.shape, align_corners=False)
        batches.append(F.grid_sample(batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[:, 0])
    return torch.cat(batches)


def load_data(root, full_test=False):
    """Disjoint raw-image identities across pretraining, both experts, and dev."""
    train_x, train_y, test_x, test_y = raw_data(root, 'MNIST')
    ids = split_indices(train_y, (2000, 5000, 5000, 1000, 1000))
    test_ids = split_indices(test_y, (2000, 2000))
    data, manifest = {}, {}
    for index, task in enumerate(TASKS):
        selected = {'base': ids[0], 'train': ids[1+index], 'dev': ids[3+index],
                    'test': torch.arange(len(test_y)) if full_test else test_ids[index]}
        manifest[task] = {key: value.tolist() for key, value in selected.items()}
        data[task] = {}
        for key, indices in selected.items():
            x, y = (test_x, test_y) if key == 'test' else (train_x, train_y)
            images = x[indices].float().div(255)
            if key != 'base':
                images = rotate_images(images, ANGLES[task])
            data[task][key] = (images, y[indices].long())
    return data, manifest

@torch.no_grad()
def accuracy(model, task, split, device):
    model.eval()
    images, labels = split
    correct = 0
    for start in range(0, len(labels), 512):
        predictions = model(images[start:start+512].to(device), task).argmax(1).cpu()
        correct += (predictions == labels[start:start+512]).sum().item()
    return 100 * correct / len(labels)

def score(models, data, split, device):
    values = {task: accuracy(models[task], task, data[task][split], device) for task in TASKS}
    return {**values, 'Mean': sum(values.values()) / len(TASKS)}

def show_examples(data):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(9, 3))
    for row, task in enumerate(TASKS):
        x, y = data[task]['dev']
        for label, ax in enumerate(axes[row]):
            index = torch.where(y == label)[0][0]
            ax.imshow(x[index], cmap='gray'); ax.set_title(f'{task}\nlabel {label}', fontsize=9); ax.axis('off')
    fig.tight_layout()
    return fig

def show_predictions(ft, smat, data, device):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(10, 3.5))
    with torch.no_grad():
        for row, task in enumerate(TASKS):
            x, y = data[task]['test']
            for label, ax in enumerate(axes[row]):
                index = torch.where(y == label)[0][0]  # Fixed before seeing predictions.
                image = x[index:index+1].to(device)
                a, b = ft(image, task).argmax().item(), smat(image, task).argmax().item()
                ax.imshow(image[0].cpu(), cmap='gray'); ax.set_title(f'{task}: {label}\nFT {a} / SMAT {b}', fontsize=9); ax.axis('off')
    fig.tight_layout()
    return fig
