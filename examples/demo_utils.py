"""Downloads and plotting only. Training and merging are visible in the notebook."""
import os
from pathlib import Path

import numpy as np
import torch
import timm
from torch.nn import functional as F
from huggingface_hub import hf_hub_download

MODEL_ID = "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k"
MODEL_REVISION = "7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106"
DATA_ID = "yanggangu/SMAT-Tiny-Demo"
DATA_REVISION = "4bce144c014354a937e84177fdb13810f298c56e"
TASKS = ("cifar10", "svhn")


def load_encoder():
    weights = os.environ.get("SMAT_DEMO_WEIGHTS") or hf_hub_download(
        MODEL_ID, "model.safetensors", revision=MODEL_REVISION
    )
    return timm.create_model(
        MODEL_ID.split("/")[1], pretrained=True, num_classes=0, img_size=64,
        pretrained_cfg_overlay={"file": weights, "custom_load": False},
    )


def data_path(split):
    local = os.environ.get("SMAT_DEMO_DATA_DIR")
    return Path(local) / f"{split}.npz" if local else hf_hub_download(
        DATA_ID, f"{split}.npz", repo_type="dataset", revision=DATA_REVISION
    )


@torch.no_grad()
def load_data(encoder, device):
    """Cache only the frozen patch embedding; all 12 Transformer blocks train."""
    data = {task: {} for task in TASKS}
    mean = torch.tensor(encoder.pretrained_cfg["mean"], device=device)[None, :, None, None]
    std = torch.tensor(encoder.pretrained_cfg["std"], device=device)[None, :, None, None]
    for split in ("train", "test"):
        with np.load(data_path(split), allow_pickle=False) as arrays:
            for task in TASKS:
                images = torch.from_numpy(arrays[f"{task}_images"]).permute(0, 3, 1, 2).float() / 255
                tokens = []
                for batch in images.split(128):
                    x = F.interpolate(batch.to(device), (64, 64), mode="bicubic", align_corners=False, antialias=True)
                    x = (x - mean) / std
                    z = encoder.norm_pre(encoder.patch_drop(encoder._pos_embed(encoder.patch_embed(x))))
                    tokens.append(z)
                y = torch.from_numpy(arrays[f"{task}_labels"]).to(device)
                data[task][split] = (torch.cat(tokens), y)
    return data


def show_examples():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(8, 3.7))
    labels = (("airplane", "automobile", "bird", "cat"), ("0", "1", "2", "3"))
    with np.load(data_path("train"), allow_pickle=False) as arrays:
        for row, task in enumerate(TASKS):
            x, y = arrays[f"{task}_images"], arrays[f"{task}_labels"]
            for c, ax in enumerate(axes[row]):
                ax.imshow(x[np.flatnonzero(y == c)[0]])
                ax.set_title(labels[row][c], fontsize=10); ax.axis("off")
    fig.text(.02, .75, "CIFAR-10\nobjects", ha="left", va="center", fontsize=11)
    fig.text(.02, .30, "SVHN\ndigits", ha="left", va="center", fontsize=11)
    fig.tight_layout(rect=(.13, 0, 1, 1))
    return fig


def show_results(rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, method, color in [(-.18, "FT", "#e19b43"), (.18, "SMAT", "#2779bb")]:
        values = [rows[f"{method} {merger}"]["Mean"] for merger in ("AVG", "TA")]
        bars = ax.bar(np.arange(2) + offset, values, width=.33, label=method, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    baseline = rows["Base"]["Mean"]
    ax.axhline(baseline, color="#6b7280", linestyle="--", label=f"Base: {baseline:.2f}%")
    for index, merger in enumerate(("AVG", "TA")):
        gain = rows[f"SMAT {merger}"]["Mean"] - rows[f"FT {merger}"]["Mean"]
        ax.text(index, 88, f"{gain:+.2f} points", ha="center", fontsize=12, color="#226a9f")
    ax.set(xticks=[0, 1], xticklabels=["AVG", "TA (coefficient 0.75)"], ylim=(0, 100),
           ylabel="Mean test accuracy (%)", title="Two tasks · one merged encoder")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.15), ncol=3, frameon=False)
    fig.tight_layout()
    return fig
