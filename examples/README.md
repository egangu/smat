# SMAT on a small image model

[Notebook](smat_image_demo.ipynb) · [Run in Colab](https://colab.research.google.com/github/egangu/smat/blob/main/examples/smat_image_demo.ipynb)

Train MNIST and Fashion-MNIST experts from a shared **25,988-parameter MLP**,
then compare AVG and Task Arithmetic (TA). This is a small **merge-interference
stress test**, not a reproduction of the CLIP/LLM benchmark. A narrow backbone
makes the difference between individual expertise and mergeability visible.

## Run

Use Python 3.10+. For a CPU-only environment, install CPU PyTorch first:

```bash
python -m pip install 'torch>=2.9' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/demo.txt
jupyter lab examples/smat_image_demo.ipynb
```

Choose **Restart Kernel and Run All**. Defaults: CPU, FP32, two threads, seed 0,
1,200 steps per expert, batch 128. All four experts actually train. CUDA is
optional; select `DEVICE="cuda"`. There is no parameter search in the notebook.

The SMAT formula is implemented visibly with `torch.func.functional_call`.
Autograd propagates Scale/Mask to the original expert parameters. Temporary
weights never overwrite the optimizer's parameters. CPU and CUDA checks compare
this implementation against the released eager stepper. The notebook itself
needs no SMAT installation, Transformers, torchvision, datasets, or Triton.

The first run downloads complete compressed MNIST and Fashion-MNIST files
(about 41 MB), even though training uses subsets. Subsequent runs reuse the
cache. The shared checkpoint is about 105 KB; no experts are downloaded.

## Fixed recipe

- MLP backbone: 784 → 32 → 16, ReLU; two 10-class heads. No dropout, BatchNorm
  or augmentation. Inputs scaled to [0, 1]. Task identity is known at evaluation.
- Joint base: 2,000 images/task, 400 alternating Adam steps, LR 0.001,
  batch 128, seed 1729. Both heads train at this stage and then stay frozen.
- Each expert: 5,000 train images/task. Separate dev set: 1,000/task.
  These sets are disjoint subsets of the official training split.
- Test: fixed balanced 2,000 images/task from official test. Set `FULL_TEST=True`
  to use all 10,000/task. Split seed 20260929; exact indices and base SHA-256
  are in `assets/shared_base.json`.
- FT and SMAT share the exact base, batches, Adam LR 0.001, and 1,200 updates.
  SMAT has separate random streams; every fourth update uses simulated weights.
- SMAT: Scale minimum 0.1, Mask probability 0.8, uniform Perturb RMS 0.01.
  Mask applies only to the two backbone weight matrices, not biases or heads.
- AVG averages the two backbones. TA is `base + delta_1 + delta_2`, with fixed
  coefficient **1.0** for both methods. For two experts, TA coefficient 0.5
  equals AVG. Both mergers retain the original task heads.

Regenerate the base with `python examples/prepare_demo_base.py` from the repo
root. Floating-point results and checkpoint file hashes can vary by platform;
the distributed checkpoint has a fixed checked hash.

## Interpretation and validation

The initial wider MLP gave only about +0.2 points; its records are retained in
`results/initial-wider-mlp/`. Design probes were evaluated on dev, outside the
notebook. The final narrower model and settings were frozen before its test
runs. Seeds **0–4** were declared before development; seed 0 was never selected
for its test result. Five-seed variation covers expert-training randomness,
not different dataset splits or pretrained bases.

**The shared base remains stronger than the merged models in this stress test.**
SMAT mitigates merge damage; it does not establish a better overall model than
joint training. Separate-expert rows use two backbones. This controlled,
matched-budget example does not establish superiority over every possible FT
learning rate, early-stopping rule, or tuned merger. It also does not measure
the paper's <2% overhead claim. Timings include first-call initialization but
exclude downloads/evaluation.

The JSON files `results/test-cpu.json` and `results/test-cuda.json` contain
individual seeds, means, sample standard deviations, and paired gains.

```bash
# The notebook and validation runner do not need the full SMAT package.
python examples/validate_demo.py --device cpu
python examples/validate_demo.py --device cuda  # optional
# Reference-parity tests additionally import the repo's SMAT source:
python -m unittest discover -s tests -p 'test_demo.py'
python -m unittest discover -s tests -p 'test_updates.py'
```

Five-seed held-out test accuracy (mean ± sample standard deviation):

| Device | Merger | FT | SMAT | Paired gain (points) |
|---|---|---:|---:|---:|
| CPU | AVG | 75.655 ± 0.555 | 79.420 ± 0.638 | +3.765 ± 1.016 |
| CPU | TA | 57.835 ± 0.925 | 64.310 ± 1.961 | +6.475 ± 2.023 |
| CUDA | AVG | 75.640 ± 0.526 | 79.315 ± 0.195 | +3.675 ± 0.372 |
| CUDA | TA | 57.835 ± 0.925 | 63.760 ± 1.217 | +5.925 ± 0.810 |

The shared base scores **81.400%**. The mean gains exceed 3 points for both
mergers and devices; individual runs are not guaranteed to do so (CPU AVG
seed 1: +2.625 points). These results apply to this fixed stress-test recipe.

Actual notebook acceptance on dgx44 (cached data, including kernel startup):
CPU **25.1 s**, peak RSS **810 MiB**, two CPU threads; CUDA **21.4 s**.
Both completed all 12 code cells without errors and loaded none of `smat`,
`transformers`, `datasets`, `accelerate`, or `torchvision`. Hardware and cache
state affect timing; these are not promises for a laptop or Colab runtime.
Execution records and the executed-code hash are in `results/notebook-*.json`.
