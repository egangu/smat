# A CPU-sized SMAT demo

[Notebook](smat_image_demo.ipynb) · [Run in Colab](https://colab.research.google.com/github/egangu/smat/blob/main/examples/smat_image_demo.ipynb)

Start with a **25,988-parameter MLP pretrained on upright MNIST**. Train one
expert for left-tilted digits and one for right-tilted digits, then merge their
backbones with AVG or Task Arithmetic (TA). Compare ordinary Adam fine-tuning
with Adam + SMAT. Both methods learn useful changes: every merged model in the
five-seed CPU/CUDA checks beats the starting point **on each domain**.

This is a controlled domain-shift demonstration, not a reproduction of the
paper's CLIP/LLM experiments. All four experts actually train during Run All.

## Run

Python 3.10+. For a CPU-only environment:

```bash
python -m pip install 'torch>=2.9' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/demo.txt
jupyter lab examples/smat_image_demo.ipynb
```

Choose **Restart Kernel and Run All**. Defaults: CPU, FP32, two threads, seed 0,
1,200 steps per expert, batch 128. CUDA is optional: select `DEVICE="cuda"`.
The notebook contains fixed settings and no dev-set parameter search.

SMAT is shown with `torch.func.functional_call`. Scale/Mask gradients reach
the original expert parameters; temporary simulated weights do not overwrite
optimizer parameters. The implementation is checked against the released eager
stepper. The notebook needs no SMAT installation, Transformers, torchvision,
datasets, or Triton. It does not load pretrained expert results.

The first run downloads complete compressed MNIST files (about **12 MB**).
Later runs reuse the cache. The pretrained base is approximately **105 KB**.

## Fixed recipe

- Backbone: 784 → 32 → 16, ReLU; fixed 10-class digit classifier. No dropout or
  BatchNorm. Both domains use identical classifier weights, so inference does
  not need an oracle to choose between different label spaces.
- Base: **2,000 upright MNIST images only**, 400 Adam updates, LR 0.001,
  batch 128, seed 1729. The base never trains on the tilted domains. Its
  upright accuracy on the selected right-domain test identities is **88.90%**;
  the domain shift reduces its mean tilted accuracy to **54.675%**.
- Domains: fixed −30° and +30° image rotations, bilinear sampling, zero padding,
  `align_corners=False`. This is the task definition, not random augmentation.
- Each expert: 5,000 training images and a separate 1,000-image dev subset.
  Pretraining, both training subsets, and both dev subsets have **disjoint
  original-image identities**, all from the official training split.
- Test: balanced 2,000 images/domain from the official test split, with disjoint
  identities between the two default subsets. `FULL_TEST=True` instead evaluates
  both orientations of all 10,000 test images. Split seed: 20260929.
- FT and SMAT share the exact base, minibatches, Adam LR 0.001, and 1,200 updates.
  SMAT draws transformations from separate RNG streams. Every fourth update
  uses Scale (minimum 0.1), Mask (probability 0.8), and uniform Perturb (RMS 0.01).
  Only the two backbone weight matrices are masked; heads stay frozen.
- AVG: equal average of the two backbones. TA:
  `base + 0.75 * (delta_left + delta_right)`. The coefficient is fixed and
  identical for FT/SMAT; 0.5 would be equivalent to AVG for two experts.

The checkpoint hash, pretraining recipe, and exact split indices are in
[`assets/shared_base.json`](assets/shared_base.json). Regenerate it from the
repository root:

```bash
python examples/prepare_demo_base.py --device cpu
# The distributed checkpoint was prepared on CUDA with PyTorch 2.11.0:
python examples/prepare_demo_base.py --device cuda
```

The published recipe reproduced the frozen development checkpoint exactly on
its original runtime. Floating-point results and serialization hashes can vary
across platforms; notebook downloads verify the distributed artifact's hash.

## Held-out results

Settings were frozen after development experiments, before evaluating this
recipe on the official test subsets. Seeds **0–4** were declared in advance;
the notebook uses **0**, not a seed selected for its test result. Variation
below covers expert-training RNGs, not multiple base checkpoints or splits.

Mean test accuracy ± sample standard deviation across five seeds:

| Device | Merger | FT | SMAT | Paired gain (points) |
|---|---|---:|---:|---:|
| CPU | AVG | 69.520 ± 0.235 | 73.215 ± 0.336 | +3.695 ± 0.472 |
| CPU | TA | 61.065 ± 0.816 | 68.110 ± 0.660 | +7.045 ± 0.570 |
| CUDA | AVG | 69.520 ± 0.235 | 73.205 ± 0.443 | +3.685 ± 0.545 |
| CUDA | TA | 61.065 ± 0.816 | 67.775 ± 1.348 | +6.710 ± 0.964 |

The shared base scores **54.675%**. All FT/SMAT × AVG/TA results exceed it on
**both domains**, for every tested seed and device. Mean gains exceed 3 points;
individual runs need not (CUDA AVG seed 4: +2.975). Full per-domain scores,
paired gains and environment details are in [`results/test-cpu.json`](results/test-cpu.json)
and [`results/test-cuda.json`](results/test-cuda.json).

Separate experts retain two backbones and are a reference, not a merged model.
The experiment demonstrates a benefit under matched training budgets and fixed
mergers; it does not establish superiority over every tuned FT learning rate,
early-stopping rule, or merger. Longer training improves individual experts
while making their updates harder to merge. Gains may differ on other tasks.
Timings are measured directly and do not test the paper's **<2% overhead** claim.

```bash
python examples/validate_demo.py --device cpu
python examples/validate_demo.py --device cuda
# Reference-parity checks additionally import the repo's SMAT source:
python -m unittest discover -s tests -p 'test_demo.py'
python -m unittest discover -s tests -p 'test_updates.py'
```
