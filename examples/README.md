# A small, CPU-friendly SMAT experiment

Train MNIST and Fashion-MNIST experts from one shared MLP, then merge their
backbones. This is an educational experiment, not a reproduction of the paper's
CLIP/LLM results. Task identity is known: each task retains its own frozen head.

## Run

From the repository root, in a Python 3.10+ environment:

```bash
python -m pip install -r requirements/demo.txt
python -m pip install --no-deps -e .
jupyter lab examples/smat_image_demo.ipynb
```

Select **Restart Kernel and Run All**. The default is CPU, FP32, two CPU threads,
seed 0, 300 steps per expert, batch size 128. All four experts really train.
The notebook imports the released `FTStepper` / `SMATStepper`; it does not copy
or approximate SMAT. Only the eager backend is used. No Transformers, datasets,
Triton or GPU is needed for this example. The first run downloads the complete
compressed official MNIST and Fashion-MNIST files (about 41 MB), even though
training uses small subsets. Subsequent runs reuse the cache.

## What is held fixed

- Shared MLP: 784 → 128 → 64, ReLU, two 10-class linear heads (110,036 parameters). No dropout, BatchNorm or augmentation.
- Joint base: 2,000 examples/task, 400 alternating Adam steps, learning rate
  0.001, batch size 128, seed 1729. Both heads train only at this stage.
- Experts: 5,000 examples/task; disjoint 1,000/task dev set, from official train.
- Test: fixed class-balanced 2,000/task from official test; optional full test.
- Split seed: 20260929. Exact indices and base SHA-256 are in
  `assets/shared_base.json`; checkpoint size is under 0.5 MB.
- FT and SMAT share the exact base, batches, Adam settings and step count.
  SMAT has independent RNG streams. Every fourth update simulates a merged
  state; the other three are ordinary FT. Only the two backbone weight matrices
  are masked; heads remain frozen. Merging averages all backbone parameters.
- Primary comparison: fixed 50/50 merged accuracy. Separate-expert rows use two
  backbones. Smaller merge loss is not sufficient if absolute accuracy declines.

Regenerate the base (run from the repository root):

```bash
python examples/prepare_demo_base.py
```

This recreates the splits and training recipe. Floating-point results and file
hashes can vary with PyTorch/platform; the distributed checkpoint's hash is
checked before loading. Never pretrain on dev or test examples.

## Validation and interpretation

Development and five-seed test records are in `results/`. Training seeds
`[0, 1, 2, 3, 4]` were declared before development; seed 0 is the default, not
chosen for its test result. Hyperparameters are selected on dev only, and all
five seeds use the same frozen shared base and dataset split. Variation across
seeds therefore measures expert-training randomness, not dataset/base variation.

The notebook displays the shared base alongside both separate and merged
experts. Improvements can be small, can vary by seed, and may not exceed the
jointly pretrained base. This toy eager implementation does not establish the
paper's training-overhead claim. Timings exclude downloads and evaluation.

Frozen settings: Adam learning rate **0.001**, SMAT scale minimum **0.5**, mask
probability **0.1**, perturbation RMS **0.005**, interval **4**. The three-LR
comparison chose the strongest FT merged dev score; the five-candidate SMAT
comparison then chose these settings at the same LR. Both searches used seed 0
and dev only. They are validation scripts, **not notebook steps**.

Five-seed test results (mean ± sample standard deviation, percentage points):

| Device | FT merged | SMAT merged | Paired gain |
|---|---:|---:|---:|
| CPU, 2 threads | 85.650 ± 0.291 | 85.880 ± 0.355 | +0.230 ± 0.212 |
| CUDA | 85.615 ± 0.266 | 85.790 ± 0.322 | +0.175 ± 0.169 |

The shared base scores 84.900. The CPU seed-4 comparison is a tie. This is a
small illustrative effect, not evidence of a universal improvement. CPU and GPU
RNG/numerics differ, so their outputs need not be identical.

To reproduce the validation separately from the notebook:

```bash
python -m unittest discover -s tests -p 'test_demo.py'
python -m unittest discover -s tests -p 'test_updates.py'
python examples/validate_demo.py --phase test --device cpu
# Optional CUDA repeat:
python examples/validate_demo.py --phase test --device cuda
```
