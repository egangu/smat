# A small Hugging Face ViT demo

[Notebook](smat_image_demo.ipynb) · [Colab](https://colab.research.google.com/github/egangu/smat/blob/main/examples/smat_image_demo.ipynb) · [Data](https://huggingface.co/datasets/yanggangu/SMAT-Tiny-Demo)

Start with the existing [HF ViT-Tiny](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k),
then train two experts: **CIFAR-10 object recognition** and **SVHN digit recognition**.
Compare Adam FT against Adam + the published `SMATStepper`, using AVG and Task
Arithmetic. Evaluate each expert on its own task before merging, then compare
the base, separate experts and merged models in one table and figure. All four
experts train during Run All; no trained expert is downloaded.

## Run

Open Colab, or use Python 3.10+ locally:

```bash
# A CPU wheel avoids downloading CUDA libraries on a CPU-only machine.
python -m pip install 'torch>=2.9' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/demo.txt
jupyter lab examples/smat_image_demo.ipynb
```

On macOS, install PyTorch from the default pip index instead. Choose **Restart
Kernel and Run All**. The notebook defaults to CPU, four threads, FP32, seed 0;
set `DEVICE="cuda"` to use a GPU. It installs the SMAT package from a pinned GitHub
commit with `--no-deps`, so the full research dependencies are unnecessary.
First-run downloads are approximately **23 MB of model weights + 22 MB of data**.

The visible notebook code covers the model, expert training, merging and
evaluation. Helpers handle downloads, linear-head calibration and plots.
There is no development-set search inside the notebook.

## Fixed experiment

- Backbone: the original pretrained ViT-Tiny encoder, about 5.5M parameters.
  Images resize from 32×32 to **64×64**. The patch embedding and positional
  embeddings stay frozen and are cached; **all 12 Transformer blocks train**.
- Each task: 2,000 balanced training images and 2,000 balanced official test
  images. The dataset includes the unused 500-image development subsets for
  provenance. Source revisions, hashes and exact indices are published with
  the data. Selected splits have no identical decoded images across splits.
- Fit two small task heads using the same training images, while the HF
  encoder remains unchanged. Freeze these heads before expert training. This
  calibrated encoder is the **Base** row. Inference knows the task identity;
  this is one shared encoder with two heads, not a unified 20-class classifier.
- Both methods use the same base, batches, Adam LR `1e-4`, batch 32 and **600
  updates per expert**. Every fourth SMAT update applies Scale (minimum 0.1),
  Mask (probability 0.8) and uniform Perturb (RMS 0.01). Only Transformer
  Attention/MLP weight matrices are masked. The external eager stepper handles
  simulated weights, gradients and restoring the expert.
- AVG uses coefficient 0.5; TA uses 0.75 in
  `base + coefficient * (delta_objects + delta_digits)`.
  Both methods use the same coefficients. Heads are retained, not averaged.

The [recipe](recipe.json) was frozen before held-out test evaluation. Seeds
**0–4** were predeclared; the notebook uses **0**, not a seed chosen for its result.
The shared learning rate and short training budget define this illustration;
it is not a comparison against every separately tuned FT configuration.

## Held-out results

The notebook reports **FT experts** and **SMAT experts** on their own tasks:
the CIFAR-10 column uses the object expert, and the SVHN column uses the digit
expert. Each expert row therefore uses **two encoders**. Base, AVG and TA each
use **one encoder** with two task heads. The Mean column averages the two task
accuracies; it does not imply that one specialist solves both tasks.

Mean task accuracy ± sample standard deviation across five CUDA training seeds:

| Merger | FT | SMAT | Paired gain (points) |
|---|---:|---:|---:|
| AVG | 59.340 ± 0.732 | 64.410 ± 0.745 | **+5.070 ± 0.430** |
| TA | 64.220 ± 1.486 | 68.285 ± 0.646 | **+4.065 ± 1.477** |

Base: **32.90%**. Every FT/SMAT merged model beats Base **on each task** for
all five CUDA seeds. All gains are positive; an individual gain can be below
3 points (TA, seed 4: +2.65). Variation covers expert-training RNGs, not
multiple bases or data splits.

In the CPU default run, the own-task expert accuracies are:

| Experts (two encoders) | CIFAR-10 | SVHN | Mean |
|---|---:|---:|---:|
| FT | 65.30 | 71.35 | 68.325 |
| SMAT | 70.45 | 77.90 | 74.175 |

These are single-seed results; the five-seed table above reports merged models.
Both CPU and CUDA Run All records below include the expert evaluations.

The CPU default seed gives AVG **59.750 → 66.150 (+6.400)** and TA
**64.875 → 69.725 (+4.850)**. CPU/CUDA floating-point and random streams differ.
Full records: [CUDA](results/test-cuda.json), [CPU default](results/test-cpu.json).
This demo does not reproduce the paper's benchmark scores or measure its
**<2% training-overhead** claim.

## Reproduce and inspect

```bash
python examples/validate_demo.py --device cuda            # five seeds
python examples/validate_demo.py --device cpu --seeds 0    # notebook default
python -m unittest discover -s tests -p 'test_demo.py'
```

The notebook's visible core functions are checked against the executable
source. Tests also check frozen heads/common initialization, disabled-SMAT
parity with FT, merge arithmetic, and routing each task to its own expert. Execution reports are in
[CPU Run All](results/notebook-cpu.json) and [CUDA Run All](results/notebook-cuda.json).

Actual Restart Kernel and Run All on dgx44, with cached downloads/installations:
**CPU 179.4 seconds**, four threads, peak RSS **1,201 MiB**; **CUDA 81.4 seconds**,
peak allocated GPU memory **433 MiB**. First-run downloads/installations and
laptop speed vary. Both executions completed the same **11 code cells** with
no errors and matched the corresponding seed-0 reference results exactly.
All six contract/code-consistency tests passed. `recipe.json` is the immutable
pre-test snapshot; completed test outcomes are recorded under `results/`.
