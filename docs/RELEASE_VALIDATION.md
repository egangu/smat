# Public release validation

Checked on 2026-09-27.

The training, merging, evaluation, configurations, data manifests and tests are
unchanged from the initial code release. The installation recipe requires
Python 3.11 for the pinned SciPy version.

## Checks

- Local CPU suite: 60 tests discovered; 43 passed and 17 CUDA tests skipped.
- GPU suite: all 60 tests passed on Baton, with no skips (3.057 seconds).
  Checks on two NVIDIA H800 GPUs include Triton numerical
  parity, FP32/BF16 updates, two-device behavior, pinned-base offload, checkpoint
  restoration, merger receipts, data splits and metric checks.
- GPU environment: Python 3.12.3, PyTorch 2.11.0+cu129, CUDA 12.9, Triton 3.6.0,
  Transformers 5.6.0, torchvision 0.26.0+cu129, datasets 4.8.5, NumPy 1.26.4,
  SciPy 1.17.1 and SGLang 0.5.12.post1. Rouge 1.0.1 and SacreBLEU 2.5.1
  (with its missing dependencies) were added to an isolated dependency directory
  because they were missing from the existing environment.

These are numerical and pipeline unit tests, not a fresh reproduction of the
full paper experiments. The GPU checks reused an existing environment and do
not establish a clean installation of every package in the CUDA 13.0 recipe.
