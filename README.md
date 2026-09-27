# SMAT: Simple and Efficient Merge-Aware Training

Official implementation of **SMAT: Simple and Efficient Merge-Aware Training**.

**Yanggan Gu¹\*, Yuanyi Wang¹\*, Zhen Li¹, Shuo Cai¹, Yuhang Liu¹,
Junzhuo Li², Zihao Wang³, Hongxia Yang<sup>1,4,5,†</sup>**

¹ The Hong Kong Polytechnic University (PolyU)<br>
² The Hong Kong University of Science and Technology (Guangzhou)<br>
³ The Chinese University of Hong Kong<br>
⁴ PolyU-Daya Bay Technology and Innovation Research Institute<br>
⁵ InfiX.ai

\* Equal contribution. † Corresponding author: [Hongxia Yang](mailto:hongxia.yang@polyu.edu.hk).

Training, merging and evaluation code for FT and SMAT with Llama-1B/8B and
CLIP ViT-B/32/L/14. Supported mergers: WA, TA, TIES, DARE and DELLA.

## Install

Use Python 3.11, PyTorch 2.11.0+cu130 and CUDA 13.0 on Linux with NVIDIA
GPUs. The paper experiments used H800 GPUs. The supplied 8B configuration uses two GPUs; the other configurations
use one GPU per expert.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/train.txt
python -m pip install -e . --no-deps
# Required for language evaluation:
python -m pip install -r requirements/eval.txt
```

The paper reports Python 3.10.12. This installation recipe uses Python 3.11
because the pinned SciPy 1.17.1 requires Python >=3.11.

## Prepare models and data

Follow [DATA.md](docs/DATA.md) for downloads and split preparation, then set:

```bash
export MODEL_ROOT=/absolute/path/to/models
export DATA_ROOT=/absolute/path/to/prepared-data
export OUTPUT_ROOT=/absolute/path/to/results
```

## Train, merge and evaluate

```bash
export CUDA_VISIBLE_DEVICES=0
python -m smat train-suite configs/main/llama1b_adamw_smat.json
python -m smat merge-suite configs/main/llama1b_adamw_smat.json "$OUTPUT_ROOT/llama1b_smat"
python -m smat eval-suite configs/main/llama1b_adamw_smat.json "$OUTPUT_ROOT/llama1b_smat"
python scripts/summarize.py "$OUTPUT_ROOT/llama1b_smat/summary.json"
```

Use `*_ft.json` for FT, `vitb32_adam_*.json` or `vitl14_adam_*.json` for vision,
and `CUDA_VISIBLE_DEVICES=0,1` for Llama-8B training.
See [RUNNING.md](docs/RUNNING.md) for configuration and command options.

## Quick check

This checks the pipeline with reduced samples and training steps.

```bash
python scripts/smoke.py configs/main/vitb32_adam_smat.json \
  "$OUTPUT_ROOT/smoke_vit" --merge wa ta ties dare della --evaluate
python -m unittest discover -s tests -v
```

The release passed all 60 tests on two H800 GPUs; see the
[validation report](docs/RELEASE_VALIDATION.md) for the environment and scope.

## Layout

- `smat/train/`: FT/SMAT updates, optimizers and GPU kernels.
- `smat/experiments/`: language and vision adapters.
- `smat/merge*.py`, `smat/eval/`: merging and evaluation.
- `configs/`: main configurations, Muon and operator ablation examples.
- `scripts/`, `manifests/`: asset preparation, checksums and split indices.
- `tests/`: numerical and pipeline checks.

See [NOTICE](NOTICE) for third-party attribution. Models and datasets are obtained separately.

## Citation

```bibtex
@misc{gu2026smat,
  title={SMAT: Simple and Efficient Merge-Aware Training},
  author={Gu, Yanggan and Wang, Yuanyi and Li, Zhen and Cai, Shuo and Liu, Yuhang and Li, Junzhuo and Wang, Zihao and Yang, Hongxia},
  year={2026},
  url={https://github.com/egangu/smat}
}
```

An arXiv identifier will be added when the preprint is announced. The code is
available under the [MIT license](LICENSE); model, dataset and third-party
licenses are described in `NOTICE`.
