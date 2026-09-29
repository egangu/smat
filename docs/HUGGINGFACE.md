# Hugging Face models and data

The release contains **28 Table 1 experts**: FT and SMAT × Llama-1B and Llama-8B
× seven TRACE tasks, trained with seed 42. Each repository contains full BF16
weights and tokenizer assets, ready for `transformers` or SGLang.

- [Models](https://huggingface.co/collections/yanggangu/smat-table-1-experts-and-data-6abb7826f636ba703e4f532e)
- [Frozen TRACE train/dev/eval data](https://huggingface.co/datasets/yanggangu/SMAT-TRACE)
- [Paper](https://arxiv.org/abs/2609.33437)

Follow the [installation instructions](../README.md#install) first. Downloads need
`huggingface_hub`; training and evaluation need the full GPU environment.
The download helper pins Hub commits and checks payload SHA256 values from
[`manifests/hf_release.json`](../manifests/hf_release.json).

## Train with the released data

```bash
export HF_ENDPOINT=https://huggingface.co
export MODEL_ROOT=/absolute/path/to/models
export DATA_ROOT=/absolute/path/to/data
export OUTPUT_ROOT=/absolute/path/to/new-training-run

python scripts/download_hf.py data --data-root "$DATA_ROOT"
python scripts/download_models.py --model-root "$MODEL_ROOT" --models llama1b
python -m smat train-suite configs/main/llama1b_adamw_smat.json
```

Use `llama1b_adamw_ft.json` for FT. Use the corresponding `llama8b` model and
configuration for 8B. Each run needs a fresh `OUTPUT_ROOT`: training refuses to
overwrite existing experts. Downloaded Table 1 experts are for evaluation and
merging, not optimizer-state training continuation.

## Evaluate and merge released experts

```bash
export HF_ENDPOINT=https://huggingface.co
export MODEL_ROOT=/absolute/path/to/models
export DATA_ROOT=/absolute/path/to/data
export OUTPUT_ROOT=/absolute/path/to/released-experts

python scripts/download_hf.py data --data-root "$DATA_ROOT"
python scripts/download_models.py --model-root "$MODEL_ROOT" --models llama1b
python scripts/download_hf.py experts --model llama1b --method smat \
  --output-root "$OUTPUT_ROOT"

python -m smat eval-suite configs/main/llama1b_adamw_smat.json \
  "$OUTPUT_ROOT/results" --methods expert
python -m smat merge-suite configs/main/llama1b_adamw_smat.json \
  "$OUTPUT_ROOT/results" --methods wa ta ties dare della
python -m smat eval-suite configs/main/llama1b_adamw_smat.json \
  "$OUTPUT_ROOT/results" --methods wa ta ties dare della
```

For FT, change both `--method smat` and the configuration suffix to `ft`.
For 8B, change `llama1b` to `llama8b` in the download and configuration commands.
The suite evaluates all seven tasks with the paper's three decoding repeats.
Allow about 17.4 GB for seven 1B experts or 112.5 GB for seven 8B experts,
plus base models and any merged checkpoints. Released metadata omits hardware
timing records, so the summary reports unavailable training costs as `null`.

## Use one expert or dataset directly

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "yanggangu/Llama-3.2-1B-Instruct-SMAT-FOMC"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")
```

```python
from datasets import load_dataset

splits = load_dataset("yanggangu/SMAT-TRACE", "FOMC")
train, dev, evaluation = splits["train"], splits["dev"], splits["eval"]
```

For byte-for-byte experiment reproduction, use the pinned download helper above.
Both helper commands accept `--tasks FOMC` to fetch a single task; complete-suite
evaluation and merging require all seven experts. The JSON records contain `prompt` and `answer`.
Train preserves repeated examples, dev holds out unique prompts, and eval is the
unchanged TRACE evaluation set. NumGLUE-cm evaluates exactly 41 examples and does
not append test. Base-model licenses and original dataset terms continue to apply.
