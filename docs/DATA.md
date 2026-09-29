# Models and data

Models, benchmark content and checkpoints are not bundled. Upstream licenses and
access requirements continue to apply. No author-specific path is required.

## Models

Place each model in `$MODEL_ROOT/<directory>`:

| Directory | Public source | Version identification |
|---|---|---|
| `Llama-3.2-1B-Instruct` | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Fixed per-file revisions in `manifests/llama_sources.json` |
| `Llama-3.1-8B-Instruct` | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Fixed per-file revisions in `manifests/llama_sources.json` |
| `clip-vit-base-patch32` | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| `clip-vit-large-patch14` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | `32bd64288804d66eefd0ccbe215aa642df71cc41` |

The 33 model-asset checksums are in `manifests/model_checksums.json`.
CLIP uses the Hugging Face revisions above. Llama files use the public
[1B](https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct) and
[8B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct)
ModelScope mirrors. Every required file's SHA256 and size matches the required
model assets. `manifests/llama_sources.json` pins each file to an immutable
Git revision; these revisions were recovered by matching the required bytes to
public metadata, rather than assuming the latest snapshot is equivalent.
The downloader verifies every file and refuses a mismatch. `--verify-only`
checks an existing local copy, regardless of where it was obtained.

```bash
python scripts/download_models.py --model-root "$MODEL_ROOT" --models llama1b vitb32
```

Llama's upstream license and acceptable-use terms apply to the mirror as well.
When obtaining assets from gated Hugging Face sources, accept the model terms
and authenticate through the normal client. Keep tokens out of configuration files.
Download model configuration, tokenizer/processor assets and weights; alternative
quantized weights do not reproduce full-parameter BF16 training. Training/evaluation
load local assets only, with no automatic network fallback.

## Prepared TRACE data on Hugging Face

For Table 1, download the frozen train/dev/eval files directly:

```bash
python scripts/download_hf.py data --data-root "$DATA_ROOT"
```

This replaces raw TRACE downloading and split preparation. The helper verifies
the exact frozen bytes. See [HF examples](HUGGINGFACE.md) for released experts and
training/evaluation commands. The raw-data workflow below remains available for
reconstructing the TRACE and vision splits from their original sources.

## Prepared CLIP8 data on Hugging Face

For Table 2, download the frozen image train/dev/test splits directly:

```bash
python scripts/download_hf.py data --suite clip8 --data-root "$DATA_ROOT"
```

The prepared Parquet shards preserve original image bytes and labels. The
included manifests use paths relative to each task directory, so the downloaded
data can be moved without changing the split. This replaces both raw vision
downloading and split preparation. See [HF examples](HUGGINGFACE.md#table-2-clip-vision-experts-and-data)
for the 32 released FT/SMAT experts and training/evaluation commands.

## Raw data layout

Choose an absolute raw-data directory, separate from the prepared split directory:

```text
raw-data/
  trace/<task>/{train,eval,test}.json
  clip8/stanford_cars/data/*.parquet
  clip8/dtd/data/*.parquet
  clip8/eurosat/data/*.parquet
  clip8/gtsrb/data/*.parquet
  clip8/mnist/mnist/*.parquet
  clip8/resisc45/data/*.parquet
  clip8/sun397/data/*.parquet
  clip8/svhn/cropped_digits/*.parquet
```

For TRACE, obtain the processed **LLM-CL-Benchmark_5000** collection from the
[official TRACE repository](https://github.com/BeyonderXX/TRACE), which links its
[benchmark archive](https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view).
Copy its seven task directories into `raw-data/trace/`. Do not substitute a
similarly named reprocessed collection: preparation verifies every source byte
hash, including original test files used for overlap exclusion.

Download the exact vision parquet shards:

```bash
python scripts/download_vision.py --raw-root /absolute/path/to/raw-data
python scripts/prepare_data.py --raw-root /absolute/path/to/raw-data \
  --output "$DATA_ROOT"
```

The download script pins the revisions in `manifests/data_sources.json` and fetches
only the required train/test shards. Preparation checks SHA256, byte size and row
count before using the committed indices. TRACE outputs must match the frozen
output SHA256 values exactly. Vision manifests reference the verified local
parquet bytes without duplicating the images. Keep the raw-data directory after
preparation. For a smaller check, both scripts accept `--tasks MNIST`; preparation
also accepts `--suite clip8` or `--suite trace`.

A checksum mismatch is a failure, not a reason to regenerate different indices.
Interrupted preparation leaves a task-specific `.partial` directory. Inspect it
and use a new output directory for a clean retry.

## Prepared splits

Committed manifests fix the training/dev indices and verify the input bytes.
Preparation reconstructs those indices without resampling. Training reads
`train.json`; `train_unique.json` is available for inspection. TRACE evaluates
`eval.json`, including 41 NumGLUE-cm examples. Vision reads its configured test
split. Keep the raw parquet files after preparing vision data.
