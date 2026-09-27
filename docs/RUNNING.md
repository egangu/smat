# Running experiments

## Configurations

| Directory / prefix | Model | Optimizer |
|---|---|---|
| `configs/main/llama1b_adamw_` | Llama-3.2-1B-Instruct | AdamW |
| `configs/main/llama8b_adamw_` | Llama-3.1-8B-Instruct | AdamW |
| `configs/main/vitb32_adam_` | CLIP ViT-B/32 | Adam |
| `configs/main/vitl14_adam_` | CLIP ViT-L/14 | Adam |
| `configs/muon/llama1b_muon_` | Llama-3.2-1B-Instruct | Muon + auxiliary AdamW |

Each prefix has `ft.json` and `smat.json`. `configs/ablations/` contains examples
that omit one SMAT operator. Merge coefficients are set in each configuration.

## Parameters

- `seed`: training seed for this run. To run another seed, copy the configuration,
  change `seed` and use a separate `OUTPUT_ROOT`.
- `train`: optimizer, learning rate, batch size, precision-related options and
  SMAT settings.
- `evaluation.repeats`: number of evaluation repetitions per checkpoint.
- `merging`: coefficients and sparsity settings for each merger.
- `MODEL_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`: local model, prepared-data and output
  directories. Undefined environment variables are errors.

Language training uses BF16, batch 8 and maximum sequence length 512. Task epoch
counts are defined in `smat/experiments/trace_llm.py`. Llama-8B splits one expert
across two GPUs. Vision trains the FP32 image encoder with batch 128 for 4,000
steps per task; the text encoder, projection and classification heads are frozen.
The prepared training split already excludes dev examples.

The fast backend requires Triton. Pinned-base offload additionally needs host
memory for initialization weights (about 2.3 GiB for Llama-1B and 15 GiB for
Llama-8B), plus data, model-loading and process memory.

## Commands and outputs

The README shows the full pipeline. Use `python -m smat --help` or a command's
`--help` for arguments.

- `train CONFIG TASK`: one expert.
- `train-suite CONFIG --tasks TASK ...`: selected experts, or all configured tasks.
- `merge-suite CONFIG OUTPUT --methods wa ta ties dare della`: selected mergers.
- `eval-suite CONFIG OUTPUT --methods expert wa ta ties dare della`: evaluation.
- `scripts/summarize.py OUTPUT/summary.json`: percentage scores and training costs.

TRACE evaluates each task's `eval` split; NumGLUE-cm contains 41 examples and
never appends `test`. Vision evaluates the configured split. Evaluation uses
SGLang for language and frozen text classifiers for vision. Available GPUs run
independent evaluation jobs. `summary.json` contains task scores and macro means;
`Avg` averages the five merger scores and excludes Expert.

Training records synchronized loop wall time and peak allocated GPU memory.
The reporting script sums expert loop times and takes the maximum per-device
memory peak; these are distinct from parallel elapsed time.

Training refuses to overwrite an expert. Failed runs may leave `.partial`
directories. Use a new output root when changing a configuration or checkpoint.
Merge/evaluation can resume matching completed work; optimizer-state training
resume is not implemented. Outputs can include resolved local paths and optional
predictions, so review them before sharing.
