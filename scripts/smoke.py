"""Run a small real-model pipeline; this does not reproduce paper scores."""

import argparse
import gc
import json
from pathlib import Path

import torch
from smat.config import load_config
from smat.train import train_suite
from smat.suite import merge_suite, evaluate_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--merge", nargs="*", choices=("wa", "ta", "ties", "dare", "della"), default=[]
    )
    args = parser.parse_args()
    if args.steps < 4 or args.samples < 2:
        parser.error("use at least 4 steps (one SMAT step) and 2 samples")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    config.pop("_config_path", None)
    config["name"] += "_smoke"
    config["tasks"] = args.tasks or (
        ["NumGLUE-cm"] if config["experiment"] == "trace_llm" else ["MNIST"]
    )
    config["work_root"] = str(output / "training")
    config["train"].update(
        max_steps=args.steps, batch_size=2, log_every=1, num_workers=0
    )
    config["evaluation"].update(
        max_samples=args.samples, batch_size=4, num_workers=0, repeats=1, concurrency=2
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    train_suite(config, max_samples=args.samples)
    gc.collect()
    torch.cuda.empty_cache()
    if args.merge:
        merge_suite(config, output / "suite", args.merge)
    if args.evaluate:
        evaluate_suite(config, output / "suite", ["expert", *args.merge])
    (output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "scope": "smoke only; reduced steps, samples and repeats",
                "config": str(Path(args.config)),
                "tasks": config["tasks"],
                "steps": args.steps,
                "samples": args.samples,
                "mergers": args.merge,
                "evaluation": args.evaluate,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
