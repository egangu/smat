"""Run the frozen recipe for every predeclared seed, without any tuning."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch
from demo_experiment import TASKS, TwoTaskViT, fit_heads, train_expert, merge_experts, evaluate
from demo_utils import MODEL_ID, MODEL_REVISION, DATA_ID, DATA_REVISION, load_encoder, load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    start = time.perf_counter()
    encoder = load_encoder().to(args.device).eval()
    base = TwoTaskViT(encoder).to(args.device).eval()
    data = load_data(encoder, args.device)
    fit_heads(base, data)
    del encoder
    base_scores = evaluate(base, data)
    records = []
    for seed in args.seeds:
        rows = {"Base": base_scores}
        for method in ("FT", "SMAT"):
            experts = [train_expert(base, data, task, method, seed) for task in TASKS]
            rows[f"{method} experts"] = evaluate(dict(zip(TASKS, experts)), data)
            for merger, coefficient in (("AVG", .5), ("TA", .75)):
                rows[f"{method} {merger}"] = evaluate(merge_experts(base, experts, coefficient), data)
            del experts
        gains = {m: rows[f"SMAT {m}"]["Mean"] - rows[f"FT {m}"]["Mean"] for m in ("AVG", "TA")}
        assert all(row[t] > base_scores[t] for name, row in rows.items() if name != "Base" for t in TASKS)
        record = {"seed": seed, "rows": rows, "gains_pp": gains}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "device": args.device, "seeds": args.seeds, "threads": 4,
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset": DATA_ID, "dataset_revision": DATA_REVISION,
        "recipe_sha256": hashlib.sha256(Path(__file__).with_name("recipe.json").read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "timm", "numpy")},
        "seconds": time.perf_counter() - start,
        "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024),
        "records": records,
        "mean_gains_pp": {m: sum(r["gains_pp"][m] for r in records) / len(records) for m in ("AVG", "TA")},
    }
    output = args.output or Path(__file__).parent / "results" / f"test-{args.device}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output), "mean_gains_pp": report["mean_gains_pp"]}), flush=True)


if __name__ == "__main__":
    main()
