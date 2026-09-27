"""Summarize suite accuracy and training cost using the paper's definitions."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(path):
    payload = json.loads(Path(path).read_text())
    scores = {
        method: row.get("macro", row.get("macro_mean"))
        for method, row in payload["scores"].items()
    }
    # Evaluation stores fractions. Tables and this display use percentages.
    values = {method: 100 * value for method, value in scores.items()}
    mergers = ("wa", "ta", "ties", "dare", "della")
    values["avg"] = (
        statistics.mean(values[m] for m in mergers)
        if all(m in values for m in mergers)
        else None
    )
    metadata = payload["training_metadata"].values()
    seconds = sum(row["loop_elapsed_seconds"] for row in metadata)
    peaks = [
        value
        for row in metadata
        for value in row.get("peak_memory_allocated_bytes_by_device", {}).values()
    ]
    return {
        "name": payload["name"],
        "scores_percent": values,
        "suite_training_seconds": seconds,
        "peak_allocated_gib_per_gpu": max(peaks) / 1024**3 if peaks else None,
        "evaluated_tasks": payload["tasks"],
        "evaluation": payload["evaluation"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.summary], indent=2))
