"""Command line entry point for the compact SMAT experiment runner."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smat", description="SMAT benchmark runner")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, methods in [
        ("merge-suite", ["wa", "ta", "ties", "dare", "della"]),
        ("eval-suite", ["expert", "wa", "ta", "ties", "dare", "della"]),
    ]:
        item = commands.add_parser(
            name, help="fixed-parameter complete suite operation"
        )
        item.add_argument("config")
        item.add_argument("output")
        item.add_argument(
            "--methods", nargs="+", choices=[*methods, "base"], default=methods
        )

    item = commands.add_parser("train", help="train one task expert")
    item.add_argument("config")
    item.add_argument("task")
    item.add_argument("--max-samples", type=int)

    suite = commands.add_parser("train-suite", help="train a suite of task experts")
    suite.add_argument("config")
    suite.add_argument(
        "--tasks", nargs="+", help="default: every task in the experiment"
    )
    suite.add_argument("--max-samples", type=int)

    evaluation = commands.add_parser("eval", help="evaluate one checkpoint on one task")
    evaluation.add_argument("config")
    evaluation.add_argument("model")
    evaluation.add_argument("task")
    evaluation.add_argument("output")
    evaluation.add_argument("--max-samples", type=int)
    evaluation.add_argument(
        "--endpoint", help="SGLang endpoint; required for trace_llm"
    )
    evaluation.add_argument("--concurrency", type=int, default=32)
    evaluation.add_argument("--seed", type=int, help="override the TRACE decoding seed")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    config = load_config(args.config)
    if args.command in {"merge-suite", "eval-suite"}:
        from .suite import merge_suite, evaluate_suite

        operation = merge_suite if args.command == "merge-suite" else evaluate_suite
        operation(config, args.output, args.methods)
        print(args.output)
        return
    # Keep ``smat --help`` and JSON inspection usable on a login machine that
    # intentionally does not have the accelerator runtime installed.
    from . import eval as evaluation_runner
    from . import train

    if args.command == "train":
        checkpoint = train.train(config, args.task, args.max_samples)
        if train.is_rank_zero():
            print(checkpoint)
    elif args.command == "train-suite":
        checkpoints = train.train_suite(config, args.tasks, args.max_samples)
        if train.is_rank_zero():
            for checkpoint in checkpoints:
                print(checkpoint)
    elif args.command == "eval":
        evaluation_runner.evaluate(
            config,
            args.model,
            args.task,
            args.output,
            args.max_samples,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            seed=args.seed,
        )
