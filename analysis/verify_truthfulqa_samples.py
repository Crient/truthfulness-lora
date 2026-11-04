#!/usr/bin/env python3
"""Read-only, CPU-only TruthfulQA paired-score audit.

Usage (from the original truthfulness-lora project root):

    python analysis/verify_truthfulqa_samples.py --root . --bootstrap 10000 --seed 42

Input: existing results_*.json and samples_truthfulqa_mc{1,2}_*.jsonl.
Output: printed per-run means, question transitions, and paired bootstrap CIs.

No model loading, downloads, training, GPU jobs, file modifications, or plots.
The bootstrap uses sorted numeric doc IDs and a fresh seed for each pair/task,
so its finite-Monte-Carlo endpoints may differ slightly from historical cells.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RUNS = {
    "original": Path("results/truthfulqa_leng_20261027/baseline"),
    "approach1": Path("results/truthfulqa_leng_20261027/tuned"),
    "f2_lite": Path("results/truthfulqa_f2_lite_seed42_full/tuned"),
}
TASKS = ("truthfulqa_mc1", "truthfulqa_mc2")
COMPARISONS = (
    ("approach1", "original"),
    ("f2_lite", "original"),
    ("f2_lite", "approach1"),
)


@dataclass(frozen=True)
class RunTask:
    scores: dict[int, float]
    docs: dict[int, dict]
    aggregate: float
    source: Path


def exactly_one(root: Path, pattern: str) -> Path:
    files = sorted(root.rglob(pattern))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one {pattern} under {root}, found {len(files)}: {files}"
        )
    return files[0]


def read_task(folder: Path, task: str) -> RunTask:
    samples = exactly_one(folder, f"samples_{task}_*.jsonl")
    aggregate_file = exactly_one(folder, "results_*.json")
    results = json.loads(aggregate_file.read_text(encoding="utf-8"))
    aggregate = float(results["results"][task]["acc,none"])

    scores: dict[int, float] = {}
    docs: dict[int, dict] = {}
    with samples.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = int(row["doc_id"])
            if doc_id in scores:
                raise ValueError(f"Duplicate doc_id={doc_id} at {samples}:{line_no}")
            raw_acc = row.get("acc")
            if raw_acc is None:
                raw_acc = row.get("metrics", {}).get("acc")
            if raw_acc is None:
                raise ValueError(f"Missing acc for doc_id={doc_id} in {samples}")
            acc = float(raw_acc)
            if not np.isfinite(acc) or not 0.0 <= acc <= 1.0:
                raise ValueError(f"Invalid acc={acc} for doc_id={doc_id}")
            if task.endswith("mc1") and acc not in (0.0, 1.0):
                raise ValueError(f"MC1 must be binary; doc_id={doc_id} got {acc}")
            scores[doc_id] = acc
            docs[doc_id] = row["doc"]

    if len(scores) != 817:
        raise ValueError(f"Expected 817 full-run questions, got {len(scores)}: {samples}")
    observed = float(np.mean(list(scores.values())))
    if not np.isclose(observed, aggregate, atol=1e-12, rtol=0):
        raise ValueError(
            f"Mean mismatch for {task}: samples={observed:.16f} "
            f"aggregate={aggregate:.16f}"
        )
    return RunTask(scores=scores, docs=docs, aggregate=aggregate, source=samples)


def same_questions(first: RunTask, second: RunTask, first_name: str, second_name: str) -> None:
    if first.scores.keys() != second.scores.keys():
        only_first = first.scores.keys() - second.scores.keys()
        only_second = second.scores.keys() - first.scores.keys()
        raise ValueError(
            f"Question ID mismatch {first_name}/{second_name}: "
            f"only first={sorted(only_first)[:10]}, only second={sorted(only_second)[:10]}"
        )
    for doc_id in first.scores:
        if first.docs[doc_id] != second.docs[doc_id]:
            raise ValueError(
                f"Question/choice document mismatch for ID {doc_id}: "
                f"{first_name} versus {second_name}"
            )


def paired_comparison(
    first: RunTask,
    second: RunTask,
    task: str,
    samples: int,
    seed: int,
) -> dict[str, float | int | tuple[float, float]]:
    ids = sorted(first.scores)
    delta = np.asarray([first.scores[i] - second.scores[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=float)
    for j in range(samples):
        bootstrap_means[j] = rng.choice(delta, size=len(delta), replace=True).mean()
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    tol = 1e-10
    improved = int(np.count_nonzero(delta > tol))
    worsened = int(np.count_nonzero(delta < -tol))
    return {
        "delta_pp": float(delta.mean() * 100),
        "ci_pp": (float(lower * 100), float(upper * 100)),
        "improved": improved,
        "worsened": worsened,
        "unchanged": len(delta) - improved - worsened,
        "n": len(delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")

    root = args.root.expanduser().resolve(strict=True)
    runs: dict[str, dict[str, RunTask]] = {}
    for name, relative in RUNS.items():
        folder = root / relative
        if not folder.is_dir():
            parser.error(f"Missing historical results directory: {folder}")
        runs[name] = {task: read_task(folder, task) for task in TASKS}

    print("SAVED METRIC VALIDATION (means equal stored aggregate JSON)")
    print("Name          Questions      MC1       MC2")
    for name, tasks in runs.items():
        print(
            f"{name:<13} {len(tasks['truthfulqa_mc1'].scores):>4}      "
            f"{tasks['truthfulqa_mc1'].aggregate * 100:>6.2f}%   "
            f"{tasks['truthfulqa_mc2'].aggregate * 100:>6.2f}%"
        )

    print("\nPAIRED COMPARISONS (left minus right; percentage points)")
    print(
        f"Resamples={args.bootstrap}, seed={args.seed}; "
        "sorted numeric IDs; fresh RNG for every comparison/task"
    )
    for left_name, right_name in COMPARISONS:
        print(f"\n{left_name} − {right_name}")
        for task in TASKS:
            left = runs[left_name][task]
            right = runs[right_name][task]
            same_questions(left, right, left_name, right_name)
            result = paired_comparison(left, right, task, args.bootstrap, args.seed)
            lower, upper = result["ci_pp"]
            print(
                f"  {task}: Δ={result['delta_pp']:+.4f} pp, "
                f"95% CI=[{lower:+.4f}, {upper:+.4f}] pp, "
                f"improved/worsened/unchanged="
                f"{result['improved']}/{result['worsened']}/{result['unchanged']}"
            )

    print("\nPASS: all full-run samples and aggregate results are internally consistent.")


if __name__ == "__main__":
    main()
