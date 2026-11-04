#!/usr/bin/env python3
"""Compare F2-lite full TruthfulQA MC1/MC2 against the saved October baseline.

CPU-only, does not modify any files. Requires the original baseline and the
new full evaluation to have generated the harness's per-question JSONL samples.
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("TRUTHFULNESS_PROJECT", Path.cwd()))
BASELINE = ROOT / "results/truthfulqa_leng_20261027/baseline"


def load_scores(folder, task):
    files = [path for path in folder.rglob("*.jsonl") if task in path.name]
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one sample JSONL for {task} in {folder}, found {files}"
        )
    scores = {}
    for line in files[0].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        score = row.get("acc")
        if score is None:
            score = row.get("metrics", {}).get("acc")
        if score is None:
            raise KeyError(f"Missing acc in {files[0]}")
        doc_id = str(row["doc_id"])
        if doc_id in scores:
            raise ValueError(f"Duplicate question ID {doc_id} in {files[0]}")
        scores[doc_id] = float(score)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    tuned_dir = ROOT / f"results/truthfulqa_f2_lite_seed42_{args.mode}/tuned"
    for task in ("truthfulqa_mc1", "truthfulqa_mc2"):
        baseline = load_scores(BASELINE, task)
        tuned = load_scores(tuned_dir, task)
        if args.mode == "full" and baseline.keys() != tuned.keys():
            raise ValueError(f"Baseline and F2 question IDs differ for {task}")
        if args.mode == "pilot" and not tuned.keys() <= baseline.keys():
            raise ValueError(f"Pilot includes questions absent from the baseline for {task}")
        paired_ids = sorted(tuned)
        differences = np.asarray([tuned[k] - baseline[k] for k in paired_ids])
        rng = np.random.default_rng(42)
        bootstrap_means = np.array(
            [rng.choice(differences, size=len(differences), replace=True).mean()
             for _ in range(10_000)]
        )
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        print(f"{task} (n={len(paired_ids)}):")
        print(f"  Baseline: {np.mean([baseline[k] for k in paired_ids]):.4f}")
        print(f"  F2-lite:  {np.mean([tuned[k] for k in paired_ids]):.4f}")
        print(f"  Delta:    {np.mean(differences) * 100:+.2f} percentage points")
        print(f"  Paired bootstrap 95% CI: [{lower * 100:+.2f}, {upper * 100:+.2f}] pp")


if __name__ == "__main__":
    main()
