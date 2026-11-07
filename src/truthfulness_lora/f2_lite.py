#!/usr/bin/env python3
"""F2-inspired multi-objective LoRA experiment for the existing research project.

Training objectives, each with assistant-target-only cross-entropy:
  1. question -> right_answer
  2. question -> knowledge
  3. question + knowledge -> right_answer

This is an F2-*inspired* ablation. It does not implement the paper's entity/
attention token weights or hallucination-prone-layer selection. The fourth
HaluEval field (hallucinated_answer) is retained for subsequent diagnostics.

Reuses the tested model loading, labeling, and Slurm protection in the original
notebook. All paths are separate from the October 27 run.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


KNOWLEDGE_TOKEN_BUDGET = 600
MAX_SEQUENCE_LENGTH = 1024
LEARNING_RATE = 2e-5


def load_original_helpers():
    from .supervised import context
    return context()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "smoke", "pilot", "full"))
    parser.add_argument("--pilot-examples", type=int, default=1000)
    args = parser.parse_args(argv)

    # Calls the original code, including its protection against running a GPU
    # job on a login node and its no-overwrite checks.
    ctx = load_original_helpers()
    ctx["MAX_LENGTH"] = MAX_SEQUENCE_LENGTH

    suffix = f"f2_lite_seed42_{args.mode}"
    ctx["RUN_DIR"] = ctx["inside_project"](f"results/{suffix}")
    ctx["ADAPTER_DIR"] = ctx["inside_project"](f"adapters/{suffix}")
    ctx["SMOKE_RUN_DIR"] = ctx["RUN_DIR"]
    ctx["SMOKE_ADAPTER_DIR"] = ctx["ADAPTER_DIR"]
    ctx["SMOKE_EXAMPLES"] = 50
    ctx["SMOKE_PARAMETERS"].update(learning_rate=LEARNING_RATE)

    full_parameters = ctx["FULL_PARAMETERS"]
    full_parameters.update(
        learning_rate=LEARNING_RATE,
        eval_steps=1500,
        save_steps=1500,
        logging_steps=50,
    )
    if args.mode == "pilot":
        full_parameters.update(
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_steps=20,
        )

    splits = ctx["splits"]
    tokenizer = ctx["tokenizer"]
    original_tokenizer = ctx["tokenize_example"]
    from datasets import Dataset

    for split_name in ("train", "validation"):
        missing = {"knowledge", "question", "right_answer", "hallucinated_answer"} - set(
            splits[split_name].column_names
        )
        if missing:
            raise ValueError(f"{split_name} is missing required columns: {sorted(missing)}")

    def tokenize_f2_split(name, limit=None):
        source = splits[name]
        max_rows = len(source)
        if limit is not None:
            max_rows = min(limit, max_rows)
        elif args.mode in ("pilot", "preflight"):
            max_rows = min(args.pilot_examples if name == "train" else 100, max_rows)

        rows = []
        clipped_count = 0
        for index, example in enumerate(source.select(range(max_rows))):
            q = example["question"]
            k = example["knowledge"]
            a = example["right_answer"]
            rejected = example["hallucinated_answer"]
            if any(not isinstance(value, str) or not value.strip()
                   for value in (q, k, a, rejected)):
                raise ValueError(f"{name} example {index}: blank/non-text HaluEval field")

            knowledge_ids = tokenizer(k, add_special_tokens=False)["input_ids"]
            if len(knowledge_ids) > KNOWLEDGE_TOKEN_BUDGET:
                clipped_count += 1
                k = tokenizer.decode(
                    knowledge_ids[:KNOWLEDGE_TOKEN_BUDGET], skip_special_tokens=True
                )

            # All three records originate from the same HaluEval item.
            # The original tokenize_example masks the user prompt and trains
            # exclusively on the assistant response, including its EOS token.
            rows.extend([
                {"objective": "qa", "question": q, "right_answer": a},
                {
                    "objective": "retrieval",
                    "question": "Give the factual background needed to answer: " + q,
                    "right_answer": k,
                },
                {
                    "objective": "grounded_qa",
                    "question": q + "\n\nSupporting knowledge:\n" + k
                                + "\n\nUse the supporting knowledge to answer the question.",
                    "right_answer": a,
                },
            ])

        print(
            f"{name}: {max_rows} source examples -> {len(rows)} training records; "
            f"knowledge clipped in {clipped_count} source examples",
            flush=True,
        )
        print("Objective distribution:", dict(Counter(r["objective"] for r in rows)))
        dataset = Dataset.from_list(rows)
        return dataset.map(
            original_tokenizer,
            with_indices=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc=f"F2-lite {name} tokenization",
        )

    ctx["tokenize_split"] = tokenize_f2_split

    # Only a Slurm GPU job may run the model. No output is overwritten.
    print(
        f"F2-LITE MODE={args.mode} | LR={LEARNING_RATE} | "
        f"MAX_LENGTH={MAX_SEQUENCE_LENGTH} | knowledge budget={KNOWLEDGE_TOKEN_BUDGET}",
        flush=True,
    )
    if args.mode == "preflight":
        # Safe on a login node: tokenizer/data only; no model or CUDA.
        training = tokenize_f2_split("train", 50)
        validation = tokenize_f2_split("validation", 10)
        print(
            f"F2 DATA PREFLIGHT PASSED: {len(training)} train records, "
            f"{len(validation)} validation records. No GPU used.",
            flush=True,
        )
        return
    if args.mode == "smoke":
        ctx["run_lora_training"](smoke=True)
    else:
        ctx["run_lora_training"](smoke=False)
    print(f"F2-LITE {args.mode.upper()} COMPLETED", flush=True)


if __name__ == "__main__":
    main()
