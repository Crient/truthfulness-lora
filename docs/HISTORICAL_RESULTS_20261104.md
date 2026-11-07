> **Historical snapshot — November 4, 2026.** The original results text is retained below. These six-demonstration harness evaluations use a different protocol from the [final matched four-model results](RESULTS.md); their scores and provisional subgroup claims remain historical.

# Results, Paired Comparisons and Exploratory Findings

**Evidence freeze:** November 4, 2026  
**Dataset:** TruthfulQA, 817 paired questions for every complete run  
**Metrics:** MC1 (binary top-choice ranking accuracy); MC2 (fraction of normalized candidate likelihood on truthful options)

## 1. Complete benchmark: score table

The exact score fractions below come from saved `lm-evaluation-harness` aggregate JSON and were independently reproduced as means of the corresponding 817 per-question JSONL `acc` scores.

| System | MC1 score | MC1 % | MC2 score | MC2 % |
|---|---:|---:|---:|---:|
| Original Qwen3-1.7B | `0.2937576499388005` | **29.38** | `0.45871982837963327` | **45.87** |
| Approach 1: LoRA | `0.2668298653610771` | 26.68 | `0.4396841546634628` | 43.97 |
| Approach 2: F2-Lite | `0.26805385556915545` | 26.81 | `0.4526701831235688` | 45.27 |

All three full runs use the same document IDs, candidate-answer sets, prompt documents and sample count in the exported files. This supports a paired within-benchmark comparison. The score ordering is **original > F2-Lite > Approach 1** on both point estimates, with very small MC1 separation between the two trained models.

## 2. Differences and uncertainty

### 2.1 Recomputed bootstrap method

We independently loaded the audit's per-question JSONL files and compared task-matched scores for the same documents. For each comparison and metric, we:

1. Matched the 817 question IDs and checked full `doc` equality.
2. Sorted the numeric IDs.
3. Calculated paired deltas (left model minus right model).
4. Resampled 817 paired question deltas with replacement, **10,000** times.
5. Used `numpy.random.default_rng(42)` **reset for each comparison and metric**.
6. Reported percentile 95% confidence intervals (`0.025`, `0.975`).

This is a nominal question-bootstrap CI for a **fixed trained checkpoint pair**, not a confidence interval across training seeds, dataset families, prompt settings, or future tasks. No multiple-testing adjustment was applied.

### 2.2 Full results of the independent recalculation

| Pair (left − right) | MC1 change, pp | MC1 nominal 95% CI, pp | MC2 change, pp | MC2 nominal 95% CI, pp |
|---|---:|---|---:|---|
| Approach 1 − Original | **−2.693** | [−5.141, −0.367] | **−1.904** | [−4.086, +0.323] |
| F2-Lite − Original | **−2.570** | [−4.896, −0.367] | **−0.605** | [−2.687, +1.465] |
| F2-Lite − Approach 1 | **+0.122** | [−1.469, +1.836] | **+1.299** | [+0.052, +2.512] |

**Reading this table:**

- The original model is ahead of both fine-tuned models in MC1 on the observed data. Each nominal full-set MC1 interval versus the original lies below zero for the specified fixed comparison.
- On MC2, Approach 1 and F2-Lite both have negative point differences versus original, but each of those two intervals includes zero.
- F2-Lite exceeds Approach 1 in MC2 by +1.30 pp, with a *narrowly positive* nominal fixed-checkpoint interval. With multiple exploratory comparisons, one training seed, and post-hoc analysis, it warrants replication rather than a broad claim of significant hallucination reduction.
- F2-Lite's MC1 advantage over Approach 1 is only **+0.12 pp**, with an interval crossing zero widely.

### 2.3 Why some earlier 95% CIs have slightly different endpoints

The October 27 comparison notebook reported **Approach 1 − original** at MC1 [−5.14, −0.37] pp and MC2 [−4.13, +0.33] pp. The earlier working README reported **F2-Lite − Approach 1** MC2 around [+0.01, +2.57] pp. Our independent implementation yields [+0.052, +2.512] pp for the latter using sorted IDs and an RNG reset for each comparison.

These differences are compatible with finite Monte Carlo uncertainty and distinct code choices: the notebook consumes one RNG stream across tasks and follows file order; `src/compare_f2_lite.py` sorts IDs and initializes a new RNG per task. The exact script for the earlier F2-versus-Approach-1 interval is **not in the ZIP**, so those older endpoints are preserved as *reported but not independently reproduced*. Use the explicitly specified independent procedure for future comparisons and preserve an immutable script/version.

## 3. Per-question changes: MC1

MC1 is binary, so we can count questions where a model changed its outcome relative to the reference model. These counts are independently reproducible from the audit samples.

| Comparison | Previously incorrect → correct | Previously correct → incorrect | Unchanged | Net correct answers | Mean accuracy change |
|---|---:|---:|---:|---:|---:|
| Approach 1 vs original | 39 | 61 | 717 | −22 | −2.69 pp |
| F2-Lite vs original | 34 | 55 | 728 | −21 | −2.57 pp |
| F2-Lite vs Approach 1 | 24 | 23 | 770 | **+1** | **+0.12 pp** |

**Explanation:** The substantial MC2 advantage of F2-Lite over Approach 1 is almost invisible in MC1. F2-Lite increases the relative probability mass on truthful choices without often changing which single answer ranks first. This is why reporting **both** metrics is necessary.

## 4. Per-question changes: MC2

MC2 is continuous, so `improved` means a positive per-question difference and `worsened` means a negative one. The following are computed with an absolute zero-difference tolerance of `1e-10`; very small/saturated values can appear unchanged.

| Comparison | Increased MC2 | Decreased MC2 | Essentially unchanged | Mean change |
|---|---:|---:|---:|---:|
| Approach 1 vs original | 383 | 424 | 10 | −1.90 pp |
| F2-Lite vs original | 428 | 379 | 10 | −0.60 pp |
| F2-Lite vs Approach 1 | 449 | 360 | 8 | +1.30 pp |

An increase in MC2 is **not the same as switching from a hallucinated generated answer to a truthful generated answer**. It describes the likelihood assigned to an existing benchmark answer-choice set.

## 5. Pilot and smoke experiments

These runs were for feasibility checks, not final model selection or reliable generalization estimates.

| Checkpoint / task scope | Questions | MC1 | MC2 |
|---|---:|---:|---:|
| F2 pilot baseline | 100 | 30.00% | 47.72% |
| F2 pilot trained adapter | 100 | 26.00% | 48.68% |
| Approach 1 tuned smoke | 20 | 25.00% | 38.73% |

Independently recomputed pilot deltas (trained minus pilot baseline, 100 paired questions): MC1 **−4.00 pp** (95% CI **[−10.00, +1.00]**), MC2 **+0.95 pp** (95% CI **[−3.19, +5.21]**). The pilot baseline score must be compared only to the *same 100 questions*, never to the entire 817-question baseline.

## 6. Answer-length hypothesis: reported exploratory checks

The research question was whether long reference answers made F2-Lite worse, or whether improvements concentrated on one-word/very-short answers. The Unity session supplied two separate quartile summaries, but the **scripts, exact word-count normalization, and question-to-length mapping were not included in the ZIP**. We therefore preserve them here as **previously reported exploratory results**, not independently audited results.

### 6.1 Reported MC2 / F2 improvement by truthful-answer-length quartile

| Quartile | Questions | Reported median truthful-answer length | Mean F2 improvement (pp) | Percentage with positive change |
|---|---:|---:|---:|---:|
| Shortest 25% | 204 | 5.0 words | +1.89 | 55.9% |
| Middle-short 25% | 204 | 7.2 words | −0.74 | 55.4% |
| Middle-long 25% | 204 | 9.2 words | +2.21 | 54.4% |
| Longest 25% | 205 | 12.3 words | +1.84 | 56.6% |

*These figures were described in conversation as F2 improvement; the original script and precise comparator are missing. Confirm the comparator before treating the bins as a formal result.*

### 6.2 Reported MC1 accuracy transitions by correct-answer-length quartile

| Quartile | Questions | Median correct-answer length | Improved | Worsened | Unchanged | Net change | Accuracy change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shortest | 204 | 4.0 words | 5 | 7 | 192 | −2 | −0.98 pp |
| Middle-short | 204 | 8.0 words | 7 | 6 | 191 | +1 | +0.49 pp |
| Middle-long | 204 | 10.0 words | 6 | 6 | 192 | 0 | 0.00 pp |
| Longest | 205 | 14.0 words | 6 | 4 | 195 | +2 | +0.98 pp |

**Conclusion:** These bins **do not show a monotonic decline with answer length**. The MC2 mean changes fluctuate; positive-change rates are similar across bins. The MC1 transitions are sparse. A binary split or simple linear model of length would need its own prespecified analysis and a clear definition of which answer's length is measured. A median of 5–14 words does not test a literal one-word-answer effect.

## 7. TruthfulQA category analysis: provisional, post-hoc

### 7.1 Category mapping reported from Unity

The session reported that the TruthfulQA `multiple_choice` dataset did not contain category labels. A separate `generation` dataset provided categories, and exact normalized-question matching reportedly joined **817/817** questions to **38** category labels. The recovered mapping and matching implementation were **not packed into the audit**. These tables should be considered *user-reported exploratory analysis* until the join script, pinned dataset revision and mapping are preserved.

Three ad hoc conceptual groups were defined after selected question-level changes had been inspected:

- **A — Context-dependent/subjective (66):** `Subjective` plus `Indexical Error: Identity`, `Location`, `Time`, `Other`.
- **B — Factual-knowledge proxy (113):** `History`, `Language`, `Misquotations`, `Confusion: People`, `Confusion: Places`, `Science`, `Statistics`.
- **C — Other/mixed (638):** all other labels.

These are *analyst-defined proxies*, not ground-truth skill annotations. Group A's post-hoc selection increases the risk of selection bias and underestimated uncertainty.

### 7.2 Reported subgroup MC2 scores

| Group | n | Original | Approach 1 | F2-Lite | F2 vs original | F2 vs Approach 1 |
|---|---:|---:|---:|---:|---:|---:|
| A — Context-dependent/subjective | 66 | 62.34% | 61.34% | **76.42%** | **+14.09 pp** | **+15.09 pp** |
| B — Factual-knowledge proxy | 113 | **31.98%** | 31.93% | 30.84% | −1.14 pp | −1.09 pp |
| C — Other/mixed | 638 | **46.63%** | 44.30% | 44.60% | −2.03 pp | +0.29 pp |

Group A is **66/817 = 8.1%** of the total questions. It contributes approximately **+1.22 pp** of the overall +1.30-pp F2-vs-Approach-1 mean difference. Because Group C is much larger and near zero on this comparison, a strong local effect can coexist with a small overall result. This is a descriptive decomposition, **not** proof of a mechanism.

### 7.3 Reported subgroup changed-question counts

| Group | MC1 improved / worsened / unchanged vs Approach 1 | MC1 difference | MC2 improved / worsened vs Approach 1 | MC2 difference |
|---|---|---:|---|---:|
| A — 66 | 2 / 1 / 63 | +1.52 pp | 53 / 13 | +15.09 pp |
| B — 113 | 1 / 2 / 110 | −0.88 pp | 67 / 46 | −1.09 pp |
| C — 638 | 21 / 20 / 597 | +0.16 pp | 334 / 304 | +0.29 pp |

### 7.4 Reported details of Group A

| Original category | n | F2 vs original MC2 | F2 vs Approach 1 MC2 | MC2 items improved vs Approach 1 |
|---|---:|---:|---:|---:|
| Indexical Error: Identity | 9 | +5.90 pp | +24.45 pp | 9/9 |
| Indexical Error: Location | 11 | +26.77 pp | +34.35 pp | 9/11 |
| Indexical Error: Other | 21 | +18.62 pp | +12.57 pp | 20/21 |
| Indexical Error: Time | 16 | +11.23 pp | **−3.65 pp** | 9/16 |
| Subjective | 9 | +1.28 pp | +21.39 pp | 6/9 |

### 7.5 Reported 66-question bootstrap

| Group A comparison | Reported mean difference | Reported nominal paired-bootstrap 95% CI |
|---|---:|---|
| F2-Lite − original | +14.09 pp | [+3.83, +24.23] pp |
| F2-Lite − Approach 1 | +15.09 pp | [+8.85, +21.97] pp |

**Provenance/status:** These two confidence intervals were provided in the original research chat. They were not independently recalculated here because the selected 66 question IDs and category-join script are absent. They are conditional on this chosen subgroup and do not account for its discovery process, multiple exploration paths, or training-seed variability.

### 7.6 What would validate the subgroup observation?

1. Save the full category-to-document mapping, normalized match keys, dataset commit and duplicate-key checks.
2. Reproduce all 38 original categories, all 817 matched IDs and the exact analyst-defined 66/113/638 partition.
3. Reproduce group means and paired CIs using immutable code plus a fixed seed.
4. Inspect candidate-answer choice counts, lengths, correct-label counts and log-likelihood distributions for a *systematic* sample, not only extremes.
5. Repeat with independent training seeds and, preferably, another set of context-dependent questions not used to design the grouping.

## 8. Research interpretation

**Strongly supported by the saved artifacts:** supervised LoRA improved within-domain validation loss but reduced overall TruthfulQA MC1 relative to original; F2-Lite recovered some MC2 relative to Approach 1, without a corresponding MC1 improvement. Overall, original Qwen3 remains the strongest point estimate on both metrics.

**Plausible, currently unproven explanation:** F2-Lite may change relative candidate scoring for questions that depend on contextual reference, ambiguity or uncertainty. The current evidence cannot distinguish learned factual grounding from answer-length/choice-count effects, prompt-format sensitivity or the combined training-parameter changes.

**Next experimental decision:** Define Approach 3 ahead of training, with controlled settings and a held-out assessment. See [`APPROACH_3.md`](APPROACH_3.md).
