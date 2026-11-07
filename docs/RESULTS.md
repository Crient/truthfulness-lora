# Final matched-protocol results

The [November 4 results](history/HISTORICAL_RESULTS_20261104.md) remain preserved as a separate historical record. The earlier six-demonstration scores are not comparable to this table.

The only headline results in this candidate use the frozen 817-question native-chat protocol. Historical six-demonstration harness evaluations are excluded from these comparisons. Values below are percentages or percentage-point differences; exact proportions and all individual/paired intervals are in [matched_results.json](../results/matched_results.json).

| Recipe | MC1 % | MC2 % |
|---|---:|---:|
| Base | 23.623 | 37.561 |
| SFT | 22.399 | 39.233 |
| F2-Lite | 21.053 | 38.176 |
| DPO | 23.501 | 42.050 |


MC1 correct counts: Base 193, SFT 183, F2-Lite 172, DPO 192 of 817. Zero maximum-score ties occurred. MC2 is the mean per-item true-choice mass after normalization over that item's choices.

| Comparison | MC1 delta pp [95% paired item CI] | MC2 delta pp [95% paired item CI] |
|---|---:|---:|
| DPO-minus-Base | -0.122 [-2.938, +2.815] | +4.489 [+1.488, +7.475] |
| DPO-minus-F2-Lite | +2.448 [+0.612, +4.284] | +3.874 [+2.348, +5.407] |
| DPO-minus-SFT | +1.102 [-0.490, +2.693] | +2.817 [+1.783, +3.891] |
| F2-Lite-minus-Base | -2.570 [-5.263, +0.122] | +0.615 [-2.228, +3.475] |
| F2-Lite-minus-SFT | -1.346 [-2.938, +0.367] | -1.057 [-2.230, +0.119] |
| SFT-minus-Base | -1.224 [-4.039, +1.591] | +1.672 [-1.335, +4.653] |


Three DPO comparisons reproduce the saved evaluator results; the other three are exploratory post-hoc comparisons. All intervals use 10,000 paired item bootstrap resamples, shared NumPy default_rng(42) indices, and [.025,.975] percentiles. They are nominal and omit training-seed, reviewer and protocol-selection uncertainty. DPO MC2 nominal intervals exclude zero against each comparator; MC1 nominal intervals include zero versus SFT/Base. MC1 versus F2 excludes zero nominally, but a post-hoc Holm correction across six MC1 exact discordant-binomial tests gives adjusted p≈0.0908 for that comparison.

## Exploratory error analysis

DPO−SFT MC1: 26 wins, 17 losses, 166 unchanged correct, 608 unchanged wrong. MC2: 430 positive, 383 negative and four near-zero changes at absolute tolerance 1e-12. The median MC2 delta is 0.000002758 in score units versus mean 0.0281665. The top 41 positive changes contribute 47.04% of positive gain mass; deleting them leaves a descriptive +0.348 pp mean over the other 776 items. This is an outcome-selected influence calculation, not a new confidence interval or principled deletion rule.

Unchanged-MC1-wrong items contribute +1.623 pp of the +2.817 pp MC2 difference, demonstrating that MC2 can improve without correcting the designated best-answer decision. The length-bin and choice-count associations are descriptive and confounded; no semantic subset or causal length explanation is confirmed. A uniform distribution over each item's choices gives average true-label mass 44.841%, above every model here, underscoring that absolute MC2 is not generated-answer factual accuracy.

The aggregate [transition CSV](../results/mc1_transitions.csv), [MC1/MC2 decomposition](../results/mc1_mc2_decomposition.csv), [feature bins](../results/feature_bins.csv), [correlations](../results/feature_correlations.csv) and [diagnostic JSON](../results/exploratory_diagnostics.json) retain gains and regressions. Per-question outputs and answer text remain private. Independent readers can reconcile aggregate arithmetic in this candidate but cannot reconstruct item resamples or logits without the excluded raw evidence.
