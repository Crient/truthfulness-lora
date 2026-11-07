# Scientific limitations and conclusions

The completed study supports a metric-specific exploratory comparison of four complete training recipes under a frozen likelihood protocol. DPO has higher MC2 than each saved comparator and MC1 near Base, with an uncertain small increase over SFT. It does not establish learned factual truthfulness, general hallucination reduction or free-generation accuracy.

- SFT/F2-Lite use HaluEval; DPO adds UltraFeedback. Data, learning rate, precision, length, objectives, updates and training history are confounded. No DPO-only causal claim is supported.
- UltraFeedback preferences are GPT-4-derived general-quality judgments, not independently verified factual labels. Original gates failed: supported28/48=58.33% vs≥80%; unusable chosen8/48=16.67% vs≤10%; answer-only features66.01% and ngrams74.69% vs≤65%. Technical optimization success does not reverse these failures.
- The blinded audit was by the construction assistant with orientation hidden, not an independent human factual reviewer. Twenty non-endorsed cases include three reversals, three ties and14 holds; they are not20 proven wrong labels. Stratification/rare-source oversampling prevents treating the unweighted audit rate as population factual accuracy.
- One full training seed per adapted recipe; item-bootstrap CIs omit seed/judge/label/protocol-selection variability. Nominal comparisons have no family-wise adjustment. An interval spanning zero is not equivalence.
- MC1 and MC2 have different candidate sets and decision semantics. MC2 measures relative true-choice mass and may improve while MC1 stays wrong. No generation evaluation was performed in the final matched run.
- Gains are heterogeneous and positive-tail influenced. Post-hoc feature/subgroup/deletion analyses do not validate a mechanism or prespecified hypothesis.
- Four HaluEval supporting passages cross train/validation. Exact question checks and lexical grouping do not prove document, semantic or pretraining decontamination. Benchmark labels may be context/time dependent.
- Historical six-shot scores are not comparable to the final native-chat table. No old subgroup mapping was promoted to a final category claim.
- Original execution provenance and environment locks are incomplete. Portable code has CPU checks but no new GPU replay. Data/model assets are excluded, and raw item resamples cannot be reconstructed from aggregate exports alone.

Future work could predefine endpoints/multiplicity, repeat seeds, control data/objective/compute, construct document-disjoint splits, obtain independent factual annotations and evaluate generated responses on independent held-out benchmarks. These proposals are not completed experiments or authorization to run them. Publication rights and approval questions are separate blockers in [BLOCKERS.md](../release/BLOCKERS.md).
