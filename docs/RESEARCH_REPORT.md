# Four Qwen3-1.7B recipes: final exploratory research report

> **Transferred research narrative.** Independent recalculation and source-audit claims below describe the earlier private Unity review. Local branch integration checks aggregate consistency and packaging; it does not rerun item-level bootstraps, original inputs, training or model inference. See [the integration review](../release/INTEGRATION_REVIEW.md) for checks performed here.

Sanitized narrative adapted from the completed private scientific review. The researcher reports faculty permission to share the educational project; current distribution and licensing notes are in [RIGHTS_AND_NOTICES.md](RIGHTS_AND_NOTICES.md).

## Abstract

We compared the original released Qwen3-1.7B checkpoint, HaluEval supervised LoRA (SFT), HaluEval F2-Lite, and SFT-initialized exploratory UltraFeedback DPO using one matched likelihood protocol on 817 frozen TruthfulQA questions. DPO achieved MC1 23.501% and MC2 42.050%. Relative to SFT, its paired changes were +1.102 percentage points (pp) MC1, nominal 95% item CI [−0.490, +2.693], and +2.817 pp MC2 [1.783, 3.891]. MC1 was effectively unchanged relative to Base (−0.122 pp), while MC2 was +4.489 pp. Independent CPU reconstruction reproduced all stored scores and intervals. Descriptive error analysis reveals both substantial regressions and gains, with a small median MC2 change and appreciable influence from large positive changes. The results establish an exploratory candidate-scoring difference between complete recipes, subject to failed preference-quality/shortcut gates and single-seed limitations. They do not demonstrate acquisition of factual truthfulness.

## 1. Introduction and research questions

Adaptation can improve an optimization objective without improving transfer to a truthfulness benchmark. This study examines that distinction for a small released Qwen checkpoint and two supervised recipes followed by general-preference DPO. “Base” means the unmodified `Qwen/Qwen3-1.7B` checkpoint, not the separately named `Qwen3-1.7B-Base` product or an assertion that the checkpoint lacks upstream instruction tuning.

RQ1: Under an identical frozen evaluation, how do the four recipes differ in MC1 and MC2? RQ2: Does DPO improve candidate scoring over its actual SFT initialization? RQ3: Are changes broad across questions or concentrated in a few transitions and answer-set properties? RQ1/RQ2 follow the saved exploratory addendum; RQ3 and all new diagnostics in this review are post-hoc. No new hypothesis was registered here.

The benchmark defines MC1 and MC2 as different candidate-scoring tasks; neither is the truth rate of generated responses. See the [official TruthfulQA description](https://github.com/sylinrl/TruthfulQA/blob/main/README.md). DPO supplies a reference-adjusted pairwise preference objective, as introduced by [Rafailov et al.](https://arxiv.org/abs/2305.18290).

## 2. Dataset provenance and construction

### HaluEval for SFT and F2-Lite

The notebook loads `pminervini/HaluEval`, configuration `qa`, split `data`, without an explicit revision. Local cache metadata identifies commit `12a856119f03975a94509091e8cada3e6be6ead7`. All 10,000 saved rows are available. The seed-42 shuffled 95/5 split was reconstructed from the cached source, including exact saved order: 9,500 train and 500 validation. No missing/blank field rows, exact duplicate rows or normalized question overlap were found. Four normalized supporting-knowledge passages do cross the split, so source-document independence is not established. HaluEval is repurposed here as training material.

SFT uses each question and its `right_answer`. F2-Lite produces three correlated examples per source item: question→answer, question→background, and question+background→answer. Its 28,500/1,500 derived training/validation records are not independent new source questions. The hallucinated answer is checked for presence but is not a negative training target. F2-Lite is an independent recipe, not a reproduction of the full Faithful Finetuning algorithm.

### UltraFeedback for DPO

The consumed source is `argilla/ultrafeedback-binarized-preferences-cleaned`, commit `770076f077c4c5e298498fa32f804857f46d5134`: 60,917 exported pairs, pinned parquet SHA-256 `27dc59041d73654aecfa7685b6f839a0bbb2014d640ee92eefa047b262d1f1ca`. The [pinned source card](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned/blob/770076f077c4c5e298498fa32f804857f46d5134/README.md) describes average-preference binarization and benchmark cleaning. The archived upstream notebook selects the highest four-aspect average and a randomly drawn strictly lower-rated response. Upstream revisions and this random draw are not fully pinned; the experiment reproduces from exported bytes.

Filters require valid two-message branches and preserved prompt/response strings, chosen rating ≥4, rating gap ≥0.5, no embedded controls, nonidentical answers, complete chat branches ≤2,048 tokens, and rejected/chosen token and word ratios in [0.5, 2]. Leading-newline incompatibilities are excluded, not rewritten. Lexical connected groups and benchmark-connected groups are excluded before source-stratified deterministic selection with length-direction counterbalancing. This yields 18,534 eligible unique groups, of which 10,000 are selected: 9,500 train / 500 validation. All 10,000 selected records match their source rows and content hashes. No selected group crosses train/validation or recorded benchmark-linked groups. These checks are bounded lexical safeguards, not proof of semantic or pretraining independence.

## 3. Experimental design and training methodology

All final arms use the same local Qwen revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Actual adapter bytes and the selected Trainer checkpoints were verified. Historical SFT/F2 loading was unpinned in source; the available local bindings are stronger than proof of their original execution-time cache resolution.

| Setting | SFT | F2-Lite | DPO stage |
|---|---|---|---|
| Initialization | Released Qwen | Released Qwen | Completed SFT adapter |
| Data | HaluEval | Same HaluEval split, three tasks | UltraFeedback |
| LoRA | r16, alpha32, all-linear | Same | Same adapter architecture |
| Dropout | 0.05 | 0.05 | Disabled |
| LR | 2e-4 | 2e-5 | 5e-6 |
| Precision | FP16 | FP16 | FP32, eager, TF32 disabled |
| Complete input ceiling | 512, right truncation | 1,024, right truncation | 2,048, whole-pair exclusion |
| Effective records/update | 4 | 4 | 4 pairs |
| Epoch budget | 1 | 1 | Up to 3 |
| Actual optimizer updates | 2,375 | 7,125 | 4,500 |
| Selected update | 2,000 | 6,000 | 3,000 |
| Selection | Lowest validation CE | Lowest validation CE | Lowest validation DPO loss |

SFT/F2 use the shared native-chat wrapper and assistant-target CE. Their learning-rate decay and remaining optimizer defaults are inherited from Transformers; saved training-argument files and package evidence remain available. F2 knowledge is capped at 600 tokens, with zero source items reported clipped, before full-example truncation. Best saved validation losses are 0.8938544393 (SFT) and 1.1275528669 (F2). These losses concern different target distributions and cannot rank factual quality.

DPO minimizes `softplus(−beta × [(policy_chosen−policy_rejected) − (reference_chosen−reference_rejected)])`, using summed masked log probabilities and beta 0.1. A separate frozen effective SFT reference is used. AdamW uses betas (0.9,0.999), epsilon 1e-8, zero weight decay, clip norm 1 and constant LR; accumulation is four preference pairs. Validation occurs every 500 updates and at epoch end, with patience three/min_delta 1e-4, and initial SFT eligibility.

Validation loss fell from 0.6931471825 initially to 0.4930704061 at update 3,000; policy raw preference-ranking accuracy at that checkpoint was 54.2% (a general-preference diagnostic). The three subsequent validations did not improve by the required threshold, triggering successful early stopping at update 4,500. Selection used no TruthfulQA score. Selected exposure was 12,000 pair presentations (1.2632 passes); stopped exposure was 18,000 (1.8947 passes). Completion metadata's `epochs_completed: 1` counts fully completed epochs, not total fractional exposure.

The DPO training run completed in 9:02:56 allocated GPU time on one Quadro RTX 8000, with 2 CPUs and 32 GiB RAM. Trainer peak allocated memory was 19.852 GiB. The matched evaluation took 0:41:44 on one RTX 8000. Both accounting records report COMPLETED, exit 0:0. Atomic checkpoints contain adapters, optimizer/scheduler, RNG and exact sampler position/order; the selected checkpoint's complete manifest was checked. These are optimization/integrity successes, independent of benchmark interpretation.

## 4. Evaluation and statistical methods

The frozen benchmark SHA-256 is `8b894abd32dbb2838ce800ab893c8cef1d09f29d3e32a8b5879fa26999cce094`. All 817 question/target records match cached dataset commit `741b8276f2d1982aa3d5b832d3ee81ed3b896490`; the frozen order is authoritative. Every arm has one aligned record per question and all expected answer scores, without duplicate target choices, missing values or nonfinite scores.

Each answer is scored independently under the question-only user text `Answer truthfully and concisely.`, a blank line, and `Question: {question}`, using the native template. The continuation includes the empty `<think>` scaffold, answer and first `<|im_end|>` token. Prompt, padding and post-EOS newline are masked out. Raw FP32 summed continuation log likelihood is used, with no answer-length normalization. There is no free generation or six-shot demonstration in this evaluator.

MC1 assigns credit to the designated best true answer among its false alternatives, with equal credit within a 1e-8 maximum-score tie (zero observed ties). MC2 is the softmax-normalized probability mass of all true choices in its separate answer set. For item q, `MC2(q) = sum_true exp(score−max_score) / sum_all exp(score−max_score)`. Aggregate metrics are equal-question means.

The saved 10,000 resamples use NumPy default_rng seed 42 and paired sampling of 817 item indices with replacement, then 2.5/97.5 percentiles. This review independently recalculated metrics from choice scores, using Python stable sums rather than importing the historical evaluator, and reproduced every stored interval. Three additional recipe pairs use the same draws and are exploratory. All intervals are nominal; no multiplicity or training-seed adjustment is implied.

## 5. Results

| Recipe | MC1 correct / 817 | MC1 % (95% item CI) | MC2 % (95% item CI) |
|---|---:|---:|---:|
| Base | 193 | 23.623 [20.808, 26.561] | 37.561 [34.347, 40.804] |
| SFT | 183 | 22.399 [19.584, 25.337] | 39.233 [36.481, 42.058] |
| F2-Lite | 172 | 21.053 [18.237, 23.868] | 38.176 [35.375, 40.977] |
| DPO | 192 | 23.501 [20.563, 26.438] | 42.050 [39.154, 44.954] |


| Comparison | MC1 delta pp (95% paired CI) | MC2 delta pp (95% paired CI) |
|---|---:|---:|
| DPO-minus-SFT | 1.102 [-0.490, 2.693] | 2.817 [1.783, 3.891] |
| DPO-minus-F2-Lite | 2.448 [0.612, 4.284] | 3.874 [2.348, 5.407] |
| DPO-minus-Base | -0.122 [-2.938, 2.815] | 4.489 [1.488, 7.475] |
| SFT-minus-Base | -1.224 [-4.039, 1.591] | 1.672 [-1.335, 4.653] |
| F2-Lite-minus-Base | -2.570 [-5.263, 0.122] | 0.615 [-2.228, 3.475] |
| F2-Lite-minus-SFT | -1.346 [-2.938, 0.367] | -1.057 [-2.230, 0.119] |


Within this conditional item-bootstrap procedure, all three DPO MC2 intervals exclude zero. DPO MC1 is distinguishable from F2-Lite at a nominal 95% interval, while its intervals versus SFT and Base include zero. This is not evidence of equivalence on MC1. The additional three non-DPO comparisons all have intervals spanning zero under the final protocol. The observed matched-protocol ordering differs from the historical six-shot evaluation and must not be spliced into historical tables.

## 6. Exploratory error and sensitivity analysis

All results in this section are post-hoc, descriptive analyses of the fixed outputs, without selection of a new model or checkpoint. Full item IDs and values remain in the private research archive. The [exclusion inventory](../release/EXCLUDED_FILES.md) describes the transfer boundary; it does not contain those records. No raw training/benchmark answer text is copied into the new package.

### MC1 transitions

| DPO versus | Win | Loss | Unchanged correct | Unchanged wrong | Net correct |
|---|---:|---:|---:|---:|---:|
| Base | 70 | 71 | 122 | 554 | -1 |
| SFT | 26 | 17 | 166 | 608 | 9 |
| F2-Lite | 41 | 21 | 151 | 604 | 20 |


Against SFT, 43/817 questions change MC1 correctness and the net is nine additional correct items. MC2 improves in 430 questions and declines in 383, with four differences at or below 1e-12 in absolute value. The median MC2 difference is only 0.000002758 in score units (+0.000276 pp), compared with mean +2.817 pp. Differences range from −0.872941 to +0.827917. Thus “DPO improves most questions” would obscure both magnitude imbalance and near-zero changes.

The 41 largest positive DPO−SFT differences (about 5% of items) contribute 47.04% of positive gain mass. Removing them leaves mean +0.348 pp on the remaining 776 questions; removing the 10 largest gains leaves +2.031 pp. Removing the 10 largest absolute changes leaves +2.208 pp. These outcome-selected deletions characterize influence; they are not robust confidence intervals or a reason to discard adverse cases. For DPO−Base, deleting the 41 largest gains reverses the residual mean to −0.446 pp, showing stronger dependence on the positive tail.

MC2 changes also occur without MC1 correction: 608 DPO/SFT items remain MC1-wrong, yet contribute +1.623 pp of the aggregate +2.817 pp MC2 difference. This is possible because MC1 considers one designated true answer, while MC2 sums multiple true choices and responds continuously to probability changes.

### Unusually large differences and qualitative inspection

The private item-level analysis, excluded from this repository, records the ten largest gains and losses for each DPO comparison, with immutable benchmark IDs, scores and length/choice features. [EXCLUDED_FILES.md](../release/EXCLUDED_FILES.md) documents exclusions, not the individual extreme-case results. For DPO−SFT, the largest gain is item 640 (+0.827917), followed by 421 (+0.785094) and 734 (+0.738371). The largest losses are 112 (−0.872941), 47 (−0.547111) and 134 (−0.464349). These are selected extreme cases, not representative cases or independently adjudicated facts. Manual reading shows that items 640 and 734 move toward the benchmark’s qualified answers on organizing behavior and dietary fat. Item 421 gains large MC2 mass while retaining an incorrect MC1 top choice about a baby-bird myth. The losses include an unsupported universal dessert-consensus answer (112), a single-test psychopathy claim that stays MC1-wrong in both models (47), and a self-identity reversal (134). This mixed pattern does not establish a coherent improved factual skill or a validated error taxonomy; it records the benchmark labels and saved rankings without independent factual adjudication.

### Answer length, choice structure and concentration

The MC1 designated-true-answer token-count quartiles have DPO−SFT mean MC2 deltas −1.038, +4.837, +4.198 and +3.289 pp, with counts 205/204/204/204. The corresponding feature ranges are 2–9, 9–11, 11–15 and 15–32 raw tokens. Equal-count bins split tied lengths by benchmark index, so shared boundary values are intentional. This pattern is not a monotonic length trend or a causal length effect. Spearman rho with correct-answer token length is 0.106; with mean true-answer length it is 0.112, weak descriptive associations.

Choice-count quartile deltas are +1.290, +3.437, +2.436 and +4.111 pp. The choice-count Spearman association is 0.069. Gains are positive across these bins, while concentrated tails and confounding among length, wording, label fraction and topic remain. All calculated feature bins/correlations are supplied, avoiding selective disclosure of only favorable subsets. No versioned category map matching all final questions was recovered; historical 66-item category results remain provisional and are not reused.

### Evaluation artifacts and probability sensitivity

All scores/labels are aligned, finite and correctly aggregated. Permuting stored choice order or adding a common score constant leaves MC2 unchanged. MC1 always stores the true answer at index zero, but candidates are scored separately without position letters; the first-choice convention therefore does not create a trivial answer-position prompt cue. These checks cover aggregation, not a fresh GPU audit of sequence logits.

A uniform distribution over each question's MC2 choices has mean true-label mass 44.841%, above every model here. This arithmetic reference is not a factual reasoner, but illustrates the influence of choice sets and why an absolute MC2 value is not a generation-accuracy percentage. DPO's normalized answer-set entropy is 0.244, lower than SFT's 0.273 and higher than Base's 0.084; these are not factual calibration measurements. Rescoring SFT with temperature 0.5 or 2 changes mean MC2 to 39.351% or 39.056%, respectively, versus 39.233% at temperature 1. Those two simple scaling examples do not reproduce DPO's 42.050%, and DPO changes the MC2 argmax versus SFT on 110 questions. This rules out those particular pure temperature transformations as complete explanations, not broader calibration or formatting effects.

Token-level outputs were not saved for this final benchmark evaluation, so answer/EOS contributions cannot be separated from sequence sums. Earlier nine-case diagnostics established score-definition sensitivity in different candidate data; they motivate caution but cannot explain this evaluation causally. No new GPU inference was performed.

## 7. Discussion

The strongest supported result is a higher average candidate-normalized true-answer mass for the exploratory DPO recipe versus its SFT initialization, with limited MC1 correction and meaningful individual regressions. The result is compatible with changes in wording preferences, answer distribution or learned knowledge, but these explanations are untested. MC1 and MC2 measure different outcomes; pooling them into a single “truthfulness improvement” would hide that distinction.

Data provenance is central: the DPO stage adds a general instruction/preference distribution to an HaluEval-adapted policy. This design cannot separate extra data exposure, source mixture, optimization method, compute or scoring-format fit. Optimization success is well evidenced, and descriptive benchmark success is metric-specific. Statistical distinguishability under item resampling does not establish cross-seed robustness or causal factual learning.

## 8. Limitations

The comparisons concern complete recipes: SFT/F2-Lite use HaluEval, whereas the DPO stage adds UltraFeedback to SFT. Learning rate, sequence length, objectives, update budgets, precision and training history also differ. No causal attribution to DPO alone is supported. UltraFeedback labels reflect general GPT-4 preferences, not independently verified factual truthfulness.

The historical quality gates remain failed: 28/48 supported preferences (58.33%; requirement ≥80%), 8/48 unusable chosen responses (16.67%; maximum 10%), and answer-only shortcut estimates 66.01% / 74.69% (maximum 65%). The audit was performed by the construction assistant with source orientation hidden, not independent human factual review. Ties and unresolved holds are not all proven incorrect labels. The v3 user authorization accepted these confounds for exploration; it did not turn earlier NO-GO decisions into scientific passes.

Item-bootstrap intervals condition on these fitted checkpoints, questions, labels and scoring rules. They omit training-seed, label/reviewer, benchmark-construction and protocol-selection uncertainty. There is one full training seed per recipe, no independent replication, and no family-wise adjustment of the headline intervals. Historical six-shot harness scores are not comparable to this final native-chat evaluation. Higher MC2 does not establish higher factual accuracy in free generation.

Additionally, four HaluEval validation passages appear in training; lexical filtering does not certify document or semantic isolation. Exclusion chains and length balancing alter UltraFeedback coverage. All labels can be imperfect; some TruthfulQA answers depend on context or time. The official project now describes a newer multiple-choice setting and changes to context-sensitive items; this study deliberately preserves its legacy frozen MC1/MC2 bytes rather than retroactively changing the benchmark. See the [maintainers' update](https://github.com/sylinrl/TruthfulQA). No independent factual adjudication of this final benchmark, held-out category validation, complete upstream data-recreation seed, or fresh model-likelihood replay is included.

## 9. Conclusions and future research

The completed experiment supports review of a reproducible, exploratory complete-recipe comparison: DPO improves MC2 over all three saved comparators under the final matched protocol, and has MC1 near Base with a small uncertain increase over SFT. It does not establish broad hallucination reduction or a DPO-specific factual-learning mechanism. Scientific review found aggregate consistency and qualified conclusions; public release remains pending.

Future work should predefine both MC endpoints and multiplicity handling, use several training seeds and independent evaluation sets, isolate data/objective/compute in controlled ablations, construct document-disjoint validation splits, and obtain independent factual audits of preferences and generated outputs. Length/EOS/scoring sensitivity should be tested under a separately authorized protocol. These are proposals, not completed experiments or authorization to launch them.


## 10. Public-safe provenance and verification boundary

The included [matched results](../results/matched_results.json), [model identities](../results/model_identities.json), [training summary](../results/training_summary.json), [source hashes/adaptations](../release/SOURCE_PROVENANCE.json) and [QC scope](../release/QC_REPORT.md) replace private operational links. Per-question source evidence remains excluded. This narrative is a sanitized derivative, not a replacement for the immutable private report. Read [reproduction instructions](REPRODUCTION.md), [limitations](LIMITATIONS.md) and [release blockers](../release/BLOCKERS.md) before interpreting the package as publicly releasable or fully replayed.
