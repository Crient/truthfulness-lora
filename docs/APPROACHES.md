# Experimental approaches and implementation

## Common model and inputs

All final evaluated arms use Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. [Input identities](../configs/input_identities.json) identify base files, frozen benchmark and historical adapter bytes. Original SFT/F2 source calls loaded an unpinned offline cache; current cache/evaluation bindings do not retrospectively prove an immutable original execution-time snapshot. The portable staging command pins recoverable revisions explicitly.

## Base

The released checkpoint with no project adaptation. Evaluate it with the same question wrapper and candidate-likelihood code used for all adapters. It is a baseline recipe, not a separately trained seed in this study.

## Approach 1: HaluEval supervised LoRA

HaluEval QA source commit `12a856119f03975a94509091e8cada3e6be6ead7` contains 10,000 items. A shuffled seed-42 95/5 split yields 9,500/500; the split order was reconstructed against cached original rows. Each question targets its right_answer, without knowledge or hallucinated-answer input. LoRA rank16/alpha32/dropout0.05/all-linear; one epoch; FP16; train microbatch1; eval batch2; accumulation4; LR2e-4; linear schedule; fused AdamW; betas0.9/0.999; epsilon1e-8; weight_decay0; clip1; no warmup; seed42. Max full sequence512 with right truncation, prompt labels masked, actual answer-token retention asserted. EOS is trained only if retained. Completed2,375 updates; validation loss selected update2,000 (0.8938544393). Exact settings are in [sft.json](../configs/sft.json).

The portable supervised module extracts the original notebook's tokenizer/masking and Trainer functions without saved outputs. A local model/split and fresh run root replace hardcoded paths. Future saved adapter config stores that local model path; original hashes are not promised for a fresh run.

## Approach 2: HaluEval F2-Lite

The same 9,500/500 source items expand to 28,500/1,500 correlated task records: question→right answer; question→supporting background; question+background→right answer. Equal record counts do not imply equal token/gradient weights. Hallucinated_answer is validated but not used as a negative loss target. Knowledge cap600 tokens (zero source items clipped in the recorded full run); full length1,024; LR2e-5; one epoch; validation/save every1,500; all other shared LoRA/batch settings inherited. Completed7,125 updates; selected6,000 (validation CE1.1275528669). See [f2_lite.json](../configs/f2_lite.json).

F2-Lite is an independent lightweight recipe, not a full Faithful Finetuning reproduction: no entity/attention weighting or hallucination-prone-layer selection is implemented. LR, length, records and compute differ from SFT, precluding objective-only causal attribution. The original F2 code now calls extracted helpers instead of executing notebook cells.

## Approach 3: exploratory UltraFeedback DPO

Initialize trainable policy and separate frozen reference from the completed SFT adapter. Use the pinned cleaned Argilla source, fixed grouping/filters and seed-42 9,500/500 split; source strings are not rewritten or truncated. DPO beta0.1, FP32/eager, TF32 off, dropout off, rank16/alpha32; AdamW LR5e-6/betas0.9,0.999/eps1e-8/weight_decay0; clip1; accumulation4; constant schedule. Up to3 epochs, validation every500 and epoch end, patience3/min_delta1e-4. Initial SFT checkpoint is eligible. Lowest validation DPO loss alone selects; benchmark scores never select.

Recorded selected update3,000 has loss0.4930704061, compared with initial0.6931471825. Three subsequent non-improvements triggered successful stop at4,500. Selected12,000 and final18,000 pair presentations correspond to1.2632/1.8947 passes; epochs_completed1 counts fully completed epochs. Selected and stopping checkpoints differ. Reference unchanged is recorded; checkpoint manifests/complete state were verified privately.

The trainer retains atomic adapter/optimizer/scheduler/RNG/order/cursor checkpoints at accumulation boundaries. Portable inputs and explicit fresh-run claims replace private lineage/authorization files. Scientific quality failures remain failures; the CLI requires acknowledgement of exploratory limitations instead of fabricating a GO gate. The command starts a fresh experiment; CLI automatic resume/continuation is not exposed, although recovery primitives remain and are CPU-tested. No scheduler submission/requeue code is supplied.

## Matched evaluation

Native chat user wrapper: `Answer truthfully and concisely.`, blank line, `Question: {question}`. Score the continuation's empty thinking scaffold, complete answer and first im_end EOS; exclude user prefix, padding and post-EOS newline. Raw summed FP32 likelihoods, no length normalization. MC1 chooses the designated best true answer vs false alternatives with 1e-8 tie credit. MC2 normalizes true-answer likelihood mass in its multi-true answer set. [Scoring code](../src/truthfulness_lora/dpo/train.py) and [aggregation code](../src/truthfulness_lora/metrics.py) expose these choices. Separate candidates are scored independently, with no A/B answer-position prompt cue. There is no free generation or six-shot prompt in this evaluation.

## Portability boundary

Original algorithmic helpers and parameter settings are retained where documented; wrappers/input acquisition/output bindings are adapted and not execution-time originals. Rebuilt preference manifests differ because default private review-packet generation is omitted. This changes metadata hashes, not the retained source-selection algorithms. Source hashes and transformation descriptions are in [SOURCE_PROVENANCE.json](../release/SOURCE_PROVENANCE.json). CPU tests cannot certify full training equality, throughput, GPU memory or cross-platform bitwise determinism.
