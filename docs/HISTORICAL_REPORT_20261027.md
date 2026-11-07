> **Historical document — November 4, 2026 release snapshot.** The original text below is preserved for research continuity. For the completed four-model study, use [current methods](APPROACHES.md), [matched results](RESULTS.md), [limitations](LIMITATIONS.md) and [release blockers](../release/BLOCKERS.md). Earlier scores and pending-work statements describe that snapshot.

> Historical report copied without changing its scientific claims. See RESULTS.md and AUDIT_AND_LIMITATIONS.md for the current interpretation, seed caveats, and bootstrap convention.

# Fine-Tuning Qwen3-1.7B with LoRA for Hallucination Mitigation

**Research Assistant:** Gechleng (Leng) Lim  
**Faculty Advisor:** Professor Salem Othman  
**Institution:** Wentworth Institute of Technology  
**Date:** October 27, 2026  
**Computing Environment:** UMass Unity HPC

---

## 1. Research Objective

The purpose of this experiment was to investigate whether fine-tuning
Qwen3-1.7B using the HaluEval dataset could improve its ability to
identify truthful answers.

We used Low-Rank Adaptation (LoRA) to fine-tune the model and
evaluated its performance using TruthfulQA.

**Research Question:**

Does supervised fine-tuning on correct HaluEval answers improve
Qwen3-1.7B's performance on TruthfulQA?

---

## 2. Dataset Preparation

We used 10,000 question-answer examples from the HaluEval QA dataset.

| Dataset | Examples |
|---------|----------|
| Training | 9,500 |
| Validation | 500 |
| Total | 10,000 |

The dataset contained four main fields:

- `knowledge`
- `question`
- `right_answer`
- `hallucinated_answer`

For this experiment, we trained the model using the question and
its correct answer.

The hallucinated answers were not explicitly used as negative
training examples.

We used random seed 42 for reproducibility.

---

## 3. Model and Training Configuration

**Base Model:** Qwen3-1.7B  
**Fine-Tuning Method:** LoRA  
**Computing Environment:** UMass Unity HPC

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | Linear layers |
| Learning Rate | 0.0002 |
| Training Epochs | 1 |
| Microbatch Size | 1 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 4 |
| Maximum Sequence Length | 512 |
| Precision | FP16 |
| Random Seed | 42 |

Approximately 17.4 million parameters were trainable through LoRA,
representing around 1% of the model's parameters.

The full training run completed successfully after 2,375 optimizer steps.

---

## 4. Training Results

During training, the validation loss consistently decreased.

| Training Step | Validation Loss |
|---------------|-----------------|
| 500 | 0.9506 |
| 1,000 | 0.9320 |
| 1,500 | 0.9149 |
| 2,000 | 0.8939 |

The lowest recorded validation loss was approximately **0.8939**,
observed at step 2,000.

This indicates that the model improved at predicting answers on
the held-out HaluEval validation set during training.

The trained LoRA adapter was saved successfully.

---

## 5. TruthfulQA Evaluation

We evaluated two models:

1. Original Qwen3-1.7B without fine-tuning.
2. Qwen3-1.7B with our trained LoRA adapter.

Both models were evaluated on the same **817 TruthfulQA questions**
using zero-shot evaluation.

### Evaluation Metrics

**MC1:** Measures whether the model ranks the single correct answer
higher than the incorrect alternatives.

**MC2:** Measures the average normalized probability assigned to
correct answers among the available alternatives.

### Evaluation Results

| Metric | Original Qwen | Fine-Tuned Qwen | Change |
|--------|---------------|-----------------|--------|
| TruthfulQA MC1 | 29.38% | 26.68% | -2.69 pp |
| TruthfulQA MC2 | 45.87% | 43.97% | -1.90 pp |

**Observation:**

The fine-tuned model performed worse than the original model on
both TruthfulQA metrics.

The reported changes were calculated using unrounded per-question
scores.

---

## 6. Statistical Analysis

We performed a paired bootstrap analysis using 10,000 bootstrap
samples to estimate 95% confidence intervals.

Both models were compared using their results on the same
817 questions.

### Confidence Interval Results

| Metric | Change | 95% Confidence Interval |
|--------|--------|-------------------------|
| MC1 | -2.69 pp | [-5.14, -0.37] |
| MC2 | -1.90 pp | [-4.13, +0.33] |

### Interpretation

**MC1:**

The 95% confidence interval is entirely below zero.

This provides evidence that fine-tuning decreased MC1 performance
under the unadjusted 95% paired-bootstrap analysis.

**MC2:**

The 95% confidence interval includes zero.

Although the fine-tuned model scored lower, the evidence is
insufficient to establish a statistically reliable difference.

These confidence intervals were not adjusted for multiple comparisons.

---

## 7. Discussion

The experiment produced an important finding:

**Improving validation loss on HaluEval did not lead to improved
TruthfulQA performance.**

We identified several possible explanations.

### 7.1 Training and Evaluation Objectives

Our model was trained to predict correct answers from HaluEval.

However, TruthfulQA measures how well the model distinguishes
truthful answers from misleading alternatives.

Improving answer prediction on HaluEval may not improve answer
ranking on a different benchmark.

### 7.2 Hallucinated Answers Were Not Used Directly

The HaluEval dataset contains both correct and hallucinated answers.

Our current training approach used only the correct answers.

The model therefore did not receive an explicit training signal
to rank correct answers above their corresponding hallucinated
alternatives.

A future experiment could investigate training methods that
compare both types of answers.

### 7.3 Possible Answer-Length Effects

Another team experiment identified potential differences between
the lengths of correct and hallucinated answers.

This raises the hypothesis that the model may learn patterns
related to answer length or when an answer should end.

We have not yet established whether answer length caused the
performance decrease observed in our experiment.

### 7.4 Limitations

This experiment used:

- One training dataset and objective
- One training epoch
- One random seed
- One external truthfulness benchmark
- One LoRA configuration

Additional experiments are necessary to determine whether the
observed results generalize across different training settings.

---

## 8. Conclusion

The LoRA fine-tuning process completed successfully.

During training, validation loss decreased from approximately
0.9506 at the first recorded evaluation to 0.8939 at the
best recorded evaluation.

However, the fine-tuned model did not improve TruthfulQA performance.

The results showed:

- MC1 decreased by approximately 2.69 percentage points.
- MC2 decreased by approximately 1.90 percentage points.
- The paired bootstrap analysis provided evidence of a decline
  on MC1.
- The MC2 result remained statistically inconclusive.

### Main Research Finding

**Supervised fine-tuning on correct HaluEval answers did not
improve Qwen3-1.7B's TruthfulQA performance under our tested
configuration.**

These findings suggest that the training objective and dataset
structure should be investigated before conducting additional
fine-tuning experiments.

---

## 9. Recommended Future Work

1. Examine questions where the original model answered correctly
   but the fine-tuned model did not.

2. Investigate whether answer length is associated with the
   observed performance changes.

3. Compare our results with the team's modified training approach
   that excludes the end-of-answer token from the training loss.

4. Experiment with a training objective that explicitly
   distinguishes correct answers from hallucinated answers.

5. Repeat promising experiments using additional random seeds
   to determine whether results are consistent.

---

## 10. Experiment Files

### Training Notebook

`notebooks/03_train_lora.ipynb`

### Comparison Notebook

`notebooks/04_compare_eval.ipynb`

### Saved LoRA Adapter

`adapters/qwen3_halueval_seed42/`

### Evaluation Results

`results/truthfulqa_leng_20261027/`

### Training Log

`slurm_lora_66178521.out`

### Evaluation Log

`slurm_truthfulqa_66187817.out`

---

## 11. References

1. Li, J., et al. (2023).
   *HaluEval: A Large-Scale Hallucination Evaluation Benchmark
   for Large Language Models.*
   EMNLP.
   https://aclanthology.org/2023.emnlp-main.397/

2. Lin, S., Hilton, J., & Evans, O. (2022).
   *TruthfulQA: Measuring How Models Mimic Human Falsehoods.*
   ACL.
   https://aclanthology.org/2022.acl-long.229/

3. Hu, E. J., et al. (2022).
   *LoRA: Low-Rank Adaptation of Large Language Models.*
   ICLR.
   https://arxiv.org/abs/2106.09685

4. Qwen Team. (2025).
   *Qwen3 Technical Report.*
   https://arxiv.org/abs/2505.09388

5. EleutherAI.
   *LM Evaluation Harness.*
   https://github.com/EleutherAI/lm-evaluation-harness