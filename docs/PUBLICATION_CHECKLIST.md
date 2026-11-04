# GitHub Publication and Research Continuity Checklist

**Scope:** prepare this work as a credible undergraduate AI research portfolio and a reusable experiment record. This file distinguishes actions that can be completed now from checks requiring advisor/team approval. It does not authorize public release or a Git push.

## 1. Before sharing publicly

- [ ] Confirm with Professor Salem Othman which research code, aggregate results, plots, analyses, and datasets are approved for a personal public repository.
- [ ] Agree on attribution and wording of the project affiliation; avoid implying endorsement or sole ownership of team-authored code.
- [ ] Establish whether any manuscript, publication timing, intellectual-property policy or embargo affects release.
- [ ] Check HaluEval, TruthfulQA, Qwen, LM Evaluation Harness and any redistributed code/data licenses and terms.
- [ ] Verify no private participant data or other researchers' files are included.
- [ ] Check Git commit history as well as the working tree for credentials and confidential files before publication.

## 2. Recommended public content

| Normally suitable after review | Keep private or review individually |
|---|---|
| `README.md` and `docs/*.md` | Full audit ZIP with internal paths and per-question raw samples |
| Source scripts with sanitized, portable paths | `.venv/`, `hf_cache/`, `.pip_cache/` |
| Small aggregate metric CSV | Model weights, `*.safetensors`, adapters/checkpoints |
| Reproducibility example commands | API keys, tokens, `.env` and credentials |
| Safe, computed plots from aggregate data | Raw downloaded datasets, caches, unrelated team directories |
| Citation and methodology references | Unsanitized notebooks and raw SLURM logs containing local paths |

The answer to **what may be distributed** depends on licenses and the research team's policy; this table is a starting point, not authorization.

## 3. Pre-push audit commands (run from the intended Git repo only)

```bash
# Verify exactly where you are
pwd
git rev-parse --show-toplevel
git status --short

# Inspect what Git would stage, without staging anything
git status --ignored --short

# Search tracked/source-like files for obvious secret-bearing names
rg -n -i 'api[_-]?key|secret|access[_-]?token|password|Authorization: Bearer' \
  --glob '*.py' --glob '*.md' --glob '*.ipynb' --glob '*.sbatch'
```

**Do not run `git add .` blindly in the Unity research directory.** The original workspace includes large caches, adapters, raw benchmark JSONL and training metadata, and its source contains hard-coded Unity account paths. Review a carefully scoped `git add` list after `.gitignore` is correct. These commands alone do not publish or change tracked files.

## 4. Suggested ignore rules to merge into existing `.gitignore`

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
.env.*
!.env.example
.pip_cache/
hf_cache/
adapters/
tmp/
slurm_*.out
slurm_*.err
*.safetensors
*.bin
*.pt
*.pth
*.zip
```

**Review rather than blindly ignoring all `results/`:** published aggregate result JSON/CSV can improve transparency, but evaluation samples and checkpoints may require permission or exceed practical repository size. `*.ipynb` should be sanitized carefully for outputs, execution metadata, and paths before publishing.

## 5. Scientific readiness checklist

- [x] Original model and both adapted checkpoints' full-set aggregate results captured.
- [x] A1 and F2-Lite source/hyperparameters recovered from exported notebook/script.
- [x] Paired sample IDs and document contents checked independently.
- [x] Aggregate means rechecked from samples independently.
- [x] Full-set paired bootstrap recalculated with documented RNG and resampling procedure.
- [x] Negative/inconclusive outcomes explicitly disclosed.
- [x] F2-Lite's differences from full F2 disclosed.
- [x] F2-Lite LR, length, source-record expansion and compute confounds disclosed.
- [ ] Freeze/publish model and tokenizer revisions and historical environment lock.
- [ ] Verify per-adapter weight hashes and checkpoint selection.
- [ ] Recover and archive 38-category mapping and the 66-item subset code/IDs.
- [ ] Rerun category/bootstrap analysis with the same exported sample files.
- [ ] Test training/benchmark contamination and candidate-length confounds.
- [ ] Capture SLURM accounting and hardware provenance, if still available.
- [ ] Run multiple training seeds and controlled ablations for confirmatory claims.
- [ ] Add an independent evaluation of free-generated answers.

## 6. Website / resume description guidance

**Accurate concise portfolio wording:**

> Investigated HaluEval-supervised LoRA fine-tuning of Qwen3-1.7B on Unity HPC; compared a standard correct-answer objective with an F2-inspired three-objective variant using all 817 TruthfulQA questions. Reproduced paired evaluation analyses and identified a small MC2 recovery versus standard LoRA, while the untouched model retained the strongest overall benchmark scores.

**Additional exploratory detail, when space allows:**

> A post-hoc analysis suggested a larger MC2 difference on 66 context-dependent questions; this result is undergoing mapping-code verification and independent replication.

**Avoid:** “solved hallucinations,” “improved truthfulness by 14%,” “model achieved 76% accuracy on context questions,” or “replicated the full F2 paper.” None follows from the evidence. The 76.42% number is MC2 candidate probability mass on a selected subgroup, not top-choice or generated-answer accuracy.

## 7. Best next tasks for a compelling, scientifically honest GitHub project

1. **Integrate documentation** into a separate research repository/branch and open a faculty review request before publicizing it.
2. **Refactor portability** of absolute Unity paths and notebook-cell-index execution without altering historical output files. Preserve the original method as an immutable release tag or commit.
3. **Restore category provenance** by adding a script that joins pinned TruthfulQA `generation` metadata to the exact 817 `multiple_choice` questions with duplicate-key and unmatched-key checks.
4. **Add reproducible figures:** grouped MC1/MC2 bars; per-step validation curves; paired per-question MC2 delta distributions. State sources and uncertainty in captions.
5. **Predeclare Approach 3** method and controlled comparison before training.
6. **Expand evaluation** with multiple seeds and an independent benchmark or free-generation study.

## 8. Safe append process when Approach 3 is ready

Create a new experiment evidence bundle; verify it independently; then append to `docs/EXPERIMENT_LOG.md`, add methods to `docs/METHODOLOGY.md`, update `docs/RESULTS.md` and `results/aggregate_metrics.csv`, and refresh the top-level README. Keep the November 4, 2026 observations and their exploratory status unchanged as a time-stamped record.

**Current status:** research documentation ready for internal review; public GitHub push intentionally not performed.
