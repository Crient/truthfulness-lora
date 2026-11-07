# Historical Slurm launchers

These six launchers retain their original bytes. They reproduce historical SFT/F2-Lite workflows, including the earlier harness evaluation; the current matched evaluator is documented in [reproduction instructions](../../docs/REPRODUCTION.md).

Submit from the **repository root**, after adapting allocation settings to your cluster and arranging the original inputs/environment:

```bash
sbatch scripts/slurm/run_full_lora.sbatch
sbatch scripts/slurm/run_truthfulqa.sbatch
sbatch scripts/slurm/run_f2_lite_pilot.sbatch
sbatch scripts/slurm/run_f2_lite_pilot_eval.sbatch
sbatch scripts/slurm/run_f2_lite_full.sbatch
sbatch scripts/slurm/run_f2_lite_eval.sbatch
```

The scripts change directory to `TRUTHFULNESS_PROJECT`, otherwise `SLURM_SUBMIT_DIR`, otherwise the submission working directory. File paths inside each script are relative to that project root, not to the launcher's new folder. For submission from elsewhere, export `TRUTHFULNESS_PROJECT` as the checkout root and ensure Slurm exports it to the job. Do not `cd scripts/slurm` before submission.

The move was checked with shell syntax validation and a stubbed launch from a temporary project root; no training, inference or scheduler job was run. Partition, QOS, GPU constraints and time limits are historical settings, not current allocation advice. Existing output refusal checks remain in place.
