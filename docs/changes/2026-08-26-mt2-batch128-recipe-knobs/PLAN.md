# Plan (Tier 1, single pass)

1. `ogpo_multitask_4task.sh`: `TASKS` + `BATCH` env knobs, defaults preserved; header docs.
2. `ogpo_multitask_4task_ref_maxlab.sbatch`: `GPU` ← `CUDA_VISIBLE_DEVICES`.
3. `ogpo_ref_smoke_maxlab.sbatch`: same `GPU` fix; memlog max across GPUs.
4. Verify: `bash -n`; DRY at defaults must be byte-identical to pre-change; DRY with
   overrides must show *only* the intended deltas; full-command diff against the
   recorded `mt4_ref` (job 10178692) flag line; re-run the recipe leg of
   `tests/ogpo/test_verifier_alignment.py` and separate new failures from the three
   known reds.
5. Then: 2-GPU / batch-128 memory smoke crossing `pg_start_step`, before the real run.
