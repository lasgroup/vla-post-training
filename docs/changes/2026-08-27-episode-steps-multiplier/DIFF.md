# DIFF — episode-steps multiplier

All exactly as planned in BLAST-RADIUS.md; no divergence.

1. `src/training/config.py` — `CollectionConfig.episode_steps_multiplier: int = 1`
   added below `max_episode_steps`, with a comment stating what it multiplies
   and that bounds are T-independent for discount < 1.
2. `src/envs/libero.py:102` — `max_steps = get_max_steps_libero(task_suite_name)
   * config.collect.episode_steps_multiplier`.
3. `src/envs/molmo.py` — `TimeLimit(env, max_episode_steps=450 *
   config.collect.episode_steps_multiplier)`.
4. `src/rl/value_distribution.py:128` — `T = int(config.collect.max_episode_steps
   * config.collect.episode_steps_multiplier)` (discount>=1 fallback only).
5. `scripts/ogpo_multitask_4task.sh` — header row `EP_MULT` (:35), knob
   `EP_MULT="${EP_MULT:-1}"` (:222), flag
   `--collect.episode_steps_multiplier "$EP_MULT"` (:283).

## Implementer's own checks
- `bash -n` recipe: OK.
- `DRY=1` ref-recipe inspection: `--collect.episode_steps_multiplier 2` present,
  tasks/rollouts/batch/`--overwrite` all as intended.
- CPU: `get_value_bounds(ref config)` identical at multiplier 1 vs 2
  ((-200.0, 0.0) both); tyro parses `--collect.episode_steps_multiplier 2`.
- `pytest tests/ogpo -k "value or bounds or config"`: 46 passed, 3 skipped,
  2 failed — both failures reproduce with the change stashed (pre-existing
  `test_verifier_alignment.py` differential reds), not caused here.

## Post-verification fixes (see VERIFICATION.md F-A..F-D)
- `scripts/repro_egl_drain.py`, `scripts/egl_safe_probe.py` (×2): stub configs
  gain `episode_steps_multiplier=1`.
- `src/training/config.py` `OnlineTrainConfig.__post_init__`: raise on
  multiplier < 1.
- `scripts/ogpo_multitask_4task.sh`: header rows reordered.
- `tests/ogpo/test_episode_steps_multiplier.py`: contract-pin docstring updated;
  new below-one rejection test. 21 passed, 3 skipped.
- `docs/code/envs.md`, `docs/code/scripts.md` updated (step 4).
