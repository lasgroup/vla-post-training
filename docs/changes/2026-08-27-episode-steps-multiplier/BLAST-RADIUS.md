# Blast radius — episode-steps multiplier

## Files to touch
1. `src/training/config.py` (~:418, `CollectionConfig`) — add
   `episode_steps_multiplier: int = 1` next to `max_episode_steps`.
2. `src/envs/libero.py:102` — `max_steps = get_max_steps_libero(task_suite_name)
   * config.collect.episode_steps_multiplier`.
3. `src/envs/molmo.py:337-340` — `max_episode_steps=450 *
   config.collect.episode_steps_multiplier` (keeps the two domains' semantics
   identical; leaving molmo un-wired would make the knob silently inert there,
   which is the deviation-protocol failure mode).
4. `src/rl/value_distribution.py:128` — `T = int(config.collect.max_episode_steps
   * config.collect.episode_steps_multiplier)` so the discount>=1 fallback can't
   drift from the env. **Dead in practice here** (every registered config has
   discount < 1, where `lower = -1/(1-gamma)` is T-independent).
5. `scripts/ogpo_multitask_4task.sh` — `EP_MULT` env knob + flag + header row.

## Verified consumers of episode length (no change needed)
- `TimeToSuccessAsRewardWrapper` (`wrappers.py:230-251`): -1 per step, bonus on
  terminate — per-step, length-free.
- `collect.py:54,154`: loops `while total_episodes < num_rollouts` — collects
  until N successes, no step budget of its own.
- `_save_episode_in_buffer` windowing (`filtered_sft_learner.py:848-895`):
  `n_windows = n_steps - act_h + 1` from the actual episode — length-generic.
- Value bounds (`value_distribution.py:116-140`): discount=0.995 < 1 in the ref
  config (`config.py:727`), so lower = -1/(1-gamma) = -200 and upper =
  success_reward_bonus — both truncation-independent. Bounds do not move.
- Replay/success buffers: capacity is transition-count (500k / 50k), not
  episode-count. Longer episodes = fewer, longer episodes at the same memory.

## Duplication sweep
- Env preamble clone family `stability_study.sh` ↔ `ogpo_multitask_4task.sh`:
  knob added to the mt4 recipe only (the request is scoped there); the two
  preambles have already diverged and knob sets differ. Noted, not replicated.
- `make_env` dispatch covers exactly libero + molmo; both wired.

## Inheritance sweep
- Config field is read in env construction and value bounds only — no learner
  method involved, no `update()` override interaction.

## Behavior deltas at EP_MULT=2 (intentional, flagged to maintainer)
- Failed rollouts take up to 2x wall time -> slower collection/eval on
  low-SR tasks (38 especially).
- Successes that use the added headroom discount the terminal bonus harder
  (0.995^T_chunks); discount was tuned for ~295-step successes.
