# Diff

Two files, one change each, plus dead-import cleanup.

## `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py`
- Scoring block (was 562-570): replaced the local
  `get_value_bounds` -> `make_value_distribution` -> `.mean()` -> `.min(axis=0)`
  chain with `summarize_critic_values(q_logits, self._config,
  critic_reduction=self._config.rl.critic.reduction)`. This is the same helper the
  critic loss uses, so the read-out path no longer re-implements it.
- Added `summarize_critic_values` to the `update_critic` import.
- Dropped the now-unused `from src.rl.value_distribution import get_value_bounds,
  make_value_distribution`.

## `src/rl/best_of_n/best_of_n_learner.py`
Identical change against its own package's `update_critic.summarize_critic_values`.

## Divergence from plan
None. The one judgement call: routing through `summarize_critic_values` rather
than adding an `if reduction == ...` branch, because it removes a clone rather
than adding a third copy of the reduction logic.

## Related, already landed earlier in the session
`advantage_weighted_sft_learner.py` `_bon_record` now also carries the per-env
untiled `state` and `prefix`, so probes can evaluate the V head at those states
without rebuilding the transform pipeline. Additive; inert when the hook is None.
