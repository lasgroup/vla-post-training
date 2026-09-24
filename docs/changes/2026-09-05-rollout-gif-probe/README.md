# Rollout-GIF probe for a specific checkpoint

**Tier 1.** New read-only analysis script in the `scripts/probe_*` family. No
algorithm, jit signature, checkpoint layout, or config-dataclass change.

## Why

Maintainer request: qualitatively inspect `mt4_mt2_b128_ep2_s0`'s final
checkpoint (config `pi05_libero_online_ogpo_ref`, the `mt2_b128_ep2` arm of
[[mt2-batch128-run]]) on its own two tasks, `libero_90_38` and `libero_90_82`,
with the episode-length doubling that run trained under
(`collect.episode_steps_multiplier=2`, `EP_MULT=2`) — "how does it behave."

No existing script does this. The closest precedents are Tier-C
(`scripts/probe_counterfactual_rollouts.py`) and
`experiments/language_grounding/stage4_trained_instructions.py`, but both
answer narrower questions (counterfactual-return ranking; scene/instruction
grounding) and hardcode their own task/scene. This probe just drives
`cfg.collect.eval_tasks` through the env's normal
`reset(options={"task_id": ...})` task-switch path (`src/envs/libero.py`
`LiberoWrapper.reset`) — no synthetic BDDL, no custom scene.

See `BLAST-RADIUS.md` for the change spec, `DIFF.md` for what was actually
written, `VERIFICATION.md` for what could and could not be checked.
