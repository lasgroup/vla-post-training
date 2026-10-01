# Add distributional (C51) critic for best-of-N

Integrates the C51 distributional Bellman backup from PR #58 onto current main
(post requeue-rework). Supersedes #58, which was branched from older main.

## What
- **C51 backup** for the best-of-N critic, opt-in via `rl.critic.use_distributional_critic`
  (default `False`). When on, the TD target projects the next-state V/Q distribution under
  `Tz = r + γz` (`categorical_project`) instead of collapsing it to a scalar; MC term unchanged.
- **`distributional_target_reduction`** (`"mean"` | `"min"`) for ensemble reduction in prob space.
- **BroNet** can now emit categorical logits (`num_bins`); `num_bins=1` keeps the prior scalar output.
- **Value-bounds bugfix**: resolved lazily in `get_value_bounds()` instead of cached in
  `__post_init__`, so CLI/YAML overrides of `discount`/`num_value_bins`/`max_episode_steps` take effect.
- **Guard**: `use_distributional_critic=True` is rejected for non-best-of-N configs (it's unimplemented
  for AWR/MPO/FlowGRPO/OGPO, which share `CriticTrainingConfig`) — fails loudly instead of silently
  running the wrong backup.

## Safety (other algorithms unaffected)
- **FilteredSFT**: imports none of the changed files; no `critic` field. Zero impact.
- **AWR**: default behavior unchanged. Shared imports (`get_value_bounds`, `make_value_distribution`,
  `_build_pi0_backbone_critic_defs`, BroNet critics) are all backward-compatible (`num_bins` defaults
  to 1; lazy bounds are transparent). AWR's own `update_critic.py`/learner are byte-identical to main.

## Behavior notes
- **Scalar BroNet best-of-N**: ensemble reduction changes `mean` → `min` (loss now branches on
  loss-type, not network-type; replaces `_bro_pessimistic_reduce(..., 0.0)`). Only affects
  best-of-N + BroNet + scalar runs.
- **Pre-existing, out of scope**: AWR + BroNet + `num_value_bins>1` builds a scalar critic (AWR's
  call site doesn't pass `num_bins`) — unchanged from main; candidate follow-up.

## Verification
- `get_config('pi05_libero_online_best_of_n')` builds; `use_distributional_critic` defaults False.
- `categorical_project` is mass-conserving with correct terminal handling (verified numerically).
- Smoke-tested best-of-N C51 on LIBERO-82 (BroNet, 101 bins) — trains, finite losses.
