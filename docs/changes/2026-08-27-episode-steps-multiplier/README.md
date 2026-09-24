# Episode steps multiplier (`collect.episode_steps_multiplier`)

**What:** a new `CollectionConfig` knob that multiplies the domain's episode
step limit — the `TimeLimit` truncation built from the hardcoded libero suite
map (`src/envs/libero.py:126-138`, libero_90 = 400) and molmo's hardcoded 450
(`src/envs/molmo.py:337-340`). Default 1 reproduces today's behavior bit-for-bit
everywhere.

**Why:** the maintainer wants a 2-task run (libero_90_38 + libero_90_82, the
two tasks PG currently hurts) with truncation doubled to 800, testing whether
the 400-step limit is what starves them. No existing flag can do this:
`collect.max_episode_steps` feeds only the value-bound fallback
(`value_distribution.py:128`), not the env.

**Why a multiplier and not wiring `TimeLimit` to `collect.max_episode_steps`:**
wiring the existing field would silently change truncation for every non-
libero_90 suite (spatial would go 220 → 400 with no flag set). The multiplier
leaves the per-suite map authoritative and is inert at its default.

**Recipe:** `scripts/ogpo_multitask_4task.sh` gains `EP_MULT` (default 1),
passed as `--collect.episode_steps_multiplier`.
