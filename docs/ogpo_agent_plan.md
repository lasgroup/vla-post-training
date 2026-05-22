# OGPO Agent for VLA post-training — Implementation Plan

This document plans the port of the **policy-extraction subroutine** of OGPO
(`/users/mananaga/OGPO`) into the VLA post-training framework
(`/users/mananaga/vla-post-training`) as a new agent called **`ogpo_agent`**,
running on LIBERO with a Pi05 flow policy. It mirrors the structure of the
existing `awr_agent` and `flow_grpo_agent` (`scripts/awr_agent/`,
`scripts/flow_grpo_agent/`).

> Scope: the "policy extraction" half of OGPO only. The OGPO Q-learning
> machinery (CalQL, ensembles with chi²-PO blending, MIP-Q, success buffer,
> q_warmup phase) is **not** in scope for the first cut — we reuse VLA's
> existing critic (`StateActionEnsembleDecoder` + `StateValueEnsembleDecoder`
> from `src/rl/networks/decoders/values/`).

---

## 1. Background

### 1.1 What OGPO's policy extraction actually does

The OGPO `square.sh` launcher configures the canonical "vanilla" OGPO path:
`use_awr=false`, `use_fpo=false`, `chi_po=false`, `adv_strategy=vanilla`,
`use_tapered_noise=true`, `error_correct_sde_to_ode=true`,
`group_num_samples=32`, `clip_epsilon=0.01`, `bc_coeff=1.0`,
`use_bc_regularization=true`. Walking through
`OGPOAgent.actor_loss` in `ogpo/agents/ogpo.py:558-861`, the **vanilla branch**
(lines 778-792) executes:

1. **Sample G action chains from the OLD policy via SDE.**
   `_sample_actions(..., use_target=True, num_samples=G)` returns
   `(actions, chains, old_lp, sigmas)` where `chains` is the full SDE
   trajectory `[x_T, x_{T-1}, ..., x_0]` and `old_lp` is the summed transition
   log-prob under the *target* (EMA / "old") actor. `sample_flow_actions_sde`
   in `pg_helper.py` implements tapered-noise SDE
   `σ_t = σ · √(1-t)` and stop-gradients on each `x_next`.

2. **Recompute log-probs of the same chains under the CURRENT actor.**
   `_compute_current_log_probs(obs, chains, grad_params)` calls
   `compute_flow_log_prob` which re-evaluates the per-step Gaussian
   transition densities along the *fixed* chain — this is the standard
   importance-sampling trick that makes a flow policy compatible with PPO.

3. **Compute group-relative advantages.**
   `_compute_advantages(...)` calls `_compute_q_values` over the G samples
   per state, then `compute_ogpo_advantages` which subtracts a per-state
   baseline (group mean / max / etc., based on `adv_strategy`) and applies
   optional clipping (`adv_clip_min`).

4. **PPO clipped surrogate.**
   `compute_ppo_loss(lp, old_lp, adv, config)` with `clip_epsilon=0.01`.

5. **BC regularization.** When `use_bc_regularization=true`, adds
   `bc_coeff *` flow-matching loss on demo actions (`bc_loss` at line 2531,
   which uses `compute_cfm_loss` / `get_flow_targets`).

6. **Optional auxiliaries** (off in square.sh, but worth being aware of):
   `entropy_coeff * ent_loss`, forward-KL regularization, denoiser head loss,
   one-step distillation, FQL.

Steps **1, 2, 4, 5** are the "policy extraction" core. Step 3 needs critic
Q-values; we'll use VLA's existing critic stack.

### 1.2 VLA post-training architecture

VLA's online learners follow a strict separation
(see `src/rl/filtered_sft_agent/filtered_sft_learner.py`):

- `Agent` ABC (`src/rl/agent.py`) defines the rollout/buffer/update interface
  the runner in `src/training/collect.py` uses.
- `FilteredSFTLearner` owns the **policy `_train_state`** (Pi0 model with
  optional EMA), the **online `ShardedReplayBuffer`**, sharding, checkpoints,
  episode preprocessing, and action sampling via
  `openpi.policies.policy_config.create_trained_policy`.
- `AdvantageWeightedSFTLearner` adds two critic train states
  (`_state_action_critic_state`, `_value_state`) and a normalizer.
- `FlowGRPOLearner` is the most relevant precedent: it sub-classes the AWR
  learner and only swaps the actor `train_step` for `flow_grpo/update_actor.py`.
- Each algorithm registers a config dataclass
  (`FlowGRPOSFTLearnerConfig` in `src/training/config.py:181`) and a config
  entry in the `_CONFIGS` list (lines 463-469), plus a `scripts/<algo>/exp.py`
  + `launcher.py`.

### 1.3 Why this port is feasible

`openpi.models.pi0.Pi0.sample_actions(..., noise_level>0, return_info_dict=True)`
already returns a structure that contains `log_prob` per SDE step
(`openpi/src/openpi/models/pi0.py:335-440`). FlowGRPO already consumes this
output. Additionally, `Pi0.get_dist_and_log_prob(x_t, sample, time,
observation, dt, noise_level)` (line 169) returns the per-step Gaussian and
its log-prob of an *arbitrary* sample — exactly what OGPO needs to score chains
under a different (current vs. old) set of weights.

So pi05's flow head already exposes the two operations OGPO requires:
- *Generate* `(chains, old_lp)` under the EMA/target weights, with full SDE
  trajectory in the info dict.
- *Score* a frozen chain under current weights (per-step
  `get_dist_and_log_prob`), summing across the chunk and across SDE steps.

---

## 2. Target file layout

Mirror the layout `flow_grpo` uses:

```
src/rl/ogpo/
    __init__.py
    ogpo_learner.py          # OGPOAgentLearner(AdvantageWeightedSFTLearner)
    update_actor.py          # JIT-friendly PPO + BC train_step
    sampling.py              # SDE chain sampling + chain rescoring helpers

scripts/ogpo_agent/
    __init__.py
    exp.py                   # copy of awr_agent/exp.py with OGPOAgentLearner
    launcher.py              # copy of awr_agent/launcher.py with new defaults

docs/ogpo_agent_plan.md      # this file
```

A new `OGPOSFTLearnerConfig` dataclass is added to
`src/training/config.py`, and a new config entry
`pi05_libero_online_ogpo_sft` is appended to `_CONFIGS`.

---

## 3. Component-by-component plan

### 3.1 Config: `OGPOSFTLearnerConfig`

In `src/training/config.py`, subclass `AdvantageWeightedSFTLearnerConfig` so
we inherit critic settings (decoder hidden dims, ensembles, EMA, normalizer,
mc_returns toggle, etc.):

```python
@dataclasses.dataclass(frozen=True)
class OGPOSFTLearnerConfig(AdvantageWeightedSFTLearnerConfig):
    # PPO / IS-ratio
    group_num_samples: int = 8          # G in OGPO; 32 in square.sh is too big for Pi05
    clip_epsilon: float = 0.01
    entropy_coeff: float = 0.0
    # Stochastic flow sampling
    num_sde_steps: int = 10             # matches flow_steps in square.sh
    ft_sde_steps: int = 10              # last K steps over which IS ratio is summed; -1 = all
    noise_level: float = 0.3            # tapered noise σ; FlowGRPO uses 0.3
    # Advantage shaping
    adv_strategy: str = "vanilla"       # 'vanilla' (group-mean) | 'max' | 'subtract_v'
    subsample_bon: bool = False         # OGPO's subsample_bon (group-relative bn)
    adv_clip_min: float | None = None
    # BC regularization
    bc_coeff: float = 1.0
    use_bc_regularization: bool = True
    # Bookkeeping
    use_mc_returns: bool = False        # if True, advantage = MC - V (like AWR)
```

Then register a config entry in `_CONFIGS` (mirroring lines 462-469):

```python
make_base_libero_config(
    name="pi05_libero_online_ogpo_sft",
    rl_config=OGPOSFTLearnerConfig(
        policy_update_interval=20,
        policy_training_start_step=100,
        group_num_samples=8,
    ),
),
```

**Design call** — `group_num_samples=32` (OGPO default) means each policy
update samples 32 SDE rollouts × `policy_batch_size` from the Pi05 transformer.
For Libero with batch 256 that is 8192 forward passes through the diffusion
suffix per update — likely OOM. Default to **8** and document the trade-off.

### 3.2 Sampling helpers: `src/rl/ogpo/sampling.py`

Two functions, both pure (suitable for `jax.jit`):

**`sample_chain_with_logprob(model, observation, noise, rng, num_steps, noise_level)`**

Reuse Pi05's `sample_actions(..., return_info_dict=True)` exactly as FlowGRPO
does (`flow_grpo/update_actor.py:83-93`). Returns:
- `actions: [B, H, D]` — the final clean action chunk (the `x_0` of the chain),
- `chain: [num_steps+1, B, H, D]` — `x_t` for every step (including the noise
  init); we read this off `outs["x"]` and `outs["x_next"]`,
- `log_prob: [B, H, num_steps]` — per-step Gaussian log-prob (already moved to
  `[..., num_steps]` axis).

Sum across the last `ft_sde_steps` and across the action-horizon axis to get a
scalar `[B]` log-prob, matching how OGPO collapses `lp` across chain & chunk
in `_compute_current_log_probs`.

**`score_chain_under_model(model, observation, chain, rng, num_steps, noise_level)`**

For each SDE step `k = num_steps-1, ..., 0`:
- `time = jnp.full((B,), (k+1)/num_steps)`,
- `x_t = chain[k]`, `sample = chain[k+1]`,
- `dt = -1.0/num_steps`,
- call `model.get_dist_and_log_prob(x_t, sample, time, observation, dt,
  noise_level)`.

Sum the resulting per-step log-probs across the same axes. This is the
"recompute log-prob under current params" step.

We must `jax.lax.scan` this to keep XLA happy and to avoid 10× unrolled
forward passes through PaliGemma.

**Implementation note (critical):** PaliGemma's prefix forward pass is
expensive but identical across the 10 SDE steps. `Pi0.sample_actions` already
caches it (line 360 — `kv_cache`). For the rescoring path, however,
`get_dist_and_log_prob` (line 169) does the *combined* prefix+suffix forward
on each call, which is wasteful. **Recommend** factoring out a thin helper
inside Pi0 (or in `sampling.py` wrapping Pi0 internals) that computes the
prefix once and then runs suffix-only forwards per step. Without this, the
OGPO update cost is roughly 10× the FlowGRPO update cost. This is the single
largest engineering risk and we should validate compile/throughput before
scaling up.

### 3.3 Train step: `src/rl/ogpo/update_actor.py`

Modeled after `flow_grpo/update_actor.py` (the closest analogue — they both
fold critic eval + flow-policy update into one JIT'd function). Pseudocode for
the body of `train_step`:

```python
policy_observation, critic_observation, actions_demo = batch

# 1. Build models from the train states.
policy_model = nnx.merge(policy_state.model_def, policy_state.params)
policy_model.train()

# OLD policy = EMA params if available, else current params with stop_grad.
old_params = policy_state.ema_params if policy_state.ema_params is not None else policy_state.params
old_model = nnx.merge(policy_state.model_def, jax.lax.stop_gradient(old_params))
old_model.eval()

state_action_critic = create_critic(state_action_critic_state, config).eval()
value_critic       = create_critic(value_state, config).eval()

# 2. Expand observations to G copies per sample (group rollouts).
G = config.rl.group_num_samples
expanded_policy_obs = jax.tree.map(lambda x: jnp.repeat(x, G, axis=0), policy_observation)
expanded_critic_obs = jax.tree.map(lambda x: jnp.repeat(x, G, axis=0), critic_observation)

# 3. Sample G chains from the OLD policy with SDE noise (stop_grad).
actions, chain, old_lp = sample_chain_with_logprob(
    old_model, expanded_policy_obs, rng=sample_rng,
    num_steps=config.rl.num_sde_steps, noise_level=config.rl.noise_level,
)  # actions: [B*G, H, D]; chain: [K+1, B*G, H, D]; old_lp: [B*G]
actions = jax.lax.stop_gradient(actions)
chain   = jax.lax.stop_gradient(chain)
old_lp  = jax.lax.stop_gradient(old_lp)

# 4. Compute Q and V on the sampled actions.
q  = summarize_critic_values(
        state_action_critic(expanded_critic_obs, flatten_action_horizon(actions)),
        critic_reduction=config.rl.critic_reduction)
v  = summarize_critic_values(value_critic(expanded_critic_obs),
                             critic_reduction=config.rl.critic_reduction)
adv = q - v   # [B*G]

# 5. Group-relative advantage normalization (OGPO 'vanilla'/'max').
B = batch_size
adv_grouped = adv.reshape(B, G)
if config.rl.adv_strategy == "vanilla":
    baseline = adv_grouped.mean(axis=1, keepdims=True)
elif config.rl.adv_strategy == "max":
    baseline = adv_grouped.max(axis=1, keepdims=True)
else:
    baseline = 0.0
adv_grouped = adv_grouped - baseline
if config.rl.adv_clip_min is not None:
    adv_grouped = jnp.maximum(adv_grouped, config.rl.adv_clip_min)
adv = adv_grouped.reshape(-1)
adv = jax.lax.stop_gradient(adv)

# 6. PPO loss — recompute log_prob of the same chains under the CURRENT model.
def loss_fn(model):
    new_lp = score_chain_under_model(
        model, expanded_policy_obs, chain, rng=score_rng,
        num_steps=config.rl.num_sde_steps, noise_level=config.rl.noise_level,
    )
    ratio = jnp.exp(new_lp - old_lp)
    clipped = jnp.clip(ratio, 1 - config.rl.clip_epsilon, 1 + config.rl.clip_epsilon)
    pg_loss = -jnp.minimum(ratio * adv, clipped * adv).mean()

    bc_loss = jnp.float32(0.0)
    if config.rl.use_bc_regularization:
        # Reuse pi0's own CFM loss on demo actions (no expansion).
        bc_loss = model.compute_loss(bc_rng, policy_observation, actions_demo, train=True).mean()

    total = pg_loss + config.rl.bc_coeff * bc_loss
    info  = {"pg_loss": pg_loss, "bc_loss": bc_loss,
             "ratio_mean": ratio.mean(), "approx_kl": (old_lp - new_lp).mean(),
             "adv_mean": adv.mean()}
    return total, info

(loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True,
                                         argnums=nnx.DiffState(0, config.trainable_filter))(policy_model)
# ... (apply optimizer update, EMA, reset_period — copy from awr_agent's train_step)
```

This deliberately keeps the BC regularization on the **un-expanded** batch
(size B) so we don't pay G× the cost on the BC term — OGPO does the same.

### 3.4 Learner class: `src/rl/ogpo/ogpo_learner.py`

```python
class OGPOAgentLearner(AdvantageWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_step = functools.partial(ogpo_train_step, self._config)
        self._refresh_update_functions()

    @at.typecheck
    def update(self) -> dict:
        # Identical structure to FlowGRPOLearner.update() (src/rl/flow_grpo/...).
        # Only changes vs FlowGRPO:
        #   - require online_ratio >= 1.0 (or implement offline-mix; OGPO is on-policy)
        #   - sub-sample by group_num_samples instead of group_size
        #   - pass mc_return only if config.rl.use_mc_returns
        ...
```

Because we inherit from `AdvantageWeightedSFTLearner`, we automatically get:
- the dual Q/V critic states and their checkpointing
  (`_rl_checkpoint_state`, `_restore_rl_checkpoint`),
- the `_online_batch_to_critic_batch` and `_sft_batch_to_actor_batch` helpers
  that recompute `prefix_embedding` from the current policy each step
  (`advantage_weighted_sft_learner.py:247-309`),
- the AWR normalizer (we will reuse it for logging only; advantages are
  already group-centered).

### 3.5 Experiment driver: `scripts/ogpo_agent/exp.py`

Copy `scripts/awr_agent/exp.py` verbatim and change three things:

1. `from src.rl.ogpo.ogpo_learner import OGPOAgentLearner`
2. `agent = OGPOAgentLearner(...)` instead of `AdvantageWeightedSFTLearner(...)`
3. Drop the `AdvantageWeightedSFTLearnerConfig` assertion at the top of
   `_build_pi0_backbone_critic_defs` and replace with `OGPOSFTLearnerConfig`.

The critic def factory (`_build_pi0_backbone_critic_defs` in
`awr_agent/exp.py:99-181`) is reused as-is — same MLP encoder + ensemble
decoders, same `[state, PREFIX_EMBEDDING]` concatenation.

### 3.6 Launcher: `scripts/ogpo_agent/launcher.py`

Copy `scripts/awr_agent/launcher.py` and change:
- `SCRIPT = "scripts/ogpo_agent/exp.py"`,
- `CONFIG_NAME = "pi05_libero_online_ogpo_sft"`,
- defaults in `applicable_configs`: add `rl.group_num_samples`,
  `rl.num_sde_steps`, `rl.clip_epsilon`, `rl.bc_coeff`.

---

## 4. Important design decisions

### 4.1 Old-policy source: EMA vs. a separate target snapshot

OGPO's `pi05/square.sh` uses `actor_tau=0.005` for a slow target-actor Polyak
average, distinct from `tau=0.05` for the critic. VLA's pi0 train state
already maintains an EMA (`ema_decay=0.999` in `make_base_libero_config`),
which is functionally equivalent to a slow target policy.

**Recommendation:** reuse `policy_state.ema_params` as the "old policy" by
default. Pros: zero new state, already restored across checkpoints. Cons:
EMA decay 0.999 ≈ τ_actor=0.001 is slower than OGPO's 0.005, which means
slightly older `old_lp` values and slightly larger PPO ratios. Document this
clearly; expose `ema_decay` as a per-config override.

### 4.2 Where the SDE chain comes from

OGPO computes `old_lp` *during sampling* (the SDE step distributions are
constructed from `v_t` evaluated under the old actor and yield a closed-form
Gaussian log-prob). The chain itself is stop-gradient'd; only the rescoring
under the *current* params contributes gradient signal. Pi05 already
implements this exact computation in `sample_actions`
(`pi0.py:408-422`), and `outs["log_prob"]` is the per-step Gaussian log-prob
we need. **No new Pi0 code is required for the sampling side.**

For the *rescoring* side we need to compute the same Gaussian per step but
parameterized by the *current* `v_t`. `get_dist_and_log_prob` exists for that,
but it does a full prefix+suffix forward each call. See §3.2 for the
optimization recommendation.

### 4.3 Group-relative advantages vs. true V(s) baseline

OGPO uses both. The vanilla path subtracts `mean(Q(s, a_g))` over the G
samples — a pure on-policy baseline that needs no critic. We *also* have V(s)
from VLA's value head. Recommendation: default to **vanilla group-mean**
(no V-bias) for parity with `square.sh`, and expose `adv_strategy="subtract_v"`
as an option that uses `Q - V` (matches AWR's advantage). The two are not
equivalent — group-mean reduces variance per-state but loses an absolute scale.

### 4.4 BC regularization: which actions?

OGPO's BC term uses demo actions from the offline dataset (or from the
success buffer when one is available). VLA already has SFT batches available
(`batch = next(self._data_iter)` in the inherited update loop). We should use
those — they are the source of demo actions. This means the OGPO actor update
mixes:
- B online states × G samples → PPO loss,
- B offline (SFT) (state, action) pairs → BC loss.

This is exactly what FlowGRPO does for its policy update batch construction.

### 4.5 What to leave out of v1

OGPO contains many variants (FPO, AWR-CFM, chi²-PO, KL-reg, denoiser head,
one-step distillation, MIP-Q, q_warmup, success buffer, calql). All of those
are gated by config flags and the square.sh launcher disables every single
one. For v1, we implement exactly the **vanilla branch** (lines 778-792 of
`ogpo.py`) plus BC regularization. Other modes can be added in follow-ups.

We also leave out:
- OGPO's `error_correct_sde_to_ode` (Theorem 17 score correction during SDE
  sampling). This is a non-trivial change to how `_get_sde_dist` is built;
  defer to v2.
- `clip_bc` / `clip_bc_threshold` / `clip_bc_wrt` — adaptive BC weight
  schedules. Start with constant `bc_coeff`.
- The OGPO BC pre-training phase (`bc_pi_steps=500_000`). VLA boots from a
  pretrained Pi05 checkpoint, so the offline BC phase is already done.

### 4.6 On-policy vs. mixed-batch

OGPO is meant to be ~on-policy: it samples actions from the *current* (well,
target) policy each update, so the replay buffer should mostly hold
recent data. The square.sh launcher uses `offline_ratio=0.0` and
`buffer_size=2M`.

In VLA, the relevant lever is `rl.online_ratio`. The AWR agent runs with
`online_ratio=1.0`. **Set OGPO default to `online_ratio=1.0`** so the PPO
ratios stay valid. Mixed offline data is fine for the BC term (which doesn't
use ratios), but the PPO half must come from on-policy rollouts.

### 4.7 Compile / memory sanity

Three things that will bite us if we are not careful:

- **Group expansion:** `B * G = 256 * 8 = 2048` Pi05 forward passes per update
  for sampling alone. Plus the rescoring loop. We should validate with
  `B=64, G=4` first.
- **Donate buffers:** copy the `donate_argnums` discipline from
  `_update_policy_jitted` in `advantage_weighted_sft_learner.py:147-163`.
- **EMA evaluation:** the "old" model needs `nnx.merge` against EMA params.
  Doing this *inside* the jitted train_step is fine, but capture the
  graphdef once at `__init__` time — don't pass it through the function args.

---

## 5. Step-by-step implementation order

1. **Add `OGPOSFTLearnerConfig`** dataclass + `_CONFIGS` entry. Lint and run
   `tyro` parsing locally to make sure the config loads.
2. **Write `src/rl/ogpo/sampling.py`** with the two helpers. Add a small unit
   test under `tests/` that sanity-checks (a) chain shape, (b) that
   `score_chain_under_model` returns the same value as `sample_chain_with_logprob`
   when called with the *same* params (i.e., `new_lp == old_lp`). This is a
   non-negotiable correctness check.
3. **Write `src/rl/ogpo/update_actor.py`** following the pseudocode in §3.3,
   starting from `flow_grpo/update_actor.py` as a template. Get it to compile
   and run a single update on a tiny batch (B=4, G=2) without sharding.
4. **Write `src/rl/ogpo/ogpo_learner.py`** subclassing
   `AdvantageWeightedSFTLearner`. Reuse all the critic-update plumbing from
   the parent; override only `_train_step` and `update`.
5. **Write `scripts/ogpo_agent/exp.py` and `launcher.py`** as direct copies of
   the awr_agent scripts with the three substitutions in §3.5.
6. **Smoke test**: `./scripts/ogpo_agent/launcher.py --mode local --dry` and
   then a 200-step run.
7. **Performance pass**: profile the sampling+rescoring portion; if it's >2×
   FlowGRPO, implement the prefix-cache optimization for `get_dist_and_log_prob`
   referenced in §3.2.
8. **Parity check vs. OGPO/square**: compare PPO ratio distribution, BC loss
   magnitudes, and Q-value progression against an OGPO run on a state-based
   env. Numbers won't match exactly (different policy class, different env)
   but the shapes should be sane: ratios near 1, KL small, advantages with
   nonzero spread.

## 6. Open questions for the user

- **G (`group_num_samples`).** OGPO uses 32. For Pi05 on Libero, what budget
  per update are we willing to spend? Suggest starting at 8 and benchmarking.
- **`use_mc_returns`.** AWR runs in VLA with `use_mc_returns=True`. Do we
  want OGPO to start there too (advantage = MC − V), or only enable it once
  the critic is trained? OGPO traditionally uses `Q − baseline`, not
  `MC − V`, so the default in v1 should be `use_mc_returns=False`.
- **Prefix-cache optimization.** Do we accept a 10× slower-than-FlowGRPO
  first version, or block on adding a prefix-cached rescoring path inside
  `Pi0`? The latter touches `openpi/`, which may or may not be in scope.

---

## 7. References

- OGPO actor loss (vanilla path): `/users/mananaga/OGPO/ogpo/agents/ogpo.py:778-792`
- OGPO SDE sampling: `/users/mananaga/OGPO/ogpo/agents/modules/pg_helper.py:125`
- OGPO chain log-prob: `/users/mananaga/OGPO/ogpo/agents/modules/pg_helper.py:264`
- OGPO PPO loss: `/users/mananaga/OGPO/ogpo/agents/modules/pg_helper.py:429`
- OGPO BC loss: `/users/mananaga/OGPO/ogpo/agents/ogpo.py:2531`
- VLA AWR learner: `/users/mananaga/vla-post-training/src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py`
- VLA FlowGRPO learner (closest precedent): `/users/mananaga/vla-post-training/src/rl/flow_grpo/flow_grpo_learner.py`
- VLA FlowGRPO actor update: `/users/mananaga/vla-post-training/src/rl/flow_grpo/update_actor.py`
- Pi05 SDE sample_actions: `/users/mananaga/vla-post-training/openpi/src/openpi/models/pi0.py:335`
- Pi05 chain rescoring: `/users/mananaga/vla-post-training/openpi/src/openpi/models/pi0.py:169`
- VLA online config registration: `/users/mananaga/vla-post-training/src/training/config.py:430-480`
- AWR launcher (template): `/users/mananaga/vla-post-training/scripts/awr_agent/launcher.py`
- OGPO `square.sh` launcher (canonical hyper-params): `/users/mananaga/OGPO/scripts/ogpo/square.sh`
