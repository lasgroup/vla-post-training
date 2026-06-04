# Implementation Doc: Distributional-RL (Categorical) Critic for Best-of-N

**Status:** Proposed
**Owner:** TBD
**Config flag:** `rl.critic.use_distributional_critic: True/False` (set in `scripts/configs/best_of_n.yaml`)

---

## 1. Goal

Borrow the **categorical critic trained via distributional RL** from
`/users/mananaga/BiggerRegularizedCategorical` (BRC, a C51-style categorical
distributional critic) and make it available in the Best-of-N pipeline of
`vla-post-training`. The feature must be toggleable from
`scripts/configs/best_of_n.yaml` with a single flag,
`use_distributional_critic`, with `False` reproducing today's behavior exactly.

**Scope locked for this implementation:**
- Import the **categorical distributional-RL logic only** — **no BroNet**
  architecture. The existing MLP critic heads are kept unchanged.
- The categorical critic applies to **both the Q and V networks**.
- Both the **MC loss and the TD loss** flow through the categorical critic.

---

## 2. TL;DR of the design

There are **two distinct things** people mean by "categorical critic," and the
distinction is exactly why the flag is named `use_distributional_critic` and not
`use_categorical_critic`:

| | What it is | Where it lives today |
|---|---|---|
| **Categorical *representation*** | Network outputs logits over value bins instead of a scalar. Controlled by `num_value_bins`. | **Already implemented** (`num_value_bins > 1` + `CategoricalValueDistribution`, `src/rl/value_distribution.py`). |
| **Distributional *training target* (C51)** | The TD target is itself a **full distribution**: take the next-state value *distribution*, apply the Bellman operator atom-by-atom (`T z_i = r + γ z_i`), project back onto the support, and minimize cross-entropy against the predicted distribution. | **This is what BRC does** (`jaxrl/agent/update.py::update_critic`). **Not yet in this repo** — this is what we add. |

`num_value_bins` is the **representation** axis; `use_distributional_critic` is
the **training-target** axis. They are orthogonal, which is why a single
"categorical" name would be ambiguous.

What `use_distributional_critic: True` changes, concretely:
- **TD branch (new code):** the scalar target `r + γV(s')` is replaced by a
  **projected distributional target** (C51 projection of the next-state value
  distribution). Applied to **both** the Q-step and the V-step.
- **MC branch (no new code):** the MC return is a scalar sample; it is already
  cross-entropied into the categorical support by
  `CategoricalValueDistribution.log_prob(scalar)` (two-hot/one-hot). It "becomes
  categorical" purely by virtue of `num_value_bins > 1`. No change needed.
- **Inference (no change):** Best-of-N selection already scores with
  `make_value_distribution(...).mean()`.

Net change surface is small and localized to `src/rl/best_of_n/update_critic.py`
+ a projection helper in `src/rl/value_distribution.py` + two config fields.

---

## 3. Background: the two codebases

### 3.1 vla-post-training — current Best-of-N critic

- **Entry / dispatch:** `scripts/exp.py:55` → `BestofNLearner`.
- **Config:** `pi05_libero_online_best_of_n` registered in
  `src/training/config.py:537` via `make_base_libero_config(..., BestofNLearnerConfig())`.
- **Critic config dataclass:** `CriticTrainingConfig`
  (`src/training/config.py:123-147`). Relevant fields:
  `encoder_hidden_dims=(512,512)`, `decoder_hidden_dims=(256,256)`,
  `num_qs=2`, `num_vs=2`, `reduction="min"`, `ema_decay=0.995`,
  `td_weight_schedule`, `num_value_bins=1`, `value_lower_bound/upper_bound`,
  `value_target_type="one_hot"`, `inference_start_step`, `pre_training_steps`.
- **Networks:** `src/rl/best_of_n/update_critic.py:46` `_build_pi0_backbone_critic_defs`
  builds an `MLPEncoder` (concatenates `["prefix_embedding","state"]`) feeding
  `StateActionEnsembleDecoder` / `StateValueEnsembleDecoder`. Each head is an
  `MLP` (Dense → LayerNorm → ReLU) ending in `out_dim = max(1, num_bins)`.
  (`src/rl/networks/mlp.py`, `src/rl/networks/decoders/values/state_action_value.py`.)
  **These heads are reused as-is — no architecture change.**
- **Two critics, decoupled (not SAC):**
  - `Q(s,a)` bootstraps off **`V(s')`**: `td_target = r + γ·V(s')`
    (`update_critic.py:336-341`).
  - `V(s)` regresses **`Q(s,a)`** on the batch action: `td_target = Q(s,a)`
    (`update_critic.py:411-419`).
  - Both blend an **MC loss** and a **TD loss** with
    `td_weight = td_weight_schedule(step)` (MC→TD over training).
  - **No entropy / temperature term** — the policy is the frozen flow model and
    Best-of-N picks the argmax; there is no learned actor in the backup.
- **Value distribution:** `src/rl/value_distribution.py`.
  - `num_value_bins == 1` → `GaussianValueDistribution` (MSE-equivalent).
  - `num_value_bins > 1` → `CategoricalValueDistribution` with
    `log_prob(scalar_target)` using `one_hot`/`two_hot` encoding, and
    `mean() = Σ softmax(logits)·bin_centers`.
- **Support bounds:** auto-resolved in `OnlineTrainConfig.__post_init__`
  (`src/training/config.py:351-383`) from reward type + discount, with a
  half-bin-width expansion when `num_bins > 1`.
- **Inference (Best-of-N selection):** `best_of_n_learner.py::sample_actions`
  (~`:295-457`): tile obs ×`n_samples`, sample N action chunks from the policy,
  compute the prefix embedding **once per env**, score each candidate with
  `Q`, reduce the ensemble, `argmax` over the N samples. Scoring uses
  `make_value_distribution(q_logits, ...).mean()` (`best_of_n_learner.py:433`).
- **Update dispatch:** `best_of_n_learner.py:531 update()` →
  `self._update_critics_jitted(...)` (jitted at `:119`, config captured as a
  **static closure** → the `use_distributional_critic` branch is resolved at
  trace time, no dynamic control-flow cost).

### 3.2 BRC — categorical distributional critic (the thing we're borrowing)

- **Framework:** JAX/**Flax (`flax.linen`)** — note: this repo is **`flax.nnx`**,
  so we port the *logic*, not code verbatim.
- **Support:** `num_bins=101`, `v_max=10.0`, `v_min=-v_max`, atoms
  `linspace(v_min, v_max, num_bins)`, `delta_z = (v_max−v_min)/(num_bins−1)`.
- **Distributional Bellman backup (`jaxrl/agent/update.py::update_critic`):**
  ```python
  next_q_probs = softmax(target_critic(s', a')).mean(axis=0)          # reduce ensemble → [B, atoms]
  Tz = r + γ·mask·(bin_values − temp·next_log_probs)                  # SAC entropy term (we drop this)
  Tz = clip(Tz, v_min, v_max)
  b  = (Tz − v_min) / delta_z;  l = floor(b);  u = ceil(b)
  target_probs = Σ_atoms next_q_probs · (interp mass split between l and u)   # C51 projection
  target_probs = stop_gradient(target_probs)
  loss = −Σ target_probs · log_softmax(q_logits)                      # cross-entropy
  ```
- **Scalar extraction:** `Q = Σ bin_values · softmax(logits)` (expectation).
- **Target net:** soft update `tau=0.005` (≈ this repo's `ema_decay=0.995`).

**What we keep vs. drop from BRC:**
- **Keep:** the categorical projection (C51), expectation-based scalar
  extraction, the cross-entropy distributional loss.
- **Drop:** the **BroNet architecture** (out of scope — we keep the existing MLP
  heads), the SAC entropy term `temp·next_log_probs` (no learned actor /
  temperature in Best-of-N), the actor/temperature updates, BRC's replay
  buffer/env scaffolding.

---

## 4. Central design decisions

### D1 — What does `use_distributional_critic` toggle?

**Decision:** It enables the **C51 distributional Bellman backup** for the **TD
target**, reusing the existing bins/support/distribution code. It is **not** an
alias for `num_value_bins > 1` (that already exists and is a scalar-target
two-hot regression).

**Requires a categorical representation.** Validation in
`OnlineTrainConfig.__post_init__`: if `use_distributional_critic and
num_value_bins <= 1`, **auto-set `num_value_bins = 51`** and log a warning, so
the flag "just works" from YAML without separately remembering to bump the bins.

### D2 — Network architecture: keep the existing MLP heads (no BroNet)

**Decision (locked):** **No architecture change.** The existing
`StateActionEnsembleDecoder` / `StateValueEnsembleDecoder` MLP heads
(Dense→LayerNorm→ReLU, output width `max(1, num_value_bins)`) are reused exactly.
We import **only** the categorical distributional-RL logic from BRC, not
BroNet. No new network file, no `critic_arch` config field.

**Rationale:** Isolates the change to the *loss/target*, making the effect of
distributional training cleanly attributable and minimizing risk. BroNet can be
revisited later as an independent follow-up if scaling the critic is desired.

### D3 — Ensemble reduction for the distributional *target*

The existing scalar path reduces the ensemble with `reduction="min"`
(conservative double-Q; `summarize_critic_values`). For the distributional
target we must decide how to combine the ensemble's *distributions*.

**What BRC actually does:** **mean-of-probs**, no `min` at all
(`jaxrl/agent/update.py:39`):
```python
next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)   # average over ensemble axis
```
This is deliberate and is the paper's thesis: **LayerNorm regularization on the
critic substitutes for the conservative `min`** (clipped double-Q / REDQ) used to
fight value overestimation. BRC averages a small ensemble and leans on
regularization instead of pessimism.

**Decision (locked): mean-of-probs, faithful to BRC.** Average the ensemble's
softmax probabilities into one target distribution; no `min`. Our reused MLP
heads already carry LayerNorm (`state_action_value.py:35`,
`use_layer_norm=True`), which is the regularization BRC relies on.

**Recorded as a config variable so the decision is explicit and reversible:**
```python
distributional_target_reduction: str = "mean"   # "mean" = mean-of-probs (BRC); "min" = min-member selection
```
- `"mean"` (default): `target = mean_over_members(softmax(logits))` — BRC.
- `"min"` (documented fallback): use per-member scalar means **only to select**
  the most pessimistic member (`argmin`), then keep **that member's full
  distribution** (not its scalar) as the target. Note: collapsing to the scalar
  `min` and re-encoding as two-hot is **not** an option — that silently reverts
  to the existing HL-Gauss regression and discards the distributional bootstrap.

**Why expose it:** we are porting BRC's loss but **not** its full regularization
stack (no BroNet, no tuned weight decay), so if value overestimation appears
under Best-of-N's argmax (which actively exploits optimistic critic error),
`"min"` is the escape hatch — without re-plumbing code.

### D4 — Bootstrap source (keep the decoupled Q/V structure), applied to both nets

This repo is decoupled: `Q` bootstraps off **`V(s')`**, and `V` regresses
**`Q(s,a)`**. We keep this and apply the categorical projection to **both**:

- **Q-step TD target:** distribution = project(`V(s')` distribution,
  `Tz = r + γ·z`). `V(s')` already outputs a categorical distribution over the
  same support.
- **V-step TD target:** distribution = reduced `Q(s,a)` distribution
  **directly** — `V` regresses `Q` with no reward and no discount, so the
  Bellman map is the **identity** and **no projection is needed**
  (`target_probs = reduce_ensemble_probs(softmax(Q(s,a)))`).

**Rationale:** Re-architecting to single-critic SAC is out of scope and would
break the rest of the Best-of-N learner. The projection composes naturally with
the existing two-network setup, and the user requested the categorical critic on
both networks.

### D5 — MC loss and TD loss both flow through the categorical critic

**Decision:** Keep the MC/TD blend; both terms are cross-entropy over the same
categorical support, so they combine cleanly:
`loss = td_weight·CE(td_target) + (1−td_weight)·CE(mc_target)`.

- **TD term:** the new **distributional** projected target (D4).
- **MC term:** the **scalar** `mc_return`, two-hot/one-hot encoded into the
  support — this is **existing behavior** of `log_prob(scalar)` and needs **no
  new code** beyond `num_value_bins > 1`. (`value_target_type` continues to
  govern one-hot vs. two-hot encoding of the MC scalar.)

**Rationale:** The MC return is a single Monte-Carlo sample, not a distribution;
its categorical form is the two-hot encoding. The distributional backup is
inherently a property of the *bootstrapped* (TD) target. Both still train the
categorical critic, satisfying "apply the categorical Q-function to both MC/TD
loss."

### D6 — No entropy term

**Decision:** Drop BRC's `temp·next_log_probs`. The Bellman map is `Tz = r + γz`.
No learned actor/temperature exists in Best-of-N.

### D7 — Support / bins (how the binning range is decided)

**Decision:** Reuse the **existing** auto-resolved support. Do **not** hardcode
BRC's symmetric `[-10, 10]`; this repo's returns are bounded, and the existing
logic already sizes the support from the reward type, discount, and horizon.

**How it works today** (`OnlineTrainConfig.__post_init__`,
`src/training/config.py:351-383`; bin centers =
`linspace(lower, upper, num_value_bins)` in `value_distribution.py::_bin_centers`):
1. **Manual override:** if both `value_lower_bound` and `value_upper_bound` are
   set, use them verbatim (early return).
2. **Auto-compute** otherwise:
   - sparse success reward → `[0.0, 1.0]`;
   - time-to-success reward → `[-(1 − γ^T)/(1 − γ), 0.0]`, where
     `T = collect.max_episode_steps`. (For the current `best_of_n.yaml`:
     `γ=0.995`, `T=400` ⇒ `lower ≈ −173`.)
3. **Half-bin-width pad** (when `num_value_bins > 1`): expand both ends by
   `half_bw = (upper − lower) / (2·(num_value_bins − 1))` so the extreme
   achievable returns land on the outermost bin centers, not at the edge.

The support therefore tracks `discount` / `max_episode_steps` / reward-type
automatically — the only knob to set is **`num_value_bins`** (recommend **51** to
start; BRC uses 101). Bounds may still be pinned manually via
`value_lower_bound` / `value_upper_bound` if a run needs a fixed support.

### D8 — Inference path

**Decision:** **No change.** Best-of-N selection scores candidates with
`make_value_distribution(q_logits, num_value_bins, ...).mean()`
(`best_of_n_learner.py:433`), which already returns the categorical expectation.

---

## 5. Implementation plan (file by file)

### 5.1 Config — `src/training/config.py`

Add to `CriticTrainingConfig` (`:123`):
```python
use_distributional_critic: bool = False        # True → C51 distributional Bellman backup (TD target)
distributional_target_reduction: str = "mean"  # "mean" = mean-of-probs (BRC); "min" = min-member selection (D3)
```
In `OnlineTrainConfig.__post_init__` (`:351`), fold into the existing critic
block: if `use_distributional_critic` and `num_value_bins <= 1`, set
`num_value_bins = 51` (with a warning) **before** the half-bin-width support
expansion, so bounds are sized for the final bin count. (The cleanest patch
computes `num_bins` once at the top of the block and reuses it for both the
bounds math and the final `dataclasses.replace`.)

> No `critic_arch` / BroNet fields — architecture is unchanged (D2).

### 5.2 Categorical projection helper — `src/rl/value_distribution.py`

Add a function that builds a **projected target distribution** (probs over bins):
```python
def categorical_project(
    next_probs,        # [b, k] reduced next-state value distribution
    reward,            # [b]
    discount,          # [b]  (γ·mask, 0 at terminal)
    bin_centers,       # [k]
):
    # Tz_j = reward + discount * bin_centers_j ; clip to [c0, c_{k-1}]
    # frac = (Tz - c0)/Δ ; l = floor(frac), u = ceil(frac)
    # distribute next_probs[:, j] onto bins l, u by linear interpolation
    # return target_probs [b, k]   (caller wraps in stop_gradient)
```
This is BRC's C51 projection, minus the entropy term, using **this repo's**
`bin_centers` (asymmetric, auto-sized). Also add a small helper
`reduce_ensemble_probs(logits, mode)` that reduces an ensemble of logits
`[num_heads, b, k]` → a single `[b, k]` target distribution per **D3**:
`mode="mean"` → `softmax(logits).mean(axis=0)` (BRC default); `mode="min"` →
gather the `argmin`-by-mean member's `softmax`.

Optionally add `CategoricalValueDistribution.cross_entropy(target_probs)`
(`Σ target_probs · log_softmax(logits)`) so the loss reads symmetrically with the
existing `log_prob(scalar)`.

### 5.3 Loss — `src/rl/best_of_n/update_critic.py`

**Q-step** (`train_q_step`, `:303`) — branch on the flag for the **TD** target;
MC term unchanged:
```python
red = config.rl.critic.distributional_target_reduction  # default "mean" (BRC)
if config.rl.critic.use_distributional_critic:
    next_logits  = target_value_model(next_observation)          # [num_vs, b, k]
    next_probs   = reduce_ensemble_probs(next_logits, red)       # D3 → [b, k]
    target_probs = stop_gradient(categorical_project(
        next_probs, reward, discount, bin_centers))
    td_loss = -mean(sum(target_probs * log_softmax(q_logits_per_head), -1))  # over heads + batch
else:
    td_targets = reward + discount * stop_gradient(bootstrapped_values)      # existing scalar path
    td_loss = -mean(q_dist.log_prob(td_targets))
mc_loss = -mean(q_dist.log_prob(mc_return))                       # unchanged (scalar two-hot)
loss = td_weight*td_loss + (1-td_weight)*mc_loss
```
**V-step** (`train_value_step`, `:378`) — categorical V-target is the reduced
`Q(s,a)` distribution **directly** (D4, identity map, no projection); MC
unchanged:
```python
if config.rl.critic.use_distributional_critic:
    q_logits_sa  = target_q_model(observation, actions)          # [num_qs, b, k]
    target_probs = stop_gradient(reduce_ensemble_probs(q_logits_sa, red))   # red = distributional_target_reduction
    td_loss = -mean(sum(target_probs * log_softmax(value_logits_per_head), -1))
else:
    ... existing scalar path ...
mc_loss = -mean(v_dist.log_prob(mc_return))                      # unchanged
loss = td_weight*td_loss + (1-td_weight)*mc_loss
```
`bin_centers` come from `get_value_bounds(config)` + `num_value_bins` (same
inputs `make_value_distribution` already uses).

### 5.4 Inference — no change
`best_of_n_learner.py::sample_actions` already scores with `dist.mean()` (D8).

---

## 6. Config wiring (`scripts/configs/best_of_n.yaml`)

`use_distributional_critic` lives on the nested critic config, so the launcher's
tyro dotted-flag convention (`launcher.py::flag_to_cli_tokens`) maps the YAML key
`rl.critic.use_distributional_critic` → `--rl.critic.use_distributional_critic`.
Add under `params:`:
```yaml
  rl.critic.use_distributional_critic: [True]        # the toggle
  rl.critic.num_value_bins: [51]                     # support resolution (auto-defaults to 51 if omitted)
  rl.critic.distributional_target_reduction: [mean]  # "mean" = BRC-faithful; "min" = conservative fallback
```
Setting `rl.critic.use_distributional_critic: [False]` (or omitting it)
reproduces today's behavior bit-for-bit. To sweep the ablation, pass a list:
`rl.critic.use_distributional_critic: [True, False]`.

> If you want a bare `use_distributional_critic:` key (no `rl.critic.` prefix) in
> YAML, add a top-level convenience field on `OnlineTrainConfig` that fans out
> into `rl.critic` in `__post_init__`. Recommended to **not** do this — keep the
> flag on the critic config where every other critic knob already lives.

---

## 7. Testing & validation

1. **Parity / no-regression:** `use_distributional_critic=False` must produce
   identical critic updates to `main` (golden-value test on `train_q_step` /
   `train_value_step`).
2. **Projection unit test:** `categorical_project` conserves mass
   (`sum(target_probs)==1`), is exact for on-support targets, and reduces to a
   two-hot when `next_probs` is a delta. Cross-check against a slow reference.
3. **Degenerate-support test:** with `num_value_bins=2`, the projected target
   matches the analytic two-hot.
4. **Sanity training run:** short `pi05_libero_online_best_of_n` run with
   `use_distributional_critic=True`; confirm `critic/q_td_loss` decreases,
   `critic/q_value_mean` stays within `[value_lower_bound, value_upper_bound]`,
   and Best-of-N selection still runs after `inference_start_step`.
5. **Ablation:** `use_distributional_critic ∈ {True, False}` with everything else
   fixed → isolates the effect of distributional training.

---

## 8. Risks & open questions

- **Support mismatch under time-to-success reward (D7):** returns are in
  `[lower, 0]`; verify `categorical_project` clipping behaves at the boundary
  bins (the half-bin-width expansion should prevent edge pile-up — confirm).
- **Min-member reduction cost (D3):** selecting per-sample min-member
  distributions adds a gather; negligible at these batch sizes and stays inside
  the jitted region (`reduction` is static config).
- **EMA vs. soft target:** repo uses `ema_decay=0.995` EMA params as the target
  (`use_ema=True`), equivalent to BRC's `tau=0.005`. The target distribution is
  computed from **EMA** params via `create_critic` — unchanged.
- **`num_value_bins` interaction with MC two-hot:** the MC branch two-hots
  scalars into the same support; `value_target_type` (`one_hot`/`two_hot`) keeps
  governing the **MC** term only — the TD term is now full-distribution.
- **51 vs 101 bins:** BRC uses 101 over a wide symmetric support; our support is
  narrow and bounded, so 51 is likely plenty. Treat as a tunable.
- **Terminal handling (correctness, not a knob):** the per-transition
  `discount` field already encodes `γ·mask` (`0` at terminal), so
  `categorical_project` must take the **per-sample `discount` vector `[b]`** —
  at a terminal it collapses `Tz = reward` → a two-hot at `reward` (e.g. value
  `0` at a success terminal). No separate mask code is added.
- **Shared-support guard (correctness):** the V-target = Q-distribution-directly
  trick (D4) is valid **only** if Q and V share identical `num_value_bins` and
  bounds. Add a one-line assertion at critic-build time; both already read the
  same config, but the distributional path now depends on the invariant.
- **Out of scope:** BroNet architecture, BRC's SAC actor, temperature
  autotuning, and its replay buffer/normalizer. We borrow only the categorical
  distributional-RL logic.

---

## 9. Deferred ablations (revisit after the core lands)

These were considered and **intentionally deferred** — none block the core
implementation, and all but one are reachable through existing config (no
structural change), so they can be swept as ablations once the distributional
path is verified. Defaults below = current behavior.

| # | Ablation | Lever | Default for now | Notes |
|---|----------|-------|-----------------|-------|
| 1 | MC target encoding | `rl.critic.value_target_type` (`one_hot`/`two_hot`) | `one_hot` (existing) | `two_hot` likely better with bins (matches the TD projection); pure config. |
| 2 | Inference scoring reduction | hardcoded `min` over ensemble (`best_of_n_learner.py:437`) | `min` | **Only item needing a small code touch** to make configurable; deliberately distinct from the *target* reduction (D3 = `mean`). |
| 3 | Critic warmup before selection | `rl.critic.inference_start_step`, `pre_training_steps` | `0` / `0` | A fresh categorical critic is ~uniform → early selection is noisy; a non-zero warmup may help. Pure config. |
| 4 | MC/TD blend | `rl.critic.td_weight_schedule` | constant `0.5` | MC term is a (non-distributional) two-hot anchor; may want to schedule toward more TD. Pure config. |
| 5 | Support resolution | `rl.critic.num_value_bins` | `51` | BRC uses 101; our support `≈[−173, 0]` is wide, so finer bins may help near 0. Pure config. |
| 6 | Distributional diagnostics | logging | — | Cheap to add during impl: target `q_min/q_max/q_mean` (BRC `update.py:63-65`) + predicted-distribution entropy (collapse = red flag). |

## 10. Key references (file:line)

**vla-post-training**
- Critic config: `src/training/config.py:123-147`; bounds resolution `:351-383`;
  config registration `:537`.
- Critic build (MLP heads, reused as-is): `src/rl/best_of_n/update_critic.py:46-111`.
- Q/V losses (edit sites): `src/rl/best_of_n/update_critic.py:303-375` (Q),
  `:378-438` (V); ensemble reduction `:142-162`.
- Value distribution (new projection helper goes here):
  `src/rl/value_distribution.py` (Categorical `:65-113`, factory `:125-152`).
- Best-of-N selection (no change): `src/rl/best_of_n/best_of_n_learner.py:295-457`
  (score `:433`); update dispatch `:531`, jit `:119`.
- Launcher flag mapping: `scripts/launcher.py:149-167`.

**BRC**
- Distributional backup + C51 projection + CE loss (the logic we port):
  `jaxrl/agent/update.py:33-71`.
- Scalar extraction (expectation): `jaxrl/agent/update.py:20-22`.
- Hyperparameters: `jaxrl/agent/brc_learner.py:122-142`.
