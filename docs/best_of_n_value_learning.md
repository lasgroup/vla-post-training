# Best-of-N Value Learning — Implementation Cheatsheet

## Overview

Best-of-N (BoN) uses a **Q-function and a V-function** to rerank candidate actions at inference time. Both critics are trained from collected rollout data (and optionally offline data). The pi0.5 policy is **frozen by default**; set `train_pi0_prefix_encoder=True` to also finetune the Pi0 VLM backbone jointly with the critics.

```
Policy (pi0.5) ──► sample N actions ──► Q-critic ranks them ──► execute best action
                         │
                 store in replay buffer
                         │
                 train Q + V critics
                 (optionally: backprop into Pi0 backbone via train_pi0_prefix_encoder)
```

---

## Architecture

### Observation Encoder

Two options, selected by `critic_encoder_type`:

#### `pi0_prefix` (default)
Concatenates pi0 prefix embedding with proprioceptive state, passes through a shared MLP.

```
pi0 prefix embedding  (2048-dim, mean-pooled tokens)         ┐
                                                               ├─► concat ──► MLP(512, 512, ReLU) ──► 512-dim
proprioceptive state  (7-dim for LIBERO, zero-padded to 32)   ┘
```

- Prefix embedding is **recomputed from observations every update** (not stored in the buffer)
- **Frozen** (default): uses EMA policy params (`ema_params` if available)
- **Trainable** (`train_pi0_prefix_encoder=True`): critic loss backpropagates through the full Pi0 VLM backbone; encoder uses its own AdamW optimizer (`pi0_encoder_lr_schedule`)

#### `resnet`
Encodes both camera images with a ResNet-18 (spatial softmax pooling), concatenates with state, passes through MLP.

```
base camera image  (224×224×3 uint8) ──► ResNet-18 + spatial softmax ──► img_features ┐
wrist camera image (224×224×3 uint8) ──► ResNet-18 + spatial softmax ──► img_features ┘──► concat + state ──► MLP ──► embedding
```

---

### Action Encoder (optional)

Controlled by `critic_action_compress_dim` (default: `None` = disabled).

```
flat action chunk (320-dim = 10 steps × 32-dim) ──► MLP(128, 32, ReLU) ──► 32-dim
```

Default architecture when enabled: **(128, 32)** two-layer MLP with final ReLU (`activate_final=True`).

Without compression the raw 320-dim chunk is used directly.

---

### Q-Function (`StateActionCritic`)

```
obs_embedding (512-dim)     ┐
                            ├─► concat ──► Ensemble of N_Q independent MLPs(256, 256) + LayerNorm ──► scalar (or K logits)
action (32-dim or 320-dim)  ┘
```

- **Ensemble size**: `critic_num_qs = 2` (default)
- **Output**: `(num_qs, batch)` for regression or `(num_qs, batch, K)` for categorical
- Implemented via `nnx.vmap` over `StateActionValueDecoder` instances

### V-Function (`StateValue`)

```
obs_embedding (512-dim) ──► Ensemble of N_V independent MLPs(256, 256) + LayerNorm ──► scalar (or K logits)
```

- **Ensemble size**: `critic_num_vs = 2` (default)
- Same architecture as Q minus the action input

---

### EMA Target Networks

Both Q and V use **EMA parameters** as targets (enabled by `use_ema_critic = True`).

```
EMA update (per gradient step):  θ_ema ← 0.995 · θ_ema + 0.005 · θ
```

- `critic_ema_decay = 0.995`
- EMA params are initialized to the live params at network creation

---

## Loss Functions

### Reward signal

Controlled by `use_time_to_success_as_reward`:
- `True` (default): reward = `−(steps_to_success)`, episode-level sparse negative time. MC return ∈ `[−(1−γ^T)/(1−γ), 0]`
- `False`: reward = 1 on success, 0 otherwise. MC return ∈ `[0, 1]`

---

### Q-Function Loss

Uses **EMA V** as the next-state value for bootstrapping.

**TD target:**
$$y_{\text{TD}} = r + \gamma \cdot \min_{j} V_{\theta_j^{\text{EMA}}}(s')$$

**Combined loss** (weighted sum of TD and MC):
$$\mathcal{L}_Q = w_{\text{TD}} \cdot \mathcal{L}_{\text{TD}} + (1 - w_{\text{TD}}) \cdot \mathcal{L}_{\text{MC}}$$

where for **regression** (`num_value_bins=1`):
$$\mathcal{L}_{\text{TD}} = \mathbb{E}\left[(Q_\theta(s,a) - y_{\text{TD}})^2\right], \quad \mathcal{L}_{\text{MC}} = \mathbb{E}\left[(Q_\theta(s,a) - G)^2\right]$$

and for **classification** (`num_value_bins > 1`), replace MSE with cross-entropy over K bins:
$$\mathcal{L}_{\text{TD}} = -\mathbb{E}\left[\log p_\theta(y_{\text{TD}} \mid s, a)\right], \quad \mathcal{L}_{\text{MC}} = -\mathbb{E}\left[\log p_\theta(G \mid s, a)\right]$$

Ensemble reduction: **min** across heads (`critic_reduction = "min"`).

---

### V-Function Loss

Uses **EMA Q** for the TD target (V learns to match the current-action Q value).

**TD target:**
$$y_{\text{TD}}^V = \min_{j} Q_{\theta_j^{\text{EMA}}}(s, a)$$

**Combined loss:**
$$\mathcal{L}_V = w_{\text{TD}} \cdot \mathcal{L}_{\text{TD}} + (1 - w_{\text{TD}}) \cdot \mathcal{L}_{\text{MC}}$$

where:
$$\mathcal{L}_{\text{TD}} = \mathbb{E}\left[(V_\phi(s) - y_{\text{TD}}^V)^2\right], \quad \mathcal{L}_{\text{MC}} = \mathbb{E}\left[(V_\phi(s) - G)^2\right]$$

(regression) or cross-entropy (classification), same as Q.

---

### TD Weight Schedule (`td_weight_schedule`)

`StepSchedule(init_value, end_value, switch_step)`:
- Before `switch_step`: weight = `init_value`
- From `switch_step` onward: weight = `end_value`

**Common settings:**
| Goal | `init_value` | `end_value` | `switch_step` |
|------|-------------|-------------|---------------|
| Pure MC | 0.0 | 0.0 | any |
| Pure TD (always) | 1.0 | 1.0 | any |
| Pure TD (via immediate switch) | 0.0 | 1.0 | 0 |
| Start MC → switch to TD | 0.0 | 1.0 | e.g. 900 |
| Constant 50/50 | 0.5 | 0.5 | any |

---

### Output Distribution

Controlled by `num_value_bins`:

| `num_value_bins` | Distribution | Loss | `value_target_type` matters? |
|---|---|---|---|
| 1 | `GaussianValueDistribution` | MSE-equivalent (`-||ŷ - y||²`) | No |
| >1 | `CategoricalValueDistribution` | Cross-entropy over K bins | Yes |

**Target encoding** for categorical (`value_target_type`):
- `"one_hot"`: hard assignment to nearest bin — prone to saturation
- `"two_hot"`: linear interpolation between two bracketing bins — smoother gradients

**Value bounds** are auto-computed from reward type + `discount` + `max_episode_steps` (400 for LIBERO-90), with half-bin-width padding when `num_value_bins > 1`.

---

## Training Loop

### UTD (Updates-to-Data ratio)

```
num_critic_updates_per_batch = 10  (default)
critic_update_interval = 1         (update every step)
```

→ **10 critic gradient steps per data collection step** (effectively UTD=10).

Each `_update_critics` call runs the inner loop:
```python
for _ in range(num_critic_updates_per_batch):
    q_state, q_info  = train_q_step(q_state, value_state, batch)
    value_state, v_info = train_value_step(value_state, q_state, batch)
```

Q and V are updated **alternately** within the same batch.

---

### Online Training Phase

```
update() called every step
  ├─ skip if step < critic_training_start_step (default: 0)
  ├─ skip if online buffer too small
  ├─ sample batch from replay buffer
  ├─ recompute prefix embeddings on-the-fly (pi0_prefix encoder)
  └─ run N critic update steps
```

**No policy updates by default** — pi0.5 is frozen. With `train_pi0_prefix_encoder=True`, critic gradients are backpropagated through the Pi0 VLM backbone (but the action decoder / flow matching head remains unchanged).

---

### Off-Policy / Replay Buffer

- Fully **off-policy**: both online-collected and optionally offline data
- `online_ratio` controls the mix: fraction of batch from online buffer (default 1.0 = all online)
- `buffer_capacity = 250,000` transitions (default)
- Buffer stores normalized actions (7-dim LIBERO → normalized, then zero-padded to 32-dim)
- MC returns are computed and stored at episode end

---

### Offline Critic Pretraining

Runs **before** online training, for `num_offline_pretraining_steps` steps (default: 0 = disabled).

```python
for pretrain_step in range(num_offline_pretraining_steps):
    batch = offline_buffer.sample(batch_size)
    q_state, value_state = update_critics(batch)
    log to wandb under "pretrain/*" x-axis (separate from online step counter)
```

- Uses the **same TD/MC mixed loss** as online training (controlled by `td_weight_schedule`)
- Offline buffer path: set via `rl.offline_buffer_load_paths`
- W&B logs: `pretrain/*` metrics use `pretrain_step` as x-axis; online training always starts at step 0

**Optimizer reset at transition**: at `training_steps == critic_pre_training_steps` (set to 0 when `td_weight_schedule.switch_step != -1`), both optimizer states are discarded and EMA params are re-initialized from live params. This prevents stale momentum from influencing online training.

---

## Inference (Best-of-N Action Selection)

Enabled from `critic_inference_start_step` (default: 900). Before that, falls back to standard pi0.5 sampling.

```
For each env at inference:
  1. Sample N=32 candidate actions from pi0.5 policy
  2. Build critic observation (normalize state, compute prefix embedding or tile images)
  3. Score all N candidates: Q(s, a_i) for i=1..N
  4. Ensemble reduction: min across Q heads → scalar score
  5. Select argmax_i Q(s, a_i)
```

- Uses **EMA Q params** for scoring (`ema_params` if available)
- Actions are normalized and zero-padded before passing to Q

---

## Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | 32 | Candidate actions sampled per step |
| `critic_encoder_type` | `"pi0_prefix"` | Observation encoder: `pi0_prefix` or `resnet` |
| `critic_action_compress_dim` | `None` | Compress 320-dim action to N-dim before Q input |
| `critic_encoder_hidden_dims` | `(512, 512)` | Obs encoder MLP hidden layers |
| `critic_decoder_hidden_dims` | `(256, 256)` | Q/V decoder MLP hidden layers |
| `critic_num_qs` | 2 | Q ensemble size |
| `critic_num_vs` | 2 | V ensemble size |
| `use_ema_critic` | `True` | Use EMA target networks for Q and V |
| `critic_ema_decay` | 0.995 | EMA decay rate |
| `num_critic_updates_per_batch` | 1 (launcher: 10) | UTD multiplier |
| `critic_training_start_step` | 0 | Step to begin critic training |
| `critic_inference_start_step` | 900 | Step to enable BoN action selection |
| `td_weight_schedule` | `StepSchedule(0.0, 1.0, 1000)` | TD vs. MC loss weighting |
| `num_value_bins` | 1 | 1=regression, >1=categorical |
| `value_target_type` | `"one_hot"` | `one_hot` or `two_hot` (categorical only) |
| `train_pi0_prefix_encoder` | `False` | Backprop critic loss into Pi0 VLM backbone |
| `pi0_encoder_lr_schedule` | `ConstantSchedule(1e-5)` | LR for Pi0 backbone when `train_pi0_prefix_encoder=True` |
| `online_ratio` | 1.0 | Fraction of batch from online buffer |
| `num_offline_pretraining_steps` | 0 | Offline pretraining steps (0=disabled) |
| `discount` | 0.99 | Discount factor γ |

---

## Optimizer

Both Q and V use **AdamW** with gradient norm clipping:
```
optimizer = AdamW(clip_gradient_norm=1.0)
lr_schedule = ConstantSchedule(value=5e-5)  # set in launcher
```

Default critic LR in config: `3e-4`. Launcher overrides to `5e-5`.

---

## W&B Logging

| Metric prefix | X-axis | Content |
|---|---|---|
| `critic/q_*` | `step` (online) | Q loss, TD loss, MC loss, grad norm, value mean |
| `critic/value_*` | `step` (online) | V loss, TD loss, MC loss, grad norm, value mean |
| `pretrain/q/*` | `pretrain_step` | Same metrics during offline pretraining |
| `pretrain/value/*` | `pretrain_step` | Same metrics during offline pretraining |
| `eval/*` | `eval/total_collected_episodes` | Success rate, episode length, etc. |
| `online_buffer_size` | `step` | Number of transitions in replay buffer |
