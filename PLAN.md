# PLAN: match OGPO's LIBERO setup to vla-post-training (pi05 action expert)

Goal: an apples-to-apples comparison between
- **A** = this repo (`/home/mananaga/OGPO`, `scripts/ogpo/libero_pg.sh`), whose OGPO algorithm is the reference, and
- **B** = `/home/mananaga/VLA/ogpo/vla-post-training` (`scripts/ogpo_libero.sh`, config `pi05_libero_online_ogpo_sft`).

We keep A's algorithm (actor/critic losses, SDE sampling, BoN, TD target, success buffer, EMA/target nets, clipping) and port B's **setup and architecture** into A.

Citations use `file:line`. A-paths are relative to `/home/mananaga/OGPO`. B-paths are relative to `vla-post-training`.

---

## 0. Summary table

| # | Item | A (this repo, now) | B (vla-post-training) | Priority |
|---|------|--------------------|-----------------------|----------|
| 1 | Policy network | Random-init 4.3M-param MLP flow actor (`networks/actors.py:126-241`) | pi05 **action expert** (Gemma-300M) initialised from `pi05_libero/params`, conditioned on the frozen PaliGemma prefix (`openpi/.../pi0.py`) | **P0** |
| 2 | Demo BC pretraining | `bc_pi_steps=500000` flow-matching BC on raw libero_100 HDF5 demos (early stop at SDE-SR ≥ 0.45) | None. `online_ratio=1.0`, no dataset loaded (FSL:275-289). Starts from the pi05_libero SFT checkpoint | **P0** |
| 3 | Language | Not used anywhere (`paligemma.py:312-315` drops Gemma) | LIBERO task string, PaliGemma tokenizer, `max_token_len=200`, no discrete state tokens | **P0** |
| 4 | Obs encoder / features | SigLIP only, 256 tokens → 2048-D after mean-pool + raw 8-D state, then a trainable TwoTier module | SigLIP + Gemma-2B prefix (3 imgs × 256 + 200 text = 968 tokens), frozen, bf16. The actor attends to it via the KV cache | **P0** (follows from 1) |
| 5 | Action representation | 4 × 7 raw env actions in [−1, 1], no normalisation | 10 × 32 quantile-normalised (q01/q99 from `pi05_libero/assets`), first 7 dims used, unnormalised before `env.step` | **P0** (follows from 1) |
| 6 | Chunk / replan | H=4, all 4 executed open-loop | H=10 predicted, **5 executed** (`replan_steps=5`, B `config.py:422`) | **P0** (decision needed, §2.3) |
| 7 | Cameras / resolution | agentview only, rendered 128×128, PIL-bilinear upsample to 224 | agentview **+ wrist**, rendered 256×256, resize to 224. Third slot is zeros with `image_mask=False` | **P1** |
| 8 | Reset warm-up | 0 no-op steps | 10 no-op steps `[0,0,0,0,0,0,-1]`, not counted toward the 400-step limit (B `envs/libero.py:79-80,109-120`) | **P1** |
| 9 | Proprio to policy | 8-D raw state goes to actor and critic | Policy sees **no state** (`discrete_state_input=False`). Critic gets quantile-normalised state padded to 32 | **P1** (follows from 1) |
| 10 | Critic input features | TwoTier(L2-normed SigLIP mean 2048 ‖ raw state 8) + raw 28-D chunk | Mean over all 968 Gemma-prefix final hidden states (2048) ‖ normalised state (32) ‖ normalised chunk (10 × 32) | **P1** |
| 11 | Critic body | 10-head vmapped MLP 1024×3 + LN, GELU | BroNet (1024, depth 2), 2 Q + 2 V towers | P2 (algorithmic, keep A. See §3) |
| 12 | Reward | −1/step, **+5** on the success step (`envs/libero_utils.py:344-365`) | −1/step, **0** on the success step (`SUCC_BONUS=0`) | P1 (decision needed) |
| 13 | Task | `libero_90_59` (tomato sauce → tray) | `libero_90_38` (right moka pot → stove) in the current script. B's earlier working OGPO run was on 59 (`tests/ogpo/OGPO_DEBUG_LOG.md:12,93`) | P1 (decision: use 59 in both) |
| 14 | Image aug | None | In the BC anchor only: crop 95 %, ±5° rotate, colour jitter (openpi `model.py:168-187`) | P2 |
| 15 | Precision / optimiser | fp32. AdamW, clip 1000, wd 1e-6, actor LR 4.5e-5 cosine-repeating | bf16 compute, frozen leaves cast to bf16. AdamW b2=0.95, clip 1.0, wd 1e-10, constant 2.5e-5 | P1 (follows from 1) |
| 16 | Eval | 20 eps sequential, init states `seed=global_step+i` (different each eval) | 32 eps, 8 parallel envs, envs rebuilt and seeded `0+i` (same init states every eval), EMA weights, ODE | P2 |
| 17 | Env parallelism | 1 train env, 1 eval env | 8 subproc envs for collection and eval | P2 |
| 18 | Sample budget / buffer | 2M env steps, UTD 1, buffer 2M of 2056-D features (~33 GB) | ~60 episodes (~24k env steps), buffer stores images + tokens + pooled prefix | **P0 practical** (see §4) |

"Follows from 1" means the item is done automatically once the pi05 policy is in.

---

## 1. Setup and architecture differences in detail

### 1.1 Policy: MLP actor → pi05 action expert (P0)

**A**
- `ActorVectorField`, random init.
- obs (2056) → TwoTier → 1024, concatenated with the noisy chunk (28) and a sinusoidal time embedding (32).
- The trunk is an MLP 1024×3 with no LN (`networks/actors.py:126-241`, built at `agents/ogpo.py:3126-3139`).
- Flow time runs 0 = noise → 1 = data, with velocity `a − x0`.

**B**
- `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` (B `config.py:547-549`), loaded from `gs://openpi-assets/checkpoints/pi05_libero/params`.
- Freeze filter (B `config.py:633-654`):
  - **frozen**: all of PaliGemma, meaning Gemma-2B (LLM stack 0) and SigLIP.
  - **trainable**: action expert (LLM stack 1), `action_in_proj`, `time_mlp_in/out`, `action_out_proj`.
- No LoRA. Frozen leaves are cast to bf16, trainable leaves stay fp32 with bf16 compute.
- Time conditioning is adaRMS.
- **Flow time runs 1 = noise → 0 = data**, with velocity `noise − a` (pi0.py). The sign and time convention are the opposite of A's.

**What to port**

A new actor class, in a separate file per the repo convention, that wraps openpi's `Pi0`. It should expose the same interface A's samplers call, i.e. `velocity(obs, x_t, t)`, with a thin adapter:
- `t_pi = 1 − t_A`
- `v_A = −v_pi`

A's ODE and SDE samplers (`agents/modules/pg_helper.py:46-79`, `:125-261`) and the log-prob code then stay untouched.

The prefix (images + prompt) goes through the frozen PaliGemma once per state. Its KV cache is reused for all flow steps and all group/BoN samples. This is B's `dedup_group_prefix` trick (B `src/rl/ogpo/sampling.py:241-300`) and it is essential for speed.

The target actor becomes a copy of the trainable (action-expert) leaves only. The frozen prefix is shared between online and target.

### 1.2 No demo BC (P0)

**A**
- 500k offline BC steps on `$LIBERO_DATASET_ROOT/libero_100/*_demo.hdf5`.
- These are the raw demos, which still contain no-op actions and are 128 px.
- Loaded by `envs/libero_utils.py:121-256` and run by `runners/bc_runner.py`.

**B**
- No demos at all. The pi05_libero checkpoint is the starting policy.
- The only BC is the online success-buffer anchor, which A already has (`use_bc_regularization`, success buffer).

**What to port**
- `bc_pi_steps=0`.
- Skip `get_dataset` entirely. Otherwise we would still need to pre-encode 50 demos through the new pipeline for nothing.
- `offline_ratio=0` is already set.
- Keep A's success-buffer BC anchor.

**Open question.** openpi's pi05_libero was fine-tuned on the four standard LIBERO suites, not libero_90. I have not verified this in code. The base success rate on 59 must therefore be measured first: run a pure eval with no training. B reached 62 % on 59 with this checkpoint, so it is non-zero.

### 1.3 Language prompt (P0)

**A**: none.

**B**
- The prompt is `task.language` from LIBERO, e.g. `"pick up the tomato sauce and put it in the tray"` for 59.
- Tokenised by `PaligemmaTokenizer.tokenize(prompt, state=None)`: `strip()`, `_` → space, BOS + text + `"\n"`, padded to 200 (openpi `models/tokenizer.py:22-48`).
- `tokenized_prompt_mask` is used in prefix attention.

**What to port**
- Tokenise once per episode. The prompt is constant for a single task.
- Store the tokens in the buffer, or rebuild them from the task id at sample time (cheaper).
- A's env already puts `task_description` into `info` (`envs/libero_utils.py:303,341`).

### 1.4 Observations (P1)

| | A | B |
|---|---|---|
| Render | 128×128 (`libero_utils.py:34-40`) | 256×256 (B `config.py:396`) |
| Cameras | agentview | agentview + `robot0_eye_in_hand` |
| Flip | 180° both axes | 180° both axes (same) |
| To 224 | PIL bilinear upsample | `resize_with_pad` (plain resize for a square image) |
| Pixel norm | `/255 → [-1,1]` | same inside openpi |
| State | eef_pos(3) + axis-angle(3) + gripper_qpos(2), raw | same 8 dims, quantile-normalised, padded to 32 |
| Warm-up | none | 10 no-op steps, gripper open |

**What to port**
- Render at 256.
- Return both cameras.
- Add the 10-step warm-up inside `reset`, not counted toward the 400 steps.
- Build the openpi `Observation` with a zeros third image and `image_mask=False`.
- Apply the pi05_libero input transforms: `LiberoInputs` → `Normalize` (quantile) → `ResizeImages(224)` → `TokenizePrompt` → `PadStatesAndActions(32)`.

We should reuse openpi's transform objects rather than re-implementing them, so the norm stats and ordering are identical.

### 1.5 Action space and normalisation (P0, follows from 1.1)

**A**
- The policy emits raw env actions in [−1, 1].
- Samplers clip means and samples to [−1, 1] (`denoised_clip_value=1.0`, `pg_helper.py`).

**B**
- The policy operates in quantile-normalised space, 10 × 32 with zero padding.
- Actions are unnormalised (`Unnormalize` → `LiberoOutputs[..., :7]`) only before `env.step`.
- Buffer actions are stored normalised.

**What to port**
- Keep A's samplers operating in the normalised 32-D space.
- Unnormalise only at env-step time.
- A's `[−1,1]` clipping becomes a clip in normalised space. That is roughly the q01/q99 box, so it is still sensible, but it does truncate the 1 % tails. Decide: clip or not. My recommendation is to keep the clip on the 7 real dims and zero the 25 padded dims (see 2.3).

### 1.6 Critic input (P1)

**A**
- Shared TwoTier(L2-norm SigLIP 2048 ‖ raw state 8) → 1024, concatenated with a raw 28-D chunk.
- Fed to a 10-head MLP ensemble (`networks/critics.py:49-185`).

**B**
- Mean over all 968 Gemma-2B prefix final hidden states gives 2048 dims.
- These are concatenated with the normalised state (32) and the normalised chunk (10 × 32 = 320), giving 2400 dims into BroNet (B `bronet_critic.py:136-146`).
- `prefix_pool_masked=False` (B `config.py:442`), so padded text slots and the zero-image tokens are included in the mean.
- B's newest commit adds `image_text_mean` pooling (4096 dims), but it is **off** in this script.

**What to port.** Give A's critic the same features: the pooled Gemma prefix (2048), the normalised state, and the normalised chunk. Keep A's ensemble body (see §3).

The pooled prefix is a cheap by-product of the prefix forward we already need for the actor. Cache it per env step in the buffer, as B does with `store_prefix_rep`.

Do **not** copy B's stale-prefix bug. B computes the prefix once per chunk and repeats it for the next 4 steps (B FSL:705-717). We should compute it per stored observation.

### 1.7 Reward (P1, decision)

- A: −1/step, +5 on the success step, terminate on first success.
- B: −1/step, 0 on the success step, terminate on first success.
- `post_success_steps=1` is a no-op in A, because LIBERO sets `done` on success.

This affects the Q scale and is arguably part of the OGPO recipe, so it is your call:
- **(a)** Keep +5 (A is the reference).
- **(b)** Set bonus = 0 to match B.

For a pure setup comparison I would match B, or set B's `SUCC_BONUS=5`, which B already supports.

### 1.8 Task (P1, decision)

A runs 59. B's current script runs 38. B's previous working OGPO number (62 %) is on 59. **Recommendation:** use `libero_90_59` for both, and change B's `TASKS=(libero_90_59)` when running the B side.

### 1.9 Eval protocol (P2)

**A**
- 20 episodes, sequential.
- Init states reseeded with `global_step + i`.
- Two evals: ODE with the online actor, and SDE-BoN with the target actor and target critic.

**B**
- 32 episodes with 8 parallel envs.
- The same init-state sequence at every eval (envs reseeded with `0 + i`).
- ODE with EMA weights.

**What to port**
- `eval_episodes=32`.
- Fixed eval seeds, i.e. the same init-state indices each eval.
- Report ODE-with-target-actor as the headline. A's target actor with `actor_tau=0.005` is the analogue of B's EMA 0.99 per policy step, so the two are not identical but are comparable.
- Keep A's SDE-BoN eval as a second number.
- Parallel eval envs are a nice-to-have. `get_env_maker` has no LIBERO branch (`utils/main_helper.py:67-84`), so `n_eval_envs>1` currently crashes.

### 1.10 Image augmentation (P2)

B augments only the success-buffer BC batch: RandomCrop 95 %, ±5° rotation on non-wrist images, and ColorJitter, all via openpi `preprocess_observation(train=True)`. A has none.

Porting this is cheap if we call openpi's `preprocess_observation` in the BC loss path. It only makes sense once images, not features, are in the buffer.

### 1.11 Precision and optimiser for the new actor (P1)

B uses bf16 compute and AdamW(b1 0.9, b2 0.95, eps 1e-8, wd 1e-10, clip 1.0) at a constant 2.5e-5 for the action expert.

A's `ppo_lr=4.5e-5` cosine-repeating schedule, `clip_grad_norm=1000` and wd 1e-6 were tuned for a 4M-param MLP from scratch, and are likely wrong for a pretrained 300M transformer. **Recommendation:** use B's optimiser settings for the actor, since this is an architecture-driven choice, and keep A's critic optimiser. Note that the critic LR is `ppo_lr` online, not `critic_lr` (`agents/ogpo.py:2899-2905`).

---

## 2. Consequences for OGPO hyperparameters forced by the new architecture

These are algorithm hyperparameters, but the pi05 actor forces changes, so they need a decision.

### 2.1 Batch × group size

A uses `ppo_batch_size=256` × `group_num_samples=32`, which is 8192 action-expert chains × 10 flow steps per update. That is infeasible with a 300M expert.

B uses 32 × 8 = 256 chains with a prefix dedup. **Proposal:** start at B's 32 × 8 and scale G up if memory allows. The goal is to keep A's G=32 if possible, since that is the algorithm's value.

### 2.2 UTD and sample budget

A runs 2M env steps at UTD 1, i.e. 2M updates, each of which would include a Gemma-2B prefix forward on 256 states. That is infeasible.

B uses ~24k env steps and 100k trainer steps (critic every step, policy every 10). **Proposal:** match the env-step budget to B (~24k–50k env steps) and choose `utd` so that total actor and critic updates are comparable. See §4 for the buffer memory this implies.

### 2.3 Chunk length and executed steps (decision)

A's Q is Q(s, a_{t:t+H}) with the whole chunk executed. pi05 predicts 10 steps, and B executes 5 while its critic sees all 10 (a mismatch).

Options:
- **(a) Recommended.** Keep the pi05 horizon at 10 and set A's `horizon_length=5`: execute 5, and let the critic and TD target use the first 5 × 7 real dims. Restrict the PG log-prob to those same executed, real dims, so gradients do not flow into unexecuted or padded outputs.
- **(b)** Execute all 10 (`horizon_length=10`). This is simplest and exactly A-consistent, but it changes pi05's closed-loop behaviour from its training setup (openpi's LIBERO eval replans every 5).
- **(c)** Match B literally: execute 5, critic on 10 × 32. This inherits B's inconsistency, so it is not recommended.

### 2.4 SDE noise

A uses the tapered σ_t = 0.1·√(1−t_A) with the drift correction. B uses σ_t = 0.02·√(t/(1−t)).

A's SDE is the algorithm, so keep it, via the time-mapping adapter. But the scale 0.1 was tuned for [−1,1] raw actions from a from-scratch MLP. In normalised space it is probably still OK, but check that the initial BoN and SDE success rates are not far below the ODE rate.

### 2.5 BC-anchor time distribution

A's flow-matching BC uses t ~ U(0,1). pi05 was pretrained with t ~ Beta(1.5, 1) (openpi `pi0.py:310`). This is minor. Keep A's unless the BC anchor visibly degrades the pretrained policy.

---

## 3. Deliberately NOT changed (algorithmic, A is correct)

- Actor objective: PPO clip, advantages, group sampling, adv_strategy.
- SDE sampler, log-prob normalisation, BoN rollouts with the target actor and target critic. B uses ODE rollouts.
- TD target: chunked n-step, min over 2 random heads of 10, `mask`/`valid` handling, success-buffer Q update.
- Critic ensemble body: 10 heads, MLP + LN. We match its **inputs** to B, not its body. Switching to BroNet with 2 Q + 2 V would change the algorithm.
- γ = 0.99 per env step (B uses 0.995), target τs, success buffer, `start_training`, per-step UTD collection (B collects in rounds of 5 episodes).
- B-only recipe knobs: PG warm-start/ramp, `grpo_conservative`, EMA-quantile normaliser, ±4 clip, critic burst. None of these are ported.

If you later want B-side runs to be closer to A, B's `pi05_libero_online_ogpo_ref` config already aligns several of these (`docs/changes/2026-08-20-ogpo-reference-alignment/`).

---

## 4. Implementation order

Each step is independently testable.

1. **Env parity** (small, `envs/libero_utils.py`):
   - Render at 256.
   - Add the wrist camera.
   - Add the 10-step warm-up.
   - Make the reward bonus a flag.
   - Fix eval seeds.
   - Default the task to 59.

   Test: compare obs and state against B's `Pi0ObservationWrapper` on the same init state.
2. **openpi in A's environment**:
   - Make `openpi` importable, via a PYTHONPATH to B's vendored copy or a path dependency.
   - Load `pi05_libero` params, norm stats and the tokenizer.

   Test: a pure ODE eval of the unchanged pi05_libero on task 59 with 5-step replanning gives the base SR. This is also the step-0 number for both codebases.
3. **pi05 actor adapter** (new file, e.g. `ogpo/networks/pi05_actor.py`):
   - Prefix forward with KV cache.
   - Action-expert velocity with the t/sign mapping.
   - Trainable/frozen split.
   - Target-actor copy of the trainable leaves only.

   Test: A's `compute_flow_actions` through the adapter reproduces openpi `sample_actions` (ODE) to bf16 tolerance.
4. **Buffer and obs path**:
   - Store uint8 224² images ×2 plus the pooled prefix (2048) per env step. The tokens are constant per task.
   - Drop the SigLIP-only pre-encoding path for this config.

   Memory at 50k steps ≈ 50k × (2 × 150 KB + 8 KB) ≈ 15 GB of host RAM, which is fine. At A's 2M it would be ~600 GB, hence §2.2.
5. **Critic features**: pooled prefix ‖ normalised state ‖ normalised executed chunk, feeding A's existing 10-head ensemble.
6. **Wire the OGPO update**:
   - The actor loss recomputes the prefix from the buffer images (frozen, `stop_gradient`, dedup across G).
   - BC anchor on the success buffer, with optional openpi augmentation.
   - bf16 compute.
   - Use B's optimiser for the actor.
7. **New launch script** `scripts/ogpo/libero_{82,79,38,31}_pi05.sh`, leaving the existing `libero_pg.sh` untouched:
   - `bc_pi_steps=0`
   - the new actor/encoder flags
   - batch and G from §2.1
   - budget from §2.2
   - `eval_episodes=32`
8. **Matching B run**: `TASKS=(libero_90_59)` and the chosen `SUCC_BONUS`, with everything else as is.

## 4b. openpi port (decided)

OGPO's `.venv` has jax 0.9.0.1 and flax 0.12.4. openpi pins jax 0.5.3 and flax 0.10.6, plus torch, lerobot and transformers. So openpi is **not** installed. Instead, the minimal model code is copied into `ogpo/networks/pi05/`, all in flax.linen:

- `gemma.py`, `siglip.py`: copied from openpi. Only the sharding, array_typing and lora imports are stripped.
- `pi05.py`: a linen port of `embed_prefix`, `embed_suffix` and the KV-cached velocity from openpi's nnx `pi0.py`.
- `normalize.py`: quantile norm/unnorm, plus loading of the `pi05_libero/assets` norm stats.
- `tokenizer.py`: PaliGemma prompt tokenisation only. Adds the `sentencepiece` dependency.

Weights are loaded from `pi05_libero/params` via orbax, the same as `encoders/paligemma.py`.

Verification is a GPU-node script, run by the user, that compares against openpi in the babel venv on identical obs and noise.

## 4c. Progress

- [x] Step 1: env parity, in `envs/libero_pi05_utils.py` (env name `libero_pi05-libero_90-59`).
  - 256 px render, agentview + wrist, 10 warm-up steps, reward unchanged.
  - The existing `libero-…` env and `libero_pg.sh` are untouched.
  - LIBERO assets come from `~/.cache/libero/assets` via hf_libero's fallback, so this only works on compute nodes.
- [x] openpi port, in `ogpo/networks/pi05/{gemma,pi05,transforms}.py`. SigLIP is reused from `encoders/paligemma.py`. `sentencepiece` was added to `pyproject.toml`.
  - Verified against openpi by `scripts/pi05_verify/{ref_openpi,check_port}.py`:
    - images and tokens exact, state to 7e-7
    - actions to ≤1.2e-2 (bf16)
    - prefix to 0.5–1 % relative, cosine ≥ 0.9999
- [x] Base SR of the unchanged pi05_libero on task 59 (`scripts/pi05_verify/eval_base.py`, VLA precision, ODE, replan 5, seeds 0–19): **10 % (2/20)**. Mean success length is 257 steps. A failed episode takes about 8 s once compiled.
- [x] Steps 3–7: the pi05 OGPO agent, runner, entry point and launch script.
  - Files:
    - `ogpo/agents/ogpo_pi05.py`
    - `ogpo/runners/pi05_online_runner.py`
    - `ogpo/main_pi05.py`
    - `scripts/ogpo/libero_{82,79,38,31}_pi05.sh`
  - The agent mirrors OGPOAgent's reference path. It calls the same pg_helper / q_helper functions and reuses the `Value` critic, two-tier encoder and TrainState.
  - `pg_helper.sample_flow_actions_sde` and `compute_flow_log_prob` gained an optional `dim_mask`. The default path is bit-identical to before.
  - The tiny end-to-end run exits cleanly. At 32 × 8 an update takes 3.3 s and peaks at about 54 GB on the GPU.
  - Checkpoint saving: at each eval, the trainable state (actor, target actor, critic and optimiser states, about 7.7 GB) is written to `<save_dir>/checkpoint_latest.pkl`.
  - Not ported: utd > 1, resume, parallel eval envs.
  - Launch-script defaults are set for about 11 h: 20k env steps, `start_training` 2k, batch 16 × G 8, eval every 4k.
- [x] First 11 h run (`8pl6bf0v`): success rose from 5 % to 55–60 % at step 8k, then fell to 10–15 % by 20k.
  - Analysed with `scripts/ogpo/wandb_analysis.py`.
  - Suspected causes:
    - within-group Q spread is 5–15× below the critic's TD RMSE
    - the critic drifts optimistic while success falls
    - the online actor drifts from the target actor, raising the clip fraction
- [x] Diagnostics added:
  - update metrics are averaged over each log window
  - `mc_returns` in the buffer, with `critic/q_mc_corr`, `q_minus_mc` and `q/mc_{success,fail}`
  - `diag/*`: PG-vs-BC gradient norms and cosine, plus actor drift vs target and init, computed once per log window
  - `env/return`
- [x] Task 82 run `ec8oub40` (job 10606015, 18.5 h), with:
  - 4 GPUs
  - critic batch 256
  - PPO 16 × G32
  - `start_training` 20k, 30k steps
  - Results:
    - ODE eval 30 % → 97 %; best-of-N 27 % → 100 %
    - success episode length 259 → 91 steps
    - no collapse
  - Critic TD RMSE / within-group spread 21 → 2.7; `q_mc_corr` 0.3–0.39
  - Actor: PG share of the gradient ~34 %; online-vs-target drift steady at ~0.5 %.
- [ ] Step 8: matching B run on task 82 (`TASKS=(libero_90_82)`, `SUCC_BONUS=5`).
- [x] Collection schedule matched to vla-post-training's FSFT / BoN / AWR (step 1 of matching the baselines).
  - `training.collection_mode=rounds` in `run_pi05_round_rl`: 10 rounds × 20 full episodes = 200, policy frozen while collecting, 1000 updates per round. The 10k updates and all hyperparameters are unchanged.
  - Launcher: `scripts/ogpo/libero_82_pi05_matched.sh`.
  - The wandb step axis is the update count, and `env/*` values are per-round means.
  - Short real runs of both schedules pass (2026-10-01, 1 GPU, small batches).
- [ ] Step 2 of matching: the update schedule (critic : policy ratio, critic head start), see the AWR breakdown.
- [ ] Budget: env steps, UTD, batch × G.

## 5. Decisions (resolved)

1. Reward bonus: **keep +5** (A's). The B-side run uses `SUCC_BONUS=5`.
2. Chunk handling: **option (a)**. The pi05 horizon is 10, with 5 executed (openpi LIBERO standard). Critic, TD and PG log-prob use only the executed 5 × 7 real dims.
3. Batch × G and env-step budget: **deferred**. Use whatever fits for now and tune later.
4. Actor optimiser: **B's**. AdamW(b1 0.9, b2 0.95, eps 1e-8, wd 1e-10), clip 1.0, constant 2.5e-5.
5. Task: **`libero_90_59`** on both. From 2026-09-29 the pi05 runs moved to **`libero_90_82`** (STUDY_SCENE3, "pick up the book and place it in the left compartment of the caddy"). Task 59's first run is `8pl6bf0v`.
