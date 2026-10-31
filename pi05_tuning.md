# Running and tuning OGPO / AWR with pi0.5

`ogpo/main_pi05.py` fine-tunes the **pi0.5 action expert** with a critic on the pi0.5 prefix features. The VLM prefix is frozen (no gradient);
checkpoints store only the trainable state, and the frozen weights come from the pi0.5 checkpoint you point at. Two env families:

| env family | env names | pi0.5 checkpoint | `agent.pi05_robot` |
|---|---|---|---|
| MolmoSpaces (FrankaPickDroid) | `molmo_pi05-<benchmark episode id>` | `pi05_droid_jointpos` | `droid` |
| LIBERO | `libero_pi05-libero_90-<task id>` | `pi05_libero` | `libero` (default) |

`main_pi05` checks that the robot matches the env family and stops otherwise. Two algorithms share the entry point:
**OGPO** (`agent.pi05_actor_objective=ppo`, the default) and **paper AWR** (`awr`, Peng et al. 2019). A third, `group_awr`, keeps OGPO's SDE
samples and group advantages and swaps the PPO term for the AWR weight (`min(exp(A/awr_beta), awr_weight_max) * CFM`).

Everything below is taken from `ogpo/main_pi05.py`, `ogpo/configs/algos/ogpo.yaml` and the four recipes. **Not yet validated end to end:** AWR on
MolmoSpaces (the AWR recipes are the OGPO recipes plus the AWR changes of the earlier LIBERO AWR recipe).

## 1. Quick start: the four MolmoSpaces recipes

| | one task (box 138) | 16 tasks (`mt16`) |
|---|---|---|
| OGPO | `scripts/ogpo_pi05/molmo_box138.sh` | `scripts/ogpo_pi05/molmo_mt16.sh` |
| paper AWR | `scripts/awr_pi05/molmo_box138.sh` | `scripts/awr_pi05/molmo_mt16.sh` |

One seed (0). The hyperparameters are those of the multi-task-144 OGPO runs (seed 1); only the task set and what scales with it differ
(`env.name`, `parallel_envs`, `inference_batch`, `buffer.size`, CPUs). They are sized for **4 x 96 GB GPUs** (`jax_frac=0.70`,
`grad_accum_steps=1`); the SBATCH headers ask for 32 CPUs (one task) or 64 CPUs (`mt16`) and `--mem=400G`.

```bash
# 1. once: the molmo Python env (.venv-molmo)
bash scripts/ogpo/setup_molmo_venv.sh --molmospaces=<lasgroup/molmospaces checkout @ 2a828d5> \
                                      --openpi_client=<openpi checkout>/packages/openpi-client

# 2. once: pi05_droid_jointpos, the PaliGemma tokenizer and the MolmoSpaces assets (needs gsutil; use a compute node)
export OPENPI_CACHE=... MLSPACES_ASSETS_DIR=... MLSPACES_CACHE_DIR=... NLTK_DATA=...   # directories to fill
mkdir -p logs && sbatch scripts/ogpo/download_molmo_assets.sbatch    # or: bash scripts/ogpo/download_molmo_assets.sbatch

# 3. train, with the same four variables exported
SAVE_DIR=... bash scripts/ogpo_pi05/molmo_mt16.sh                    # or sbatch it
```

* **Paths have no defaults.** The scripts exit with a message if `SAVE_DIR`, `MLSPACES_ASSETS_DIR`, `MLSPACES_CACHE_DIR` or `NLTK_DATA` is
  empty, and also `OPENPI_CACHE` unless both `--pi05_checkpoint=` and `--pi05_tokenizer=` are given. Any of them can be passed as `--VAR=...`.
* **Overrides:** `--variable=value` after the script name sets the shell variable of that name in the script's config block. **An unknown
  name is silently ignored**, so check the first lines of the output (`=== OGPO-pi05 MOLMO ... ===` prints the main settings).
* **Weights & Biases:** `log=true` logs to W&B. Run `wandb login`, or `export WANDB_MODE=offline` and `wandb sync` later. `--log=false` turns
  W&B off **and** stops the CSV logs and checkpoints from being written (the save directory becomes `None`).
* **`TASKS` for the download:** `download_molmo_assets.sbatch` prefetches the 16 `mt16` tasks by default (the one-task recipe's task 138 is
  one of them). For other tasks: `TASKS=164,86,... sbatch scripts/ogpo/download_molmo_assets.sbatch`.

### Using a `vla-post-training` checkout

`vla-post-training` carries `openpi` and `molmospaces` as submodules (`.gitmodules`), so its checkouts can supply the three source trees:

| argument | where it can come from |
|---|---|
| `--molmospaces=` | `<vla-post-training>/molmospaces` (the molmo recipes were run with the commit `2a828d5`) |
| `--openpi_client=` | `<vla-post-training>/openpi/packages/openpi-client` |
| `OPENPI_CACHE` | a directory holding `openpi-assets/checkpoints/pi05_droid_jointpos/` and `big_vision/paligemma_tokenizer.model` (the earlier recipes defaulted to `<vla-post-training>/cache/openpi`) |

## 2. The command line

The recipes end in `python ogpo/main_pi05.py --algo=ogpo --section.key=value ...`. Flags are the standard `ogpo.yaml` keys plus the pi05 ones.
The ones you will touch most:

| flag | meaning |
|---|---|
| `--env.name=a,b,c` | one or more tasks of **one** family. Several = multi-task (needs `--training.collection_mode=rounds`) |
| `--env.eval_name=...` | evaluation tasks (default: the training tasks) |
| `--env.vector_envs=true` `--env.parallel_envs=m` `--env.inference_batch=c` | molmo needs vector envs (the sim must stay out of the trainer process). `m` workers per task; `c` rows per encode / policy call, split over the GPUs. Workers = tasks x `m` must fit the CPU allocation (the scripts check `workers + 2`) |
| `--env.worker_egl_devices=auto` | spread the workers' EGL contexts over the run's GPUs (default `""`: every worker on the trainer's GPU) |
| `--training.collection_mode=rounds` | each round: collect `rollouts_per_round` episodes **per task** with the policy frozen, then run `updates_per_round` updates. The step axis is the update count |
| `--training.gpu_mode=multi` + `CUDA_VISIBLE_DEVICES` | data parallel over the visible GPUs; every batch size is global and must divide by the GPU count |
| `--agent.pi05_checkpoint` `--agent.pi05_tokenizer` `--agent.pi05_robot` `--agent.pi05_action_horizon` | the pi0.5 weights, tokenizer, robot profile and action chunk length of the model (`droid`: 15; the LIBERO recipes used 10) |
| `--buffer.storage_dir=<dir>` | keep the replay images (`images`, `next_images`, about 0.6 of the 0.62 MB per transition) in memmaps on node-local disk instead of RAM; the scripts expose this as `--buffer_storage_root=<dir>` |

## 3. What to tune

Values are those the recipes use. Defaults are in `ogpo/configs/algos/ogpo.yaml`.

**Schedule.** `num_rounds=10`, `rollouts_per_round=20` (per task), `updates_per_round=1000`, `eval_episodes=10` (per task),
`eval_interval=2000`, `log_interval=1000`, `buffer_size` 100k (one task) / 1.2M (`mt16`), FIFO. `horizon_length=5` executed steps of each
15-step chunk; `discount=0.995`; `post_success_steps=0` (molmo ends the episode on success: reward -1 per step, no bonus).

**OGPO** (`pi05_actor_objective=ppo`)

| knob | recipe value | what it does |
|---|---|---|
| `best_of_n` | 32 | candidates scored by the critic at collection |
| `num_qs`, `pi05_bon_agg` | 10, `subsample` | Q ensemble size; BoN head reduction `min \| subsample (min of 2 random heads) \| mean` |
| `adv_strategy` | `conservative` | per-head advantages, zeroed unless all `num_qs` heads agree on the sign (alternatives: `vanilla`, `awr`) |
| `group_num_samples`, `ppo_batch_size` | 32, 32 | `G` SDE chains per PPO state, and PPO states per update |
| `clip_epsilon` | 0.01 | PPO clip |
| `bc_coeff`, `use_bc_regularization` | 1.0, true | BC anchor on the actor; `success_buffer.*` sets the BC and success-buffer critic batches (512) |
| `use_tapered_noise`, `constant_noise_std` | true, 0.1 | SDE noise schedule |
| `q_variance_reduction`, `q_vr_num_samples`, `q_vr_reduction` | true, 16, `median` | the TD target reduces Q over 16 next actions |
| `pi05_critic_arch`, `pi05_critic_pooling` | `bronet_vla`, `all` | BroNet Q head as in vla-post-training (needs pooling `all`); `pi05_bronet_hidden_dim` 1024, `pi05_bronet_depth` 2 |
| `ppo_lr` / `pi05_lr` | 1e-4 / 2.5e-5 | critic / actor learning rate; `pi05_clip_grad_norm` 1.0, `pi05_weight_decay` 1e-10 |
| `tau`, `actor_tau` | 0.05, 0.005 | target-network rates |
| `act_min` / `act_max`, `clip_intermediate_actions` | -inf / inf, false | no action clipping for molmo (joint deltas; +-1 is DROID's 1st / 99th percentile) |

**AWR** (`pi05_actor_objective=awr`): needs `collection_mode=rounds`, `use_value_fn=true`, `v_training_mode=lambda_return`, and
`freeze_actor=false`; `first_round_policy` stays `bon`. Each round: collect with the online actor, recompute `R^lambda` for the buffer, fit `V`
for `awr_value_steps` steps, then `updates_per_round` policy steps with weight `min(exp((R^lambda - V)/awr_beta), awr_weight_max)`. Eval is ODE only.

| knob | recipe value | note |
|---|---|---|
| `awr_beta`, `awr_weight_max`, `awr_lambda` | 0.05, 20.0, 0.95 | the paper's values |
| `awr_value_steps`, `awr_value_refreshes` | 200, 1 | `V` steps per refresh; refreshes per round |
| `awr_normalize_adv` | false | standardise `A` over the buffer before the exponential |
| `v_ensemble_size`, `v_agg` | 2, `mean` | at least 2 heads |
| `best_of_n`, `bc_coeff`, `use_bc_regularization`, `q_variance_reduction` | 1, 0.0, false, false | no best-of-N, no BC anchor, no Q critic |

**Memory and throughput** (what to change when a run does not fit):

* `agent.grad_accum_steps=K` splits each update into `K` sequential micro-batches (global batches unchanged). The recipes use 1 with
  `pi05_drop_empty_camera=true`, which skips the always-masked right-wrist slot in KV-only prefix passes (about 1/3 fewer prefix tokens, same KV);
  without it, `K=1` ran out of memory in the mt144 fit test on 96 GB GPUs (`K=2` fit).
* `inference_batch`: BoN memory grows linearly in it (about 17.8 MB x rows / n_devices x `best_of_n` per device).
* `jax_frac` (`XLA_PYTHON_CLIENT_MEM_FRACTION`): the XLA pool; the rest of the GPU holds the EGL worker contexts. The recipes use 0.70 on 96 GB.
* `--training.zero_actor_state=true` shards the actor optimiser state (about 15 GiB per GPU less) but was about 20x slower on 8 x L40S in one
  test; the recipes keep it off.
* Host RAM: images dominate. A full 1.2M buffer is about 744 GB; use `buffer.storage_dir` (node-local disk) if the allocation is smaller.

## 4. Outputs

A run writes to `<SAVE_DIR>/<project>/<run_group>/<env_name>/<wandb_name>seed<seed>/` (`env_name` is hashed if longer than 100 characters):
`flags.json` (the resolved config), `checkpoint_latest.pkl` (trainable state, overwritten at every evaluation) and one CSV per logging prefix:

| file | contents |
|---|---|
| `env.csv` | per round: collection statistics (success, episodes, env steps) |
| `online_agent.csv` | update metrics |
| `eval.csv` / `eval_sde.csv` | the ODE evaluation / the best-of-N evaluation (`eval_interval` updates; AWR: ODE only) |
| `awr.csv` | AWR only: per-round `R^lambda` and `V`-fit statistics |

The same metrics go to W&B when it is online.

## 5. Resume and evaluation

* **Evaluate a checkpoint:** run the recipe's `python ogpo/main_pi05.py ...` command with the same agent flags and add
  `--training.restore_checkpoint=<dir>/checkpoint_latest.pkl --training.eval_only=true` (optionally `--training.eval_modes=ode,bon` and
  `--env.eval_name=...`). It runs one evaluation at the checkpoint's step and exits. The recipes do not expose these as variables, so
  copy the final command out of the script.
* **Resume a job** (rounds only): `--training.resume_s3=<s3:// URI or directory>` (and `--training.resume_dir=<local scratch>`) mirrors the
  replay buffer and the agent every `resume_ckpt_interval` updates; a restarted job with the same settings continues from it.
  `--training.resume_rebase_rounds=true` continues a finished round with a new `updates_per_round`. The recipes do not set these.
* `training.resume_snapshot_dir` continues from a directory holding `agent.pkl`, `buffer/` and `csv/`. The tool that took such a snapshot of
  a running job is no longer in this repository (`save_buffer_dump` in `ogpo/utils/resume.py` writes the `buffer/` part).

## 6. LIBERO

The code path is the same (`--env.name=libero_pi05-libero_90-<id>[,...]`, `pi05_robot=libero`, `pi05_checkpoint=<...>/pi05_libero`, install with
`uv sync --extra cuda12 --extra libero`). This branch carries no LIBERO recipe: the LIBERO OGPO and AWR recipes
(`scripts/ogpo/libero_pi05_mt4_bronet_vla.sh`, `libero_pi05_mt4_awr.sh`; discount 0.99, `pi05_action_horizon=10`, `post_success_steps=1`)
are on the `pranav/pi05-awr` branch. LIBERO has no `vector_envs` requirement, but the multi-task schedules still need `collection_mode=rounds`.

## 7. Tests

CPU tests, no GPU or assets needed. Run them in two processes: one process with all the files runs out of memory or threads on a busy
machine and reports spurious `Cannot allocate memory` failures.

```bash
export JAX_PLATFORMS=cpu
python -m pytest tests/test_pi05_group_awr.py tests/test_actor_gradient_flow.py tests/test_ema_and_is.py tests/test_reparam_logprob.py \
                 tests/test_merge_vla_ogpo.py tests/test_pi05_bronet.py tests/test_pi05_pooling.py -q
python -m pytest tests/test_pi05_example_scripts.py tests/test_pi05_awr.py tests/test_pi05_schedule.py tests/test_resume.py \
                 tests/test_resume_snapshot.py tests/test_buffer_storage_dir.py tests/test_pi05_drop_empty_camera.py tests/test_pi05_robot_profile.py -q
```

Expected: the first command has 6 failures that predate the recipes (five in `test_actor_gradient_flow.py`, whose test calls
`OGPOAgent.batch_update` with one argument too many, and `test_verbatim_copies_match_git_head`); the second has 6 skips that need a GPU and the
checkpoint. `tests/test_pi05_example_scripts.py` runs each recipe in a sandbox with a fake `python` and checks the resolved flags, so it
catches a changed hyperparameter or path handling without a GPU. It cannot show that a run starts or learns.
