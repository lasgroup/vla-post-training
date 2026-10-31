# Plan: Import LIBERO environment into OGPO (env-only, no algo changes)

## Goal

Add LIBERO as a new env family to OGPO so we can run the existing
`square_pg_libero.sh` recipe — frozen PaliGemma (pi05_libero) encoder + MLP
actor/critic, OGPO update — on a LIBERO task. Deliverable: a working
`scripts/ogpo/libero_pg.sh`. **No algorithmic changes.** All work is in the
env / dataset adapter layer.

The reference env wrapper lives in
`/home/mananaga/VLA/vla-post-training/src/envs/libero.py`; we will port (not
import) it so OGPO stays free of the VLA package.

---

## What OGPO requires from a new env

Two contracts, both currently met by `envs/robomimic_utils.py`:

### 1. Runtime env

`make_env_and_datasets(env_name, ...)` in `envs/env_utils.py:106` returns
`env, eval_env, train_dataset, val_dataset`. `env` / `eval_env` must be a
`gymnasium.Env` returning the standard 5-tuple
`(obs, reward, terminated, truncated, info)`. `obs` must be a dict with keys
that the actor/critic's online encode path can read. In the
`square_pg_libero.sh` recipe, the env is only consulted for online rollouts —
the online runner calls `encode_fn(state, image)` per step, so the env's obs
must expose a single agentview image and a flat proprio vector compatible
with whatever shape the offline dataset uses.

### 2. Offline dataset

`Dataset` (see `ogpo/utils/datasets.py`) with the following keys for an
image env (mirroring `robomimic_utils.get_dataset`,
`envs/robomimic_utils.py:306-309`):

- `observations` — proprio (T, D_proprio), float32
- `next_observations` — same
- `actions` — (T, 7), float32
- `rewards` — (T,), float32
- `terminals` / `masks` — per OGPO convention
- `images` — (T, H, W, 3) uint8, the camera that PaliGemma will encode
- `next_images` — same
- `full_states` — (T, D_full) float32 (object pose + proprio); folded into
  the encoded representation by `main.py:_pre_encode_dataset`
- `next_full_states` — same

For OGPO with `needs_mc_returns=True` (`main.py:216`), MC returns are computed
inside `process_train_dataset`, so we don't need to precompute them — but the
dataset must carry `terminals` / episode boundaries correctly.

### 3. Q-bounds

`get_q_bounds(env_name, discount)` (in `ogpo/utils/main_helper.py`) must
recognize LIBERO env names so OGPO can clip Q targets. We'll add a branch.

---

## Design

### Naming

- env_name format: `libero-<suite>-<task_id>`, e.g.
  `libero-libero_90-0`, `libero-libero_spatial-3`. Routing in `env_utils.py`
  triggered by `env_name.startswith("libero-")`.
- This keeps suite/task explicit and avoids colliding with Robomimic env
  naming.

### New files

1. **`envs/libero_utils.py`** — mirror of `envs/robomimic_utils.py`:
   - `make_env(env_name, seed, post_success_steps, ...)` — builds the
     Gymnasium env. Internally:
     - Parse `env_name` → `(task_suite_name, task_id)`.
     - Use `libero.libero.benchmark.get_benchmark_dict()` and
       `libero.libero.get_libero_path("bddl_files")` to resolve the BDDL file
       (same as VLA `libero.py:30-38`).
     - Instantiate `OffScreenRenderEnv` with
       `camera_heights=camera_widths=224` (PaliGemma SigLIP expects 224×224;
       confirm against `paligemma.py`).
     - Wrap with a thin `LiberoOGPOWrapper` (below).
   - `get_dataset(env_name)` — load LIBERO demos from HDF5:
     - LIBERO's offline demos live under
       `<libero install>/libero/datasets/<suite>/<task>.hdf5` (or wherever
       `get_libero_path("datasets")` points). For a single task we read one
       HDF5; for the multi-task variant, concatenate across files.
     - Each demo has keys `obs/agentview_rgb`, `obs/eye_in_hand_rgb`,
       `obs/ee_pos`, `obs/ee_ori`, `obs/gripper_states`, `actions`, `dones`.
       Concatenate across demos.
     - Build the dict with the same key names as `robomimic_utils.get_dataset`
       so `_pre_encode_dataset` Just Works.
     - `proprio_state = concat(ee_pos, axisangle(ee_ori), gripper_states)`
       (same composition VLA uses in `wrappers.py:obs_to_pi_zero_input`).
     - `full_state = concat(proprio_state, object_state)` if a flat object
       state is exposed by LIBERO; else fall back to proprio only (the
       two-tier encoder still works — `img_feat_dim` just becomes larger).
     - Clip actions to [-1+eps, 1-eps] (matches `env_utils.py:202`).
   - `_get_max_episode_length(task_suite_name)` — mirror VLA's
     `get_max_steps_libero` (libero_90→400, libero_10→520, etc.).
   - `LiberoOGPOWrapper(gym.Wrapper)` — small adapter exposing an obs dict
     with the same fields as the dataset:
     - `image` (agentview, uint8 HWC)
     - `state` (proprio vector)
     - `full_state` (proprio + object pose, used by `encode_fn`)
     - Standard Gymnasium 5-tuple step; passes through `terminated`,
       computes `truncated` via TimeLimit.
     - Provides `seed()` and reads `options.task_id` on reset (we'll fix
       task_id to the env_name parse so reset signature stays standard).
     - `post_success_steps` support: same trick as Robomimic — keep stepping
       N steps after `terminated=True` so the agent learns to settle, then
       emit `truncated`.

2. **`scripts/ogpo/libero_pg.sh`** — clone of `square_pg_libero.sh` with:
   - `env_name=libero-libero_90-0` (or whichever task we choose first; expose
     as a CLI override).
   - `run_name`, `wandb_name`, `run_group`, `project`, job-name renamed.
   - Drop `post_success_steps` to LIBERO's task horizon (start with 1, same
     as `square_pg_libero.sh`; revisit).
   - Keep encoder path `pi05_libero/params` (LIBERO-finetuned PaliGemma is
     the right encoder for LIBERO).
   - Tweak `actor_hidden_dims` / `value_hidden_dims` only if input dim
     changes warrant it; default to `[1024,1024,1024]` like the reference.
   - All other OGPO hyperparams unchanged (this is an env import, not a
     retune).

### Edits to existing files

1. **`envs/env_utils.py`** — add a branch before the final `else`:
   ```python
   elif env_name.startswith("libero-"):
       from envs import libero_utils
       env = libero_utils.make_env(env_name, seed=0, post_success_steps=post_success_steps)
       eval_env = libero_utils.make_env(env_name, seed=42, post_success_steps=post_success_steps)
       env = EpisodeMonitor(env)
       eval_env = EpisodeMonitor(eval_env)
       dataset = libero_utils.get_dataset(env_name)
       train_dataset, val_dataset = dataset, None
   ```

2. **`ogpo/utils/main_helper.py`** — extend `get_q_bounds` to recognize
   `libero-` prefix. LIBERO is sparse-reward by default (1 on success, 0
   otherwise) → `q_min=0, q_max=1/(1-discount)` as a starting point. Verify
   against existing Robomimic image-env bounds for consistency.

3. **`pyproject.toml`** — add `libero` and `robosuite` to deps (or a
   `libero` extras group) so `uv sync` provisions the LIBERO env. If
   conflicts with OGPO's pinned deps, install LIBERO into the venv as an
   editable sibling repo and document in the script.

### Dependencies and assets

- **LIBERO install**: `pip install libero-robosuite` plus the `libero`
  package itself. Note: LIBERO pins robosuite to a specific version; check
  compatibility with OGPO's current robosuite (Robomimic uses robosuite too).
  If versions conflict, isolating LIBERO behind `if env_name.startswith
  ("libero-")` keeps Robomimic runs unaffected, but the runtime venv has to
  pick one robosuite — this is the highest-risk item.
- **LIBERO benchmark assets**: BDDL files and demonstration HDF5s ship with
  the `libero` package; verify after install via `get_libero_path("datasets")`.
- **EGL / MuJoCo**: already set up by the Robomimic scripts
  (`MUJOCO_GL=egl`); LIBERO uses the same backend.

---

## Risks / open questions

1. **robosuite version conflict** between Robomimic (used by
   `square_pg_libero.sh`) and LIBERO. Mitigation: pick one robosuite, smoke
   test both env families in a single venv before declaring done.
2. **Camera resolution**: PaliGemma SigLIP expects 224×224. LIBERO renders
   at whatever `camera_heights/widths` we pass; set to 224 explicitly. The
   demonstration HDF5s may have been recorded at a different resolution — if
   so, we'll need to either rerender demos via offline rollouts of the
   ground-truth actions, or resize on load (resize will lose detail but is
   the pragmatic first pass).
3. **Single task vs. multi-task**: `square_pg_libero.sh` is single-task.
   Start with single-task LIBERO (`libero_90-0`) to keep the diff minimal;
   multi-task can be a follow-up. The VLA launcher hints at multi-task
   (`["libero_90_1-14", "libero_90_16-89"]`) but that's its concern.
4. **Language conditioning**: LIBERO tasks have natural-language goals
   (`task.language`). The PaliGemma encoder in OGPO is being used as a pure
   vision tower (no text prompt in the current path — confirm against
   `ogpo/networks/encoders/paligemma.py`). For a first pass, ignore language
   — single-task means the goal is implicit. Flag for future work if
   multi-task perf is poor.
5. **Reward shape**: LIBERO returns terminal 1/0. OGPO's `process_train_dataset`
   handles sparse rewards (`sparse=False` is set in the script but the env
   reward function is what matters); verify the `rew_fn` path doesn't try to
   reshape LIBERO rewards.
6. **Action-chunk horizon**: OGPO uses `horizon_length=4`. LIBERO demos are
   single-step; OGPO's `process_train_dataset` should chunk them. Confirm it
   handles LIBERO without a chunk-specific code path.

---

## Implementation order

1. Install LIBERO into the OGPO venv; verify `OffScreenRenderEnv` runs and
   demo HDF5s load. **Gate**: a 5-line script that resets the env and reads
   one demo without crashing.
2. Write `envs/libero_utils.py` (env + dataset). **Gate**: standalone unit
   that calls `make_env_and_datasets("libero-libero_90-0", ...)` and prints
   shapes matching the Robomimic image-env contract.
3. Add the `env_name.startswith("libero-")` branch in `env_utils.py` and the
   `q_bounds` branch.
4. Smoke-test the BC-only path: copy `square_pg_libero.sh`, set
   `online_steps=0`, `bc_pi_steps=2000`, `eval_episodes=2`. **Gate**: BC
   training runs to completion with PaliGemma pre-encoding, env eval calls
   succeed, no shape mismatches.
5. Author `scripts/ogpo/libero_pg.sh` with full hyperparams; launch a short
   end-to-end run (BC + a few thousand online steps) as a sanity check.

Each gate is small and self-contained; failures localize to one component.

---

## Out of scope (explicitly)

- Any change to OGPO agents, runners, networks, or PaliGemma encoder.
- Multi-task or language-conditioned LIBERO.
- Online async rollout (VLA-style); we use OGPO's existing single-process
  online runner.
- Tuning hyperparameters for LIBERO; start with `square_pg_libero.sh`'s
  values and only revisit if the smoke test reveals dim/scale mismatches.
