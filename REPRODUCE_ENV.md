# REPRODUCE_ENV.md — exact environment reproduction

*Companion to AGENTS.md (which covers experiments/operations). This file is only
about going from a bare machine to a working training environment, with every
version pin and failure mode we actually encountered (Jul–Aug 2026).*

## 0. Reference hardware/software matrix (what this was validated on)

| Component | Known-good |
|---|---|
| OS | Ubuntu 22.04 (SkyPilot `skypilot-gpu:latest` ECR image) |
| GPU | H200 (EGL render) and B200 (OSMesa render); 1 GPU per run, ~110GB VRAM used |
| Python | 3.12 (created by uv from `pyproject.toml`) |
| JAX | pinned via `uv.lock` (CUDA 12 wheels) |
| Driver | 580.126.20 (B200 node) — any driver with CUDA 12 support worked |
| Disk | ≥150GB free for one run (weights cache ~24GB + checkpoints ~12GB/save + buffers) |

## 1. System packages (required BEFORE `uv sync`)

```bash
sudo apt-get update && sudo apt-get install -y \
  cmake                    # egl-probe builds from source; uv sync FAILS without it \
  libglvnd0 libegl1 libgl1 libopengl0 libgles2 \  # GLVND dispatch layer (EGL render) \
  libglib2.0-0             # libgthread — opencv/robosuite import fails without it \
  libosmesa6 libosmesa6-dev  # CPU rendering fallback (see §5)
```
Observed failures if skipped: `uv sync` → "hf-egl-probe … build environment"
error (cmake); env creation → `ImportError: libgthread-2.0.so.0` (glib);
OSMesa launch → silent GL context failure (libosmesa6).

## 2. Repos and pins

```bash
git clone <vla-post-training remote> ~/vla-post-training
cd ~/vla-post-training
git checkout shashwat/stability-study        # the study branch (all flags default-off)

# Submodules are declared with SSH URLs in .gitmodules; simplest is direct clone:
git clone <openpi remote> openpi
(cd openpi && git checkout 11f08089af6e507c4e76e193ba42a3ded217113e)   # REQUIRED pin
git clone <molmospaces remote> molmospaces                             # optional for LIBERO
(cd molmospaces && git checkout 2a828d549307f5d2e1d5384dc0662b3a1666954a) || mkdir -p molmospaces
```
The openpi pin matters: the learner reaches into `openpi/src/openpi` internals
(sampling, policy config); other commits may break the split-jit interfaces.

## 3. Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"    # nohup'd/ssh shells miss ~/.local/bin — export in scripts
cd ~/vla-post-training && uv sync       # ~5-10 min; creates .venv (Python 3.12)
```
Everything runs as `uv run <script>` from the repo root. PYTHONPATH for the
submodules is set inside `scripts/stability_study.sh` — do not rely on the
shell env.

## 4. LIBERO: the two mandatory manual steps

**4a. Config file (else: headless run hangs/crashes on an interactive prompt
buried in `libero/__init__.py` asking "custom dataset folder? (Y/N)"):**
```bash
mkdir -p run_store/libero
V=$PWD/.venv/lib/python3.12/site-packages/libero/libero
cat > run_store/libero/config.yaml <<EOF
benchmark_root: $V
bddl_files: $V/./bddl_files
init_states: $V/./init_files
datasets: $V/../datasets
assets: $V/./assets
EOF
```
(`LIBERO_CONFIG_PATH` is pointed at `run_store/libero` by the launcher.)

**4b. Scene assets (~408MB).** Auto-download from HuggingFace triggers on first
env creation — but **HF rate-limits datacenter IPs** ("We had to rate limit
your IP…" → `FileNotFoundError: …libero_kitchen_tabletop_base_style.xml`).
Reliable path: copy the cache from any machine that has it:
```bash
# on a working machine:  tar czf /tmp/libero_assets.tgz -C ~/.cache/libero assets
# on the new machine:
mkdir -p ~/.cache/libero && tar xzf libero_assets.tgz -C ~/.cache/libero
ln -sfn ~/.cache/libero/assets $V/assets
```

## 5. Rendering backend

- **EGL (GPU, default)**: needs `libEGL_nvidia.so` (`ldconfig -p | grep EGL_nvidia`).
  The launcher auto-discovers/creates the GLVND vendor JSON — including the case
  where only Mesa's ICD is registered (it writes its own `10_nvidia.json` under
  `run_store/egl/`). If the NVIDIA EGL lib is entirely absent from the container
  (true on some pods), EGL CANNOT work — use OSMesa.
- **OSMesa (CPU)**: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` env vars at
  launch. ~2× slower collection; training math identical. Verify with:
  `uv run python -c "import mujoco; g=mujoco.GLContext(64,64); g.make_current(); print('OK')"`
- Known transient: robosuite EGL contexts can die mid-run with
  `EGLError: EGL_NOT_INITIALIZED` during env teardown (kills the run).
  Rare; relaunch (runs checkpoint every 10k) or switch that run to OSMesa.

## 6. Model weights

`gs://openpi-assets/checkpoints/pi05_libero` (params + norm-stats assets),
auto-downloaded to `run_store/cache/openpi/` on first run (~24GB incl. cache;
needs GCS egress). All caches (HF/torch/wandb/etc.) are kept INSIDE
`run_store/` by the launcher — nothing writes to $HOME.

## 7. Validation gauntlet (run in order)

```bash
# 1. Unit/equivalence suite — MUST be 15/15 (CPU, ~6 min):
JAX_PLATFORMS=cpu uv run pytest tests/ogpo/ -x -q

# 2. GL context (pick your backend, §5).

# 3. Smoke run — baseline recipe, expect "Progress on: …/100kit" within ~8 min
#    (first 60-episode collection at ~10-20s/ep, then training steps):
env GPU=0 ARM=smoke SEED=0 nohup bash scripts/stability_study.sh > ~/stab_smoke.log 2>&1 &
tr '\r' '\n' < ~/stab_smoke.log | grep -a "Progress on" | tail -1
# Healthy signs in metrics.jsonl within the first 1k steps: critic/q_loss
# decreasing from ~35, actor rows appearing after step 900, zero Tracebacks.
kill %1   # after verifying
```

## 8. Known environment failure modes (quick table)

| Symptom | Cause | Fix |
|---|---|---|
| `uv sync` fails on hf-egl-probe | no cmake | §1 |
| Run hangs at start, "(Y/N)" in log | LIBERO first-run prompt | §4a |
| `FileNotFoundError …base_style.xml` + HF rate-limit lines | assets download blocked | §4b |
| `libEGL warning: … dri2` + `PLATFORM_DEVICE` ImportError | no NVIDIA EGL in container | OSMesa (§5) |
| `libgthread-2.0.so.0` missing | glib not installed | §1 |
| `uv: command not found` in nohup'd script | PATH lacks ~/.local/bin | launcher exports it; do the same in custom scripts |
| GPU OOM at model init | another process holds the GPU | check `nvidia-smi --query-compute-apps` before launching |
| Pod wiped / "pod already exists" on relaunch | k8s ephemeral-storage eviction | prune `run_store` of finished runs; `sky down` then relaunch (AGENTS.md §6) |

## 9. Determinism notes

Same seed + same GPU arch reproduces trajectories closely but not bitwise
(cuDNN nondeterminism + env physics divergence chaos). Cross-arch (H200 vs
B200) and cross-renderer (EGL vs OSMesa) differ more; compare seeds
statistically, never step-by-step. The study's own seed-variance data
(AGENTS.md §5, reports/) is the calibration for what "same config" spread
looks like.
