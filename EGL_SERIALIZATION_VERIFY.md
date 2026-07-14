# Handoff: verify that SERIAL EGL context creation prevents the D-state node drain

Read this first, then `scripts/egl_safe_probe.py`, `scripts/egl_probe.sh`,
`mujoco_egl_drain_repro.txt`, and the memory note `vla-egl-drain-repro`.

## Why this note exists
`scripts/awr_libero_babel.sh` drains the maxlab node at the step-10000 collection
phase. We root-caused it (see below). We have a **hypothesis** that serializing
EGL context creation prevents the drain, but we could NOT test it because the
node we were on (`babel-q9-32`, 1x L40S 46GB) is now itself wedged from
reproducing the bug. **Run the verification below on a FRESH, throwaway GPU node**
(NOT maxlab — the test can drain the node it runs on).

## What is already established (do not re-derive)
- EGL offscreen rendering WORKS with VRAM headroom: `--pin-vram-frac 0.0`,
  8 contexts all build/reset/step and release the GPU cleanly. Confirmed live.
- The drain is a **driver-level D-state deadlock**, NOT a catchable crash:
  under `--pin-vram-frac 0.95` (~2.3GB free) + 8 CONCURRENT framebuffer allocs,
  workers wedge in `D` (uninterruptible) state inside the NVIDIA driver
  (`wchan = os_acquire_rwlock_write / os_acquire_semaphore`), **survive
  `kill -9`**, and `nvidia-smi` then hangs. Only a reboot recovers it. Confirmed
  live (that is what wedged babel-q9-32).
- Therefore the bounded `close_env()` teardown in `src/envs/venv.py:495`
  (commit 5d29037) is necessary but INSUFFICIENT: you cannot SIGKILL a D-state
  process. Prevention is the only reliable fix.
- Two joint triggers: (1) VRAM starvation, (2) many contexts contending the
  driver allocator lock CONCURRENTLY. Killing either should prevent the deadlock.

## Hypothesis to verify
**Creating EGL contexts one at a time (serially) prevents the D-state deadlock,
even under VRAM pressure** — because only one context is ever in the driver's
allocation critical section at once. Expected: under pressure, serial creation
yields either success or a *catchable* `mujoco.FatalError 0x8cdd` (killable),
but NOT a D-state wedge.

## The probe supports this directly
- `--serial` : event-based — parent waits for each worker to report `built`
  (context allocated, i.e. it left the driver allocator) before launching the
  next. This is the real serialization, not the time-based `--stagger`.
- `--hold-seconds N` : each worker keeps its context resident N seconds after
  stepping, so contexts COEXIST in VRAM the way the real 16-context run does,
  while creation stays serialized.
- `--build-timeout S` : if a context never reports `built` within S seconds it
  is classified as a **D-state deadlock** (the failure we're hunting).
- `--pin-vram-frac F` : JAX preallocates F of the GPU first, reproducing the
  real run's starvation.
- `--min-free-mib` / preflight : aborts BEFORE building if a GPU is starved, so
  the probe is safe by default. Serial-vs-concurrent tests under pressure need
  `--force` to intentionally push into the danger zone.

Every context is an isolated forked child; per-phase timeouts + a top-level
SIGALRM watchdog (`--deadline`) SIGKILL all children. The harness itself never
hangs — but note a D-state CHILD it spawns can still wedge the GPU (that's the
thing under test), so only run on an expendable node.

## Verification protocol (run on a fresh throwaway GPU node)

Setup (each run goes through the wrapper, which sets PYTHONPATH / LIBERO / EGL
vars exactly like the training script):
```bash
cd /home/mananaga/vla-post-training
# confirm GPU is healthy first — nvidia-smi must respond instantly:
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

Choose the pin fraction to match ~2.3GB free on the test GPU. On a 46GB card
that is ~0.93-0.95; on a 24GB card use ~0.88; on 80/96GB use ~0.97. Goal: leave
only a couple GB free so allocation is under real pressure.

### Run A — CONCURRENT baseline under pressure (EXPECT: drain)
Confirms the test GPU reproduces the wedge at all. On babel-q9-32 this drained
the node; skip if you accept the prior result and want to save a node.
```bash
scripts/egl_probe.sh --mode probe --env-num 8 --pin-vram-frac 0.95 \
  --hold-seconds 30 --force --env-timeout 180 --deadline 300
```
- WEDGE signature: some ranks report `wedged`; after the run, in another shell:
  `ps -eo pid,stat,wchan,cmd | grep egl_safe_probe | grep ' D '`
  shows D-state procs stuck in `os_acquire_*`; `nvidia-smi` hangs. Node is drained.

### Run B — SERIAL under the SAME pressure (KEY TEST; EXPECT: no D-state)
This is the hypothesis. Use `--serial` + `--hold-seconds` so contexts coexist
but are created one at a time.
```bash
scripts/egl_probe.sh --mode probe --env-num 8 --pin-vram-frac 0.95 \
  --serial --hold-seconds 30 --force --build-timeout 60 \
  --env-timeout 300 --deadline 420
```
Interpretation:
- PASS (hypothesis holds): every rank reaches `built` (then `reset`/`stepped`/
  `holding`), OR fails with `fatal` (0x8cdd, catchable). Summary shows NO
  `wedged`. After the run `ps ... grep ' D '` shows NO D-state procs and
  `nvidia-smi` still responds. => serialization prevents the deadlock.
- FAIL: probe prints "did NOT report 'built' ... D-STATE DEADLOCK during serial
  allocation" and/or `ps` shows D-state procs. => serialization is insufficient;
  fall back to VRAM headroom / GPU isolation as the primary fix.

### Run C — SERIAL with headroom (sanity; EXPECT: clean pass, node fine)
```bash
scripts/egl_probe.sh --mode probe --env-num 16 --num-devices 1 \
  --serial --hold-seconds 20 --pin-vram-frac 0.0
```
All 16 should reach `holding`/`done`, node stays healthy.

### After EVERY run, verify the node is clean before reusing it:
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader   # must respond
ps -eo pid,stat,wchan,cmd | grep -E 'egl_safe_probe|vram_pin' | grep -v grep
# any process in state 'D' stuck in os_acquire_* => node is wedged, needs reboot
```

## If Run B PASSES -> implement the fix in the real code path
Serialize the actual context creation (not just in the probe):
- `src/envs/libero.py` `LiberoWrapper.__init__` and the `env.close()+reopen` on
  task change (line ~41-43): wrap the `OffScreenRenderEnv(**args)` call in a
  cross-process lock (a `multiprocessing`/file lock shared by all render
  workers) so only one context is allocated at a time. Env creation is one-time
  per rollout, so the serialization cost is a few seconds.
- Keep as belt-and-suspenders: worker-side VRAM preflight that raises a normal
  Python exception (killable) instead of entering the driver alloc when free
  VRAM < floor, so any residual failure is reaped by the bounded `close_env()`.
- Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` (or 0.70). If a spare GPU exists,
  dedicate it to rendering (point `render_gpu_device_id` / `MUJOCO_EGL_DEVICE_ID`
  at a GPU JAX's CUDA_VISIBLE_DEVICES excludes) — starvation then impossible.

## Then, before the real maxlab run (safe — cannot drain):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/egl_probe.sh --mode probe \
  --env-num 16 --num-devices 4 --pin-vram-frac 0.75 --serial --hold-seconds 20
scripts/egl_probe.sh --mode teardown --env-num 8 --pin-vram-frac 0.75
```
Preflight aborts (does not wedge) if headroom is short. If both pass on maxlab
hardware, the real collection phase does the same thing with the same headroom.
```
