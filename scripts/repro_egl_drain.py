#!/usr/bin/env python
"""Fast repro for the MuJoCo/EGL offscreen-framebuffer crash + node-drain.

Skips ~2h of training and reproduces the failure directly:
  1. Pin each visible GPU to `--frac` VRAM with JAX (exactly like the real run,
     which sets XLA_PYTHON_CLIENT_MEM_FRACTION and preallocates).
  2. Build the LIBERO offscreen render envs via the project's make_env_libero,
     spread across GPUs by render_gpu_device_id = rank % num_devices.
  3. reset() -> triggers MjrContext build -> expect mujoco.FatalError 0x8cdd.
  4. Time close() -> with FIX 1 this must return within ~20s/worker.
  5. Report any leaked GPU worker processes.

Usage (on an interactive GPU node, same env as the runscript):
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    uv run scripts/repro_egl_drain.py --frac 0.95 --env-num 8       # expect crash
    uv run scripts/repro_egl_drain.py --frac 0.75 --env-num 8       # expect pass?
    uv run scripts/repro_egl_drain.py --no-pin  --env-num 8         # control (headroom)
"""
import argparse
import os
import sys
import time
import types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.95,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION to pin per GPU (mimics real run)")
    ap.add_argument("--no-pin", action="store_true",
                    help="skip JAX VRAM pinning (control: full headroom)")
    ap.add_argument("--env-num", type=int, default=8,
                    help="number of offscreen envs (real run: env_num + eval_env_num = 16)")
    ap.add_argument("--num-devices", type=int, default=0,
                    help="devices for rank %% num_devices (0 = number of visible GPUs)")
    ap.add_argument("--resolution", type=int, default=256, help="camera H=W (config default 256)")
    ap.add_argument("--task", type=str, default="libero_90_0")
    args = ap.parse_args()

    # --- render env vars (fallback if the shell didn't already export them) ---
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", cvd.split(",")[0])
    n_visible = len([x for x in cvd.split(",") if x != ""])
    num_devices = args.num_devices or max(n_visible, 1)
    print(f"[repro] CUDA_VISIBLE_DEVICES={cvd}  n_visible={n_visible}  "
          f"num_devices={num_devices}  MUJOCO_EGL_DEVICE_ID={os.environ['MUJOCO_EGL_DEVICE_ID']}",
          flush=True)

    # --- Stage 1: pin VRAM with JAX, exactly like the training process ---------
    if not args.no_pin:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.frac)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        print(f"[repro] pinning VRAM: XLA_PYTHON_CLIENT_MEM_FRACTION={args.frac}", flush=True)
        import jax
        import jax.numpy as jnp
        devs = jax.devices()
        print(f"[repro] jax devices: {devs}", flush=True)
        # Force preallocation on each device (first op grabs `frac` of the GPU).
        _pinned = [jax.device_put(jnp.ones((1024, 1024)), d) for d in devs]
        for x in _pinned:
            x.block_until_ready()
        print(f"[repro] VRAM pinned on {len(devs)} device(s).", flush=True)
    else:
        print("[repro] --no-pin: skipping VRAM pinning (control run).", flush=True)

    # --- Stage 2: build the LIBERO offscreen envs (imports after jax, as real run) ---
    from src.envs.libero import make_env_libero
    from src.envs.venv import SubprocVectorEnv

    # Minimal stub config: make_env_libero only reads env_resolution + num_steps_wait.
    config = types.SimpleNamespace(
        collect=types.SimpleNamespace(env_resolution=args.resolution, num_steps_wait=10)
    )
    env_fn = make_env_libero(config, tasks=[args.task], num_devices=num_devices)
    env_fns = [(lambda rank=i: env_fn(rank)) for i in range(args.env_num)]

    print(f"[repro] building SubprocVectorEnv with {args.env_num} envs "
          f"(resolution={args.resolution})...", flush=True)
    crashed = False
    venv = None
    try:
        venv = SubprocVectorEnv(env_fns)
        print("[repro] reset() -> builds MjrContext offscreen framebuffer...", flush=True)
        venv.reset()
        print("[repro] RESET OK -- no crash. (framebuffer allocated with headroom)", flush=True)
    except BaseException as e:  # mujoco.FatalError is not always an Exception subclass
        crashed = True
        print(f"[repro] *** CRASH during build/reset: {type(e).__name__}: {e}", flush=True)

    # --- Stage 4: time the teardown (FIX 1 = bounded) --------------------------
    if venv is not None:
        print("[repro] calling venv.close() -- timing teardown (FIX 1 must bound this)...",
              flush=True)
        t0 = time.time()
        try:
            venv.close()
        except BaseException as e:
            print(f"[repro] close() raised {type(e).__name__}: {e}", flush=True)
        dt = time.time() - t0
        print(f"[repro] close() returned in {dt:.1f}s "
              f"({'OK bounded' if dt < 30 else 'TOO SLOW -- FIX 1 not effective'})", flush=True)

    # --- Stage 5: report leaked worker processes -------------------------------
    time.sleep(1)
    leaked = os.popen(
        "ps -eo pid,cmd | grep -i 'repro_egl_drain\\|multiprocessing' "
        "| grep -v grep | grep -v %d" % os.getpid()
    ).read().strip()
    print("[repro] leaked child procs:\n" + (leaked if leaked else "  (none)"), flush=True)

    print(f"[repro] DONE. crashed={crashed}  frac={'none' if args.no_pin else args.frac}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
