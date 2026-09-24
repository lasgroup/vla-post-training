#!/usr/bin/env python
"""
Safe standalone probe for the MuJoCo/EGL offscreen-render crash + node-drain.

Background (see mujoco_egl_drain_repro.txt):
  awr_libero_babel.sh drains the Slurm node at the step-10000 collection phase.
  JAX preallocates most VRAM -> robosuite/MuJoCo cannot build the EGL offscreen
  framebuffer -> `mujoco.FatalError 0x8cdd (GL_FRAMEBUFFER_UNSUPPORTED)` -> the
  render worker wedges in a C-level EGL teardown -> the vector env's close()
  hung unbounded -> unkillable step -> node drained.

Why this script cannot drain the node
--------------------------------------
  1. Every EGL context is created in a *forked child process*, never in this
     process. If a child wedges in a driver call, it is isolated.
  2. Every phase (build / reset / step / close) is bounded by an explicit
     timeout. A child that overruns is reaped with terminate() -> kill().
  3. A top-level SIGALRM watchdog (--deadline) is the ultimate backstop: when it
     fires it SIGKILLs every child we spawned and os._exit()s. Nothing in this
     harness ever does an unbounded recv()/join().
  4. On the fatal framebuffer error, workers call os._exit() immediately instead
     of unwinding through EGL destructors (the unwind is what wedges, because
     mujoco.FatalError subclasses Exception and the stock worker catches it and
     returns, running EGL teardown at interpreter shutdown).

Modes
-----
  probe    (default) Build N EGL offscreen render contexts concurrently, reset
           and step each, report which built/rendered/crashed/wedged. With
           --pin-vram-frac > 0 a JAX process first preallocates that fraction of
           the render GPU (mimicking XLA_PYTHON_CLIENT_MEM_FRACTION) so you can
           reproduce 0x8cdd on demand and find the safe headroom.

  teardown Build a real src.envs.venv.SubprocVectorEnv of N LIBERO envs (the
           exact class whose close_env() drains the node), reset it, then time
           close() and check for leaked GPU processes. Validates the bounded
           SIGTERM->SIGKILL teardown fix.

Typical usage (via scripts/egl_probe.sh, which sets PYTHONPATH / LIBERO paths):
  # 1) Does EGL offscreen rendering work at all on this node, with headroom?
  scripts/egl_probe.sh --mode probe --env-num 8 --pin-vram-frac 0.0
  # 2) Reproduce the 0x8cdd crash by starving VRAM the way the real run does:
  scripts/egl_probe.sh --mode probe --env-num 8 --pin-vram-frac 0.95
  # 3) Find the safe fraction (expect pass ~0.75):
  scripts/egl_probe.sh --mode probe --env-num 8 --pin-vram-frac 0.75
  # 4) Validate teardown never hangs (crash or not, close() returns quickly):
  scripts/egl_probe.sh --mode teardown --env-num 8 --pin-vram-frac 0.95
"""

# --- EGL env vars MUST be set before importing mujoco/libero (any child inherits) ---
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)
# First visible CUDA index -> the physical GPU EGL renders on (matches runscript).
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0] if _cvd else "0")

import argparse
import multiprocessing as mp
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

# fork keeps children lightweight and lets them inherit the EGL env above without
# re-pickling. We deliberately do NOT import jax/mujoco/libero in this parent for
# `probe` mode, so forked render children start from a clean slate.
_CTX = mp.get_context("fork")

# Registry of everything we spawn, so the watchdog can guarantee cleanup.
_CHILDREN: list = []


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _nvidia_smi(query: str, timeout: float = 8.0):
    """Run nvidia-smi WITHOUT ever blocking on it. On a wedged driver nvidia-smi
    itself goes into unkillable D-state, so we poll and, on timeout, abandon the
    child (best-effort kill, never wait()) and return None."""
    try:
        proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except Exception:  # noqa: BLE001
        return None
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            try:
                return proc.stdout.read().strip()
            except Exception:  # noqa: BLE001
                return None
        time.sleep(0.2)
    try:
        proc.kill()  # best-effort; may be ignored if D-state
    except Exception:  # noqa: BLE001
        pass
    return None  # treated as "wedged / unknown"


def gpu_snapshot(tag: str) -> None:
    out = _nvidia_smi("index,memory.used,memory.free,utilization.gpu")
    print(f"[gpu:{tag}] {out if out is not None else '(nvidia-smi unresponsive -- driver may be wedged)'}",
          flush=True)


def free_mib_per_gpu(timeout: float = 8.0):
    """Return {index: free_MiB} for all GPUs, or None if nvidia-smi hangs/fails
    (a hung nvidia-smi itself means the driver is already wedged)."""
    out = _nvidia_smi("index,memory.free", timeout=timeout)
    if out is None:
        return None
    res = {}
    for line in out.splitlines():
        try:
            idx, free = line.split(",")
            res[int(idx)] = int(free)
        except ValueError:
            pass
    return res


def preflight_headroom(min_free_mib: int, force: bool) -> bool:
    """Refuse to build EGL contexts when a visible GPU is starved. This is what
    makes the probe safe to run on maxlab: below the floor it ABORTS cleanly
    instead of driving the driver into an unkillable D-state deadlock."""
    free = free_mib_per_gpu()
    if free is None:
        print("[preflight] ABORT: nvidia-smi did not respond in time -- the GPU "
              "driver may already be wedged. Not building any EGL context.",
              flush=True)
        return force
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [int(x) for x in cvd.split(",") if x != ""] or sorted(free)
    starved = {i: free[i] for i in visible if i in free and free[i] < min_free_mib}
    print(f"[preflight] free MiB per visible GPU: "
          f"{ {i: free.get(i) for i in visible} }  (floor={min_free_mib})",
          flush=True)
    if starved and not force:
        print(f"[preflight] ABORT: GPU(s) {starved} below the {min_free_mib} MiB "
              "headroom floor. Building offscreen framebuffers now risks the "
              "unkillable D-state driver deadlock that drains the node. Lower "
              "XLA_PYTHON_CLIENT_MEM_FRACTION / --pin-vram-frac, or pass --force.",
              flush=True)
        return False
    if starved:
        print(f"[preflight] WARNING: GPU(s) {starved} below floor but --force "
              "set; proceeding (this node may drain).", flush=True)
    return True


def list_leaked_procs() -> str:
    """Any of our children still alive counts as a leak."""
    leaked = []
    for c in _CHILDREN:
        try:
            if c.is_alive():
                leaked.append(c.pid)
        except Exception:  # noqa: BLE001
            pass
    return ", ".join(str(p) for p in leaked) if leaked else "none"


def kill_all_children(sig=signal.SIGKILL) -> None:
    for c in _CHILDREN:
        try:
            if c.is_alive():
                os.kill(c.pid, sig)
        except (ProcessLookupError, OSError, AttributeError):
            pass


def _watchdog(signum, frame):  # noqa: ARG001
    print("\n[WATCHDOG] global deadline hit -> SIGKILL all children and exit.",
          file=sys.stderr, flush=True)
    print(f"[WATCHDOG] children still alive before kill: {list_leaked_procs()}",
          file=sys.stderr, flush=True)
    kill_all_children(signal.SIGKILL)
    gpu_snapshot("watchdog")
    os._exit(3)


# --------------------------------------------------------------------------- #
# VRAM pinning (reproduce the starvation the real run causes)
# --------------------------------------------------------------------------- #
def _vram_pin_worker(frac: float, ready_conn) -> None:
    """Preallocate `frac` of the render GPU with JAX, then idle until killed."""
    egl_dev = os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = egl_dev  # pin JAX to the render GPU
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(frac)
    try:
        import jax
        import jax.numpy as jnp
        # Touch a device array to force the preallocation to actually happen.
        x = jnp.ones((1024, 1024), dtype=jnp.float32)
        _ = float((x @ x.T).sum())
        ready_conn.send(("pinned", jax.devices()[0].device_kind))
    except Exception as e:  # noqa: BLE001
        ready_conn.send(("pin_failed", repr(e)))
        os._exit(1)
    ready_conn.close()
    # Hold the memory. Parent kills us when the probe is done.
    while True:
        time.sleep(3600)


def start_vram_pin(frac: float, timeout: float = 120.0):
    if frac <= 0:
        return None
    parent_conn, child_conn = _CTX.Pipe()
    p = _CTX.Process(target=_vram_pin_worker, args=(frac, child_conn), daemon=True)
    p.start()
    _CHILDREN.append(p)
    child_conn.close()
    if parent_conn.poll(timeout):
        status, detail = parent_conn.recv()
        if status == "pinned":
            print(f"[pin] JAX preallocated ~{frac:.2f} of render GPU ({detail}).",
                  flush=True)
        else:
            print(f"[pin] WARNING: pin failed: {detail}", flush=True)
    else:
        print(f"[pin] WARNING: pin did not confirm within {timeout}s "
              "(continuing anyway).", flush=True)
    gpu_snapshot("after-pin")
    return p


# --------------------------------------------------------------------------- #
# probe mode: isolated EGL render workers
# --------------------------------------------------------------------------- #
def _render_worker(rank, num_devices, resolution, num_steps_wait, task, n_steps,
                   conn, hold_seconds=0.0) -> None:
    """Build ONE LIBERO offscreen env exactly like the training run, render a
    few steps, optionally HOLD the context alive (so many contexts coexist while
    creation is serialized), then die hard. Never unwinds EGL."""
    import numpy as np

    def emit(stage, extra=""):
        try:
            conn.send((rank, stage, extra))
        except Exception:  # noqa: BLE001
            pass

    try:
        import mujoco  # noqa: F401  (for FatalError type)
        from src.envs.libero import make_env_libero

        cfg = SimpleNamespace(
            collect=SimpleNamespace(
                env_resolution=resolution,
                num_steps_wait=num_steps_wait,
                episode_steps_multiplier=1,
            )
        )
        env_fn = make_env_libero(cfg, tasks=[task], num_devices=num_devices)
        emit("building")
        env = env_fn(rank)  # <-- OffScreenRenderEnv build == EGL context alloc
        emit("built")
        obs, info = env.reset(options={"task_id": task})
        emit("reset")
        # LIBERO is a 7-DoF action (6 pose + gripper). Don't query action_space:
        # the raw OffScreenRenderEnv doesn't expose one.
        try:
            act = np.zeros(env.action_space.shape, dtype=np.float32)
        except Exception:  # noqa: BLE001
            act = np.zeros(7, dtype=np.float32)
        for _ in range(n_steps):
            env.step(act)
        emit("stepped")
        if hold_seconds > 0:
            # Keep the EGL context resident so that, in --serial mode, later
            # contexts are allocated while this one still occupies VRAM -- i.e.
            # the same coexistence the real 16-context run has, but with
            # creation serialized.
            emit("holding")
            time.sleep(hold_seconds)
        # Success. Do NOT close()/unwind EGL — dying releases the context safely.
        conn.close()
        os._exit(0)
    except mujoco.FatalError as e:  # the 0x8cdd framebuffer crash
        emit("fatal", str(e).splitlines()[0][:200])
        conn.close()
        os._exit(42)  # <-- critical: skip EGL destructors that wedge the driver
    except Exception as e:  # noqa: BLE001
        emit("error", repr(e)[:200])
        conn.close()
        os._exit(1)


def run_probe(args) -> int:
    print(f"[probe] launching {args.env_num} EGL render contexts concurrently "
          f"(num_devices={args.num_devices}, res={args.resolution}, "
          f"task={args.task}, steps={args.steps})", flush=True)
    gpu_snapshot("start")
    start_vram_pin(args.pin_vram_frac)

    if not preflight_headroom(args.min_free_mib, args.force):
        print("[probe] aborted before building any EGL context (node safe).",
              flush=True)
        return 2

    mode = ("SERIAL (wait for each 'built' before next)" if args.serial
            else f"CONCURRENT (stagger={args.stagger}s)")
    print(f"[probe] context-creation mode: {mode}; hold={args.hold_seconds}s",
          flush=True)

    status = {}
    detail = {}
    workers = []  # (rank, process, parent_conn)

    def _launch(rank):
        parent_conn, child_conn = _CTX.Pipe()
        p = _CTX.Process(
            target=_render_worker,
            args=(rank, args.num_devices, args.resolution, args.num_steps_wait,
                  args.task, args.steps, child_conn, args.hold_seconds),
            daemon=True,
        )
        p.start()
        _CHILDREN.append(p)
        child_conn.close()
        status[rank] = "spawned"
        detail[rank] = ""
        workers.append([rank, p, parent_conn])
        return workers[-1]

    def _drain(conn, rank):
        try:
            while conn.poll(0):
                _r, stage, extra = conn.recv()
                status[rank] = stage
                if extra:
                    detail[rank] = extra
        except (EOFError, OSError):
            pass

    _BUILT = ("built", "reset", "stepped", "holding", "done", "fatal", "error")

    if args.serial:
        # Allocate ONE context at a time: wait until this worker reports it has
        # built its EGL context (i.e. left the driver allocator) before spawning
        # the next. If a build wedges (alive but no 'built' past the timeout),
        # that IS the deadlock -> reap and stop; serialization did not save us.
        wedged_during_build = False
        for rank in range(args.env_num):
            _, p, conn = _launch(rank)
            bdl = time.time() + args.build_timeout
            while time.time() < bdl:
                _drain(conn, rank)
                if status[rank] in _BUILT or not p.is_alive():
                    break
                time.sleep(0.2)
            if status[rank] not in _BUILT and p.is_alive():
                status[rank] = "wedged"
                print(f"[probe] rank {rank} did NOT report 'built' within "
                      f"{args.build_timeout}s -> D-STATE DEADLOCK during serial "
                      "allocation. Serialization insufficient.", flush=True)
                wedged_during_build = True
                break
            print(f"[probe] rank {rank}: {status[rank]}"
                  + (f" ({detail[rank]})" if detail[rank] else ""), flush=True)
        if wedged_during_build:
            print("[probe] stopping early; not launching remaining contexts.",
                  flush=True)
    else:
        for rank in range(args.env_num):
            _launch(rank)
            if args.stagger > 0 and rank < args.env_num - 1:
                time.sleep(args.stagger)

    deadline = time.time() + args.env_timeout

    # Poll all worker pipes until each terminates or the per-run deadline passes.
    while time.time() < deadline:
        alive = False
        for rank, p, conn in workers:
            if status[rank] in ("done", "fatal", "error", "wedged"):
                continue
            try:
                while conn.poll(0):
                    _r, stage, extra = conn.recv()
                    status[rank] = stage
                    if extra:
                        detail[rank] = extra
            except (EOFError, OSError):
                pass
            if not p.is_alive():
                code = p.exitcode
                if status[rank] == "stepped" or code == 0:
                    status[rank] = "done"
                elif code == 42:
                    status[rank] = "fatal"
                elif status[rank] not in ("fatal", "error"):
                    status[rank] = "error"
                    detail[rank] = detail[rank] or f"exitcode={code}"
            else:
                alive = True
        if not alive:
            break
        time.sleep(0.5)

    # Reap anything that overran the deadline (the wedge case).
    for rank, p, _ in workers:
        if p.is_alive():
            status[rank] = "wedged"
            print(f"[probe] rank {rank} overran {args.env_timeout}s -> "
                  "terminate()->kill()", flush=True)
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join(5)

    # Report.
    counts = {}
    print("\n[probe] per-context result:", flush=True)
    for rank, _, _ in workers:
        s = status[rank]
        counts[s] = counts.get(s, 0) + 1
        line = f"    rank {rank:2d}: {s}"
        if detail[rank]:
            line += f"  ({detail[rank]})"
        print(line, flush=True)
    print(f"\n[probe] summary: {counts}", flush=True)
    gpu_snapshot("end")
    print(f"[probe] leaked child procs: {list_leaked_procs()}", flush=True)

    ok = counts.get("done", 0) == args.env_num
    if counts.get("wedged"):
        print("[probe] NOTE: wedged contexts detected -- these are the ones that "
              "would drain a real node. On the real run the bounded close_env() "
              "must reap them; validate with --mode teardown.", flush=True)
    if counts.get("fatal"):
        print("[probe] NOTE: 0x8cdd framebuffer crash reproduced -- reduce "
              "--pin-vram-frac (or XLA_PYTHON_CLIENT_MEM_FRACTION) for headroom.",
              flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# teardown mode: validate SubprocVectorEnv.close() is bounded
# --------------------------------------------------------------------------- #
def run_teardown(args) -> int:
    print(f"[teardown] building real SubprocVectorEnv with {args.env_num} envs "
          "and timing close()", flush=True)
    gpu_snapshot("start")
    start_vram_pin(args.pin_vram_frac)

    if not preflight_headroom(args.min_free_mib, args.force):
        print("[teardown] aborted before building any EGL context (node safe).",
              flush=True)
        return 2

    from src.envs.libero import make_env_libero
    from src.envs.venv import SubprocVectorEnv

    cfg = SimpleNamespace(
        collect=SimpleNamespace(
            env_resolution=args.resolution,
            num_steps_wait=args.num_steps_wait,
            episode_steps_multiplier=1,
        )
    )
    env_fn = make_env_libero(cfg, tasks=[args.task], num_devices=args.num_devices)
    env_fns = [(lambda r=r: env_fn(r)) for r in range(args.env_num)]

    venv = None
    try:
        venv = SubprocVectorEnv(env_fns)
        print("[teardown] vector env constructed; resetting (builds EGL "
              "contexts in workers)...", flush=True)
        try:
            venv.reset(options={"task_id": args.task})
            print("[teardown] reset OK (no crash).", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[teardown] reset raised {type(e).__name__}: "
                  f"{repr(e)[:200]}", flush=True)
            print("[teardown] this is the crash path -- now timing close()...",
                  flush=True)
        gpu_snapshot("before-close")
        t0 = time.time()
        venv.close()
        dt = time.time() - t0
        print(f"\n[teardown] close() returned in {dt:.1f}s", flush=True)
        max_expected = 20.0 * args.env_num  # ~15s/worker budget + slack
        if dt <= max_expected:
            print(f"[teardown] PASS: bounded (<= {max_expected:.0f}s budget).",
                  flush=True)
            verdict = 0
        else:
            print(f"[teardown] FAIL: exceeded {max_expected:.0f}s budget -- "
                  "teardown is NOT bounded, would risk a drain.", flush=True)
            verdict = 1
    finally:
        # Belt and suspenders: the watchdog + this ensure no leaked workers.
        if venv is not None and not getattr(venv, "is_closed", True):
            try:
                venv.close()
            except Exception:  # noqa: BLE001
                pass
    gpu_snapshot("end")
    print(f"[teardown] leaked child procs: {list_leaked_procs()}", flush=True)
    return verdict


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["probe", "teardown"], default="probe")
    ap.add_argument("--env-num", type=int, default=8,
                    help="number of EGL contexts (real run: 8 collect + 8 eval)")
    ap.add_argument("--num-devices", type=int, default=1,
                    help="GPUs to spread render_gpu_device_id over "
                         "(1 here, 4 on maxlab)")
    ap.add_argument("--resolution", type=int, default=256,
                    help="camera H=W (config.collect.env_resolution)")
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--task", type=str, default="libero_90_44")
    ap.add_argument("--steps", type=int, default=5,
                    help="env.step()s after reset per context")
    ap.add_argument("--pin-vram-frac", type=float, default=0.0,
                    help="JAX preallocates this fraction of the render GPU "
                         "before building envs (0=off, 0.95=reproduce crash)")
    ap.add_argument("--min-free-mib", type=int, default=8000,
                    help="preflight headroom floor: abort (don't build) if any "
                         "visible GPU has less free VRAM than this")
    ap.add_argument("--force", action="store_true",
                    help="build even if below the headroom floor (may drain!)")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="seconds between launching each context; >0 serializes "
                         "creation to dodge the concurrent-alloc driver deadlock")
    ap.add_argument("--serial", action="store_true",
                    help="TRUE serialization: wait for each context to report "
                         "'built' before creating the next (event-based, not "
                         "time-based). Use with --hold-seconds to keep contexts "
                         "coexisting like the real run.")
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="each worker keeps its EGL context resident this long "
                         "after stepping, so many contexts coexist in VRAM")
    ap.add_argument("--build-timeout", type=float, default=60.0,
                    help="[--serial] if a context does not report 'built' within "
                         "this many seconds it is treated as a D-state deadlock")
    ap.add_argument("--env-timeout", type=float, default=180.0,
                    help="per-run budget for all probe workers to finish")
    ap.add_argument("--deadline", type=float, default=600.0,
                    help="hard global watchdog: SIGKILL everything after this")
    args = ap.parse_args()

    # Arm the ultimate backstop.
    signal.signal(signal.SIGALRM, _watchdog)
    signal.alarm(int(args.deadline))
    print(f"[main] mode={args.mode} deadline={args.deadline}s "
          f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)

    try:
        if args.mode == "probe":
            rc = run_probe(args)
        else:
            rc = run_teardown(args)
    finally:
        signal.alarm(0)
        kill_all_children(signal.SIGKILL)  # kill the pin process + any stragglers
    print(f"[main] done rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
