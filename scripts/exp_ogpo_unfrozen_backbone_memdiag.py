# ruff: noqa: E402
"""Diagnostic entry point: OGPO unfrozen-backbone run with an OPERATION/MEMORY
HISTORY log, to pinpoint exactly which operation OOMs.

What it does on top of scripts/exp_ogpo_unfrozen_backbone.py:

  * Writes an append-only JSONL history (env MEMDIAG_LOG) where every record
    carries a timestamp, the event/phase name, the training step, and the XLA
    allocator stats (bytes_in_use / peak_bytes_in_use / ...) per GPU. Because
    the file is line-buffered, the tail survives the crash: the LAST
    `phase_start` without a matching `phase_end` is the operation that OOM'd.
  * Instrumented phases (each logs mem before/after and catches the OOM):
      - init_train_state(jit)      : fp32 params + Adam state for the unfrozen
                                     backbone (first big allocation)
      - agent/awsft init brackets  : critic/value init, buffer, policy load
      - collect_data / evaluate_policy / save_epoch_state
      - start/end_data_collection  : EMA params moved on/off device
      - sample_action(collect)     : policy inference during rollouts
      - buffer_sample              : replay batch device_put
      - critic_update(jit)         : Q+V train step
      - policy_update(jit)         : the OGPO PPO step (expected OOM site)
      - ema_update(jit)
  * Every wrapped jitted call is jax.block_until_ready'd, so JAX's async
    dispatch cannot mis-attribute the failure to a later phase.
  * A daemon thread samples allocator stats every MEMDIAG_SAMPLE_SEC seconds
    (default 1.0; 0 disables) into the same JSONL, giving a continuous VRAM
    curve even *inside* a long jitted call.
  * Right before the FIRST policy update it saves a pprof snapshot of live
    device buffers (live_buffers_before_first_policy_update.prof) so the
    baseline occupancy (weights + opt state + EMA + batch) is separable from
    the policy-update transient that OOMs.

Everything is monkey-patched from the outside; nothing under src/ is edited
(same single-source-of-truth rationale as scripts/exp_ogpo_unfrozen_backbone.py).

Run via scripts/ogpo_libero_interactive_unfrozen_backbone_memdiag.sh.
"""
import os
import sys

# Make scripts/ importable so `import exp` resolves to scripts/exp.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importing exp performs the fragile process setup exactly once (warning
# filters, lerobot log filter, mp "spawn") and hands us the same main().
import exp as _exp
from exp_ogpo_unfrozen_backbone import _register_unfrozen_config

import itertools
import json
import logging
import threading
import time
import traceback
from datetime import datetime

import jax

import src.training.config as _config
import src.rl.filtered_sft_agent.filtered_sft_learner as _fsl
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner


_MEM_KEYS = (
    "bytes_in_use",
    "peak_bytes_in_use",
    "bytes_limit",
    "largest_alloc_size",
    "bytes_reserved",
    "num_allocs",
)


def _gib(n) -> float:
    return round(float(n) / 2**30, 3)


class MemTracer:
    """Append-only JSONL event log with XLA allocator stats per record."""

    def __init__(self, path: str, sample_sec: float = 1.0):
        self.path = path
        self.dir = os.path.dirname(path) or "."
        os.makedirs(self.dir, exist_ok=True)
        self._f = open(path, "a", buffering=1)  # line-buffered: survives crash
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._seq = itertools.count()
        self._sample_sec = sample_sec

    # -- memory ------------------------------------------------------------
    def _mem(self) -> dict:
        mem: dict = {}
        try:
            for i, d in enumerate(jax.local_devices()):
                stats = d.memory_stats() or {}
                mem[f"gpu{i}"] = {k: int(stats[k]) for k in _MEM_KEYS if k in stats}
        except Exception as e:  # noqa: BLE001 - diagnostics must never crash the run
            mem["error"] = repr(e)
        return mem

    # -- events ------------------------------------------------------------
    def event(self, ev: str, *, console: bool = True, **fields) -> None:
        mem = self._mem()
        rec = {
            "seq": next(self._seq),
            "t": round(time.monotonic() - self._t0, 3),
            "wall": datetime.now().isoformat(timespec="seconds"),
            "ev": ev,
            **fields,
            "mem": mem,
        }
        with self._lock:
            self._f.write(json.dumps(rec, default=str) + "\n")
        if console:
            g = mem.get("gpu0", {})
            extras = " ".join(
                f"{k}={fields[k]}"
                for k in ("phase", "step", "call", "dur_s", "buffer_size", "err")
                if k in fields
            )
            # print (not logging): logging isn't configured until exp.main runs.
            print(
                f"[memdiag +{rec['t']:9.2f}s] {ev:<14} {extras} "
                f"use={_gib(g.get('bytes_in_use', 0)):.2f}GiB "
                f"peak={_gib(g.get('peak_bytes_in_use', 0)):.2f}GiB",
                file=sys.stderr,
                flush=True,
            )

    def write_text(self, name: str, text: str) -> None:
        with open(os.path.join(self.dir, name), "w") as f:
            f.write(text)

    # -- background sampler --------------------------------------------------
    def start_sampler(self) -> None:
        if self._sample_sec <= 0:
            return
        threading.Thread(
            target=self._sample_loop, name="memdiag-sampler", daemon=True
        ).start()

    def _sample_loop(self) -> None:
        while True:
            self.event("sample", console=False)
            time.sleep(self._sample_sec)

    # -- phase wrapper -------------------------------------------------------
    def wrap_phase(
        self,
        name: str,
        fn,
        *,
        block: bool = True,
        log_every: int = 1,
        step_obj=None,
        step_from_arg0: bool = False,
        pre_hook=None,
        dump_mem_analysis: bool = False,
    ):
        """Wrap fn so entry/exit/error are logged with memory stats.

        block=True runs jax.block_until_ready on the output so an async OOM
        surfaces inside THIS phase, not at some later sync point. log_every>1
        thins chatty per-call phases (first 3 calls always logged).
        """
        counter = itertools.count(1)
        tracer = self

        def wrapped(*args, **kwargs):
            n = next(counter)
            if dump_mem_analysis and n == 1:
                # memory_analysis() needs the traced arg avals, which only `wrapped` holds (the
                # pre_hook contract passes just call_n). .lower().compile() does NOT execute, so a
                # donated-args jit (jit-3) is not consumed here -- the same fn(*args) below still runs
                # and donates normally. It also does NOT warm the executable cache, so the first real
                # call recompiles (~2x compile on the first policy update -- not a hang). Gate on n==1
                # so we compile-for-analysis at most once per jit.
                try:
                    compiled = fn.lower(*args, **kwargs).compile()
                    safe = name.replace("(jit)", "").strip().replace("/", "_")
                    tracer.write_text(
                        f"{safe}.memory_analysis.txt", str(compiled.memory_analysis())
                    )
                    tracer.event("memory_analysis_dumped", phase=name)
                except Exception as e:  # noqa: BLE001 - diagnostics must never crash the run
                    tracer.event("memory_analysis_failed", phase=name, err=repr(e)[:500])
            obj = args[0] if (step_from_arg0 and args) else step_obj
            step = None
            if obj is not None:
                try:
                    step = int(obj.training_steps)
                except Exception:  # noqa: BLE001
                    step = None
            do_log = log_every <= 1 or n <= 3 or n % log_every == 0
            if pre_hook is not None:
                try:
                    pre_hook(n)
                except Exception:  # noqa: BLE001
                    pass
            if do_log:
                tracer.event("phase_start", phase=name, call=n, step=step)
            t0 = time.monotonic()
            try:
                out = fn(*args, **kwargs)
                if block:
                    out = jax.block_until_ready(out)
            except BaseException as e:  # noqa: BLE001 - log then re-raise (incl. OOM)
                tracer.event(
                    "phase_ERROR",
                    phase=name,
                    call=n,
                    step=step,
                    dur_s=round(time.monotonic() - t0, 3),
                    err=repr(e)[:1500],
                )
                raise
            if do_log:
                tracer.event(
                    "phase_end",
                    phase=name,
                    call=n,
                    step=step,
                    dur_s=round(time.monotonic() - t0, 3),
                )
            return out

        return wrapped


def _install_instrumentation(tracer: MemTracer) -> None:
    # --- functions exp.main calls by (module-local) name --------------------
    _exp.collect_data = tracer.wrap_phase(
        "collect_data", _exp.collect_data, block=False
    )
    _exp.evaluate_policy = tracer.wrap_phase(
        "evaluate_policy", _exp.evaluate_policy, block=False
    )
    _exp.save_epoch_state = tracer.wrap_phase(
        "save_epoch_state", _exp.save_epoch_state, block=False
    )

    # --- the first big allocation: fp32 params + Adam state (unfrozen LLM) --
    _fsl.init_train_state = tracer.wrap_phase(
        "init_train_state(jit)", _fsl.init_train_state, block=True
    )

    # --- init brackets, outermost (OGPO) and the critic-adding layer (AWSFT) -
    orig_ogpo_init = OGPOAgentLearner.__init__

    def _ogpo_init(self, *a, **k):
        tracer.event("agent_init_start", cls=type(self).__name__)
        orig_ogpo_init(self, *a, **k)
        tracer.event("agent_init_end", cls=type(self).__name__)

    OGPOAgentLearner.__init__ = _ogpo_init

    # --- base-learner init: wrap per-instance jitted fns once they exist ----
    orig_base_init = FilteredSFTLearner.__init__

    def _base_init(self, *a, **k):
        tracer.event("base_init_start", cls=type(self).__name__)
        orig_base_init(self, *a, **k)
        self._ema_update_fn = tracer.wrap_phase(
            "ema_update(jit)", self._ema_update_fn, block=True, step_obj=self
        )
        buf = self._online_data_buffer
        buf.sample = tracer.wrap_phase(
            "buffer_sample", buf.sample, block=True, step_obj=self
        )
        tracer.event("base_init_end", cls=type(self).__name__)

    FilteredSFTLearner.__init__ = _base_init

    # --- wrap the jitted critic/policy updates whenever they are (re)built --
    saved_profile = {"done": False}

    def _save_profile_before_first_policy_update(call_n: int) -> None:
        if call_n != 1 or saved_profile["done"]:
            return
        saved_profile["done"] = True
        path = os.path.join(tracer.dir, "live_buffers_before_first_policy_update.prof")
        try:
            jax.profiler.save_device_memory_profile(path)
            tracer.event("saved_device_memory_profile", path=path)
        except Exception as e:  # noqa: BLE001
            tracer.event("device_memory_profile_failed", err=repr(e)[:500])

    orig_refresh = AdvantageWeightedSFTLearner._refresh_update_functions

    def _refresh(self):
        orig_refresh(self)
        self._update_critics_jitted = tracer.wrap_phase(
            "critic_update(jit)", self._update_critics_jitted, block=True, step_obj=self
        )
        self._update_policy_jitted = tracer.wrap_phase(
            "policy_update(jit)",
            self._update_policy_jitted,
            block=True,
            step_obj=self,
            pre_hook=_save_profile_before_first_policy_update,
        )

    AdvantageWeightedSFTLearner._refresh_update_functions = _refresh

    # OGPOAgentLearner has its own _refresh_update_functions that calls super() (-> the AWR _refresh
    # patch above, wrapping critic + the inert built-but-uncalled mono jit) and then builds the four
    # OGPO jits. Wrap those four here. Move the pprof "before first policy update" snapshot onto jit-1
    # (the FIRST policy-update op under the split); the mono jit's pre_hook never fires under OGPO, and
    # the shared saved_profile["done"] flag keeps the snapshot once-only from whichever fires first.
    orig_ogpo_refresh = OGPOAgentLearner._refresh_update_functions

    def _ogpo_refresh(self):
        orig_ogpo_refresh(self)
        self._sampler_advantage_jitted = tracer.wrap_phase(
            "sampler_adv(jit)", self._sampler_advantage_jitted, block=True, step_obj=self,
            pre_hook=_save_profile_before_first_policy_update, dump_mem_analysis=True,
        )
        # jit-2 is now two passes (Option 2): jit-2a PPO scan grads, jit-2b BC anchor
        # accumulated into the donated grads_pg. Wrap both so each HLO module + phase
        # timing is attributable (loss_grad_pg -> module A, bc_grad_acc -> module B).
        self._loss_grad_pg_jitted = tracer.wrap_phase(
            "loss_grad_pg(jit)", self._loss_grad_pg_jitted, block=True, step_obj=self,
            dump_mem_analysis=True,
        )
        self._bc_grad_accumulate_jitted = tracer.wrap_phase(
            "bc_grad_acc(jit)", self._bc_grad_accumulate_jitted, block=True, step_obj=self,
            dump_mem_analysis=True,
        )
        self._optimizer_tail_jitted = tracer.wrap_phase(
            "opt_tail(jit)", self._optimizer_tail_jitted, block=True, step_obj=self,
            dump_mem_analysis=True,
        )
        self._policy_param_norm_jitted = tracer.wrap_phase(
            "param_norm(jit)", self._policy_param_norm_jitted, block=True, step_obj=self,
            dump_mem_analysis=True,
        )

    OGPOAgentLearner._refresh_update_functions = _ogpo_refresh

    # --- per-step marker with buffer size ------------------------------------
    orig_update = OGPOAgentLearner.update

    def _update(self):
        step = int(self.training_steps) + 1
        tracer.event(
            "update_start", step=step, buffer_size=int(self._online_data_buffer.size)
        )
        out = orig_update(self)
        tracer.event("update_end", step=step, console=False)
        return out

    OGPOAgentLearner.update = _update

    # --- collection lifecycle (EMA attach/detach = big VRAM delta) -----------
    FilteredSFTLearner.start_data_collection = tracer.wrap_phase(
        "start_data_collection",
        FilteredSFTLearner.start_data_collection,
        block=False,
        step_from_arg0=True,
    )
    FilteredSFTLearner.end_data_collection = tracer.wrap_phase(
        "end_data_collection",
        FilteredSFTLearner.end_data_collection,
        block=False,
        step_from_arg0=True,
    )
    # Rollout inference is called once per env chunk -> thin the log.
    FilteredSFTLearner._sample_action = tracer.wrap_phase(
        "sample_action(collect)",
        FilteredSFTLearner._sample_action,
        block=False,
        log_every=25,
        step_from_arg0=True,
    )


if __name__ == "__main__":
    _register_unfrozen_config()
    config = _config.cli()

    log_path = os.environ.get("MEMDIAG_LOG") or os.path.join(
        os.getcwd(), f"memdiag_{config.exp_name}.jsonl"
    )
    tracer = MemTracer(
        log_path, sample_sec=float(os.environ.get("MEMDIAG_SAMPLE_SEC", "1.0"))
    )
    tracer.event(
        "run_start",
        argv=sys.argv,
        batch_size=int(config.batch_size),
        num_sde_steps=int(getattr(config.rl, "num_sde_steps", -1)),
        policy_training_start_step=int(config.rl.policy.training_start_step),
        policy_update_interval=int(config.rl.policy.update_interval),
        env={
            k: os.environ.get(k)
            for k in (
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_FLAGS",
                "JAX_LOG_COMPILES",
                "JAX_TRACEBACK_FILTERING",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    )
    _install_instrumentation(tracer)
    tracer.start_sampler()
    try:
        _exp.main(config)
        tracer.event("run_end_ok")
    except SystemExit as e:
        tracer.event("run_end_sysexit", code=e.code)
        raise
    except BaseException as e:  # noqa: BLE001 - record the fatal error, then re-raise
        tracer.event("run_FATAL", err=repr(e)[:2000])
        tracer.write_text("traceback.txt", traceback.format_exc())
        raise
