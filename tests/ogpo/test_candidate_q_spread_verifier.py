"""Independent verification of scripts/probe_candidate_q_spread.py
(docs/changes/2026-09-07-candidate-q-spread-rollouts/).

Written without the implementing session's reasoning. Two halves:

1. A CPU smoke of the REAL ``AdvantageWeightedSFTLearner.sample_actions`` body
   (the `_bon_record` contract the probe depends on), driven on a stub ``self``
   built with ``object.__new__`` -- the pattern
   tests/ogpo/test_per_task_critics_verifier2.py:333 uses for the
   ``end_data_collection`` guard. Real config (``pi05_libero_online_ogpo_ref``),
   real grouping / tiling / reshape / argmax / scatter / record code; fakes only
   for the pi0.5 forward, the transforms and the two nnx models. No GPU, no
   LIBERO, no weights. tests/ogpo/test_candidate_q_spread.py exercises the probe
   against a hand-written *imitation* of that contract, which cannot catch a
   drift in the contract itself.

2. Edge cases of the probe's own helpers and rollout loop that the implementer's
   tests do not reach: max_chunks truncation, hook state after an exception,
   dead-env querying, obs scatter, NaN in the returned chunk, JSON dumping of the
   exact objects main() writes, and PNG magic bytes.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import types

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "probe_candidate_q_spread_verifier", _ROOT / "scripts" / "probe_candidate_q_spread.py"
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


# =========================================================================== #
# 1. The real sample_actions hook contract
# =========================================================================== #
M = 4          # candidates per query
H = 6          # action horizon
AD = 7         # robot action dim (< model.action_dim, so _pad_last_dim fires)
SD = 8         # state dim == _transition_state_dim (so the pad is a no-op)
PD = 5         # prefix embedding dim


@pytest.fixture(scope="module")
def awr():
    """The real learner module + config. Module-scoped: the import costs ~10 s."""
    import src.training.config as cfg_mod
    from src.rl.advantage_weighted_sft import advantage_weighted_sft_learner as mod

    return types.SimpleNamespace(mod=mod, cfg_mod=cfg_mod)


def _make_cfg(cfg_mod, *, n_samples=M, store_prefix_rep=False, reduction="mean"):
    cfg = cfg_mod.get_config("pi05_libero_online_ogpo_ref")
    rl = dataclasses.replace(
        cfg.rl,
        n_samples=n_samples,
        critic=dataclasses.replace(
            cfg.rl.critic, inference_start_step=1, reduction=reduction
        ),
    )
    return dataclasses.replace(
        cfg,
        rl=rl,
        collect=dataclasses.replace(cfg.collect, store_prefix_rep=store_prefix_rep),
    )


def _stub_learner(mod, cfg, *, training_steps=5):
    """A stub ``self`` carrying exactly the attributes sample_actions reads."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    class _TinyQ(nnx.Module):
        def __init__(self):
            self.bias = nnx.Param(jnp.zeros(()))

        def __call__(self, obs, actions):
            # Two heads whose MEAN is a constant and whose MIN is not, so the
            # test can tell which reduction the scores came from.
            base = jnp.sum(jnp.asarray(actions), axis=-1) + jnp.sum(
                jnp.asarray(obs["state"]), axis=-1
            )
            return jnp.stack([base, -base], axis=0) + self.bias

    class _TinyPrefixModel(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.zeros((2,)))

    obj = object.__new__(mod.AdvantageWeightedSFTLearner)
    obj._config = cfg
    obj.training_steps = training_steps
    obj._rng = jax.random.key(0)
    obj._task_registry = None
    obj._bon_record = None
    obj._transition_state_dim = SD

    qdef, qstate = nnx.split(_TinyQ())
    obj._state_action_critic_state = types.SimpleNamespace(
        ema_params=None, params=qstate, model_def=qdef
    )
    pdef, pstate = nnx.split(_TinyPrefixModel())
    obj._train_state = types.SimpleNamespace(
        model_def=pdef, params=pstate, ema_params=pstate
    )

    obj.sample_batches = []

    def _process_obs_for_pi0(observations, task_description=None):
        return {
            "observation/state": np.asarray(observations["state"], dtype=np.float32),
            "prompt": task_description,
        }

    def _sample_action(observations, rng, train_state, return_prefix_rep=False):
        # Row r of the tiled batch carries the value r, so a test can read off
        # which tiled draw a candidate came from.
        n = int(np.asarray(observations["observation/state"]).shape[0])
        obj.sample_batches.append(n)
        return np.arange(n, dtype=np.float32)[:, None, None] * np.ones(
            (1, H, AD), dtype=np.float32
        )

    def _input_transform(d):
        return {
            "image": {"base_0_rgb": np.zeros((2, 2, 3), dtype=np.float32)},
            "image_mask": {"base_0_rgb": np.bool_(True)},
            "state": np.asarray(d["observation/state"], dtype=np.float32),
        }

    def _get_prefix_rep_with_model(m, observation):
        n = int(np.asarray(observation.state).shape[0])
        return np.arange(n * PD, dtype=np.float32).reshape(n, PD)

    obj._process_obs_for_pi0 = _process_obs_for_pi0
    obj._sample_action = _sample_action
    obj._state_normalize = lambda d: {"state": np.asarray(d["state"], np.float32)}
    obj._action_normalize = lambda d: {"actions": np.asarray(d["actions"], np.float32)}
    obj._policy = types.SimpleNamespace(_input_transform=_input_transform)
    obj._get_prefix_rep_with_model = _get_prefix_rep_with_model
    return obj


# Two envs share a prompt and one does not, so the groups are NON-CONTIGUOUS
# ([0, 2] and [1]) -- exactly the case gather_records exists for.
_DESC = ["pick the bowl", "open the drawer", "pick the bowl"]
_ENV_NUM = 3


def _run(mod, obj, obs=None):
    obs = obs if obs is not None else {
        "state": np.arange(_ENV_NUM * SD, dtype=np.float32).reshape(_ENV_NUM, SD)
    }
    obj._bon_record = []
    out = mod.AdvantageWeightedSFTLearner.sample_actions(
        obj, obs, task_description=list(_DESC), task_id=["t0", "t1", "t0"]
    )
    return out, obj._bon_record


def test_real_hook_record_shapes_and_dtypes(awr):
    """The four fields gather_records reads, from the real code path."""
    cfg = _make_cfg(awr.cfg_mod)
    obj = _stub_learner(awr.mod, cfg)
    out, rec = _run(awr.mod, obj)

    assert [r["indices"] for r in rec] == [[0, 2], [1]], "prompt grouping order"
    for r in rec:
        g = len(r["indices"])
        assert r["candidates"].shape == (g, M, H, AD)
        assert r["candidates"].dtype == np.float32
        assert r["scores"].shape == (g, M)
        assert r["scores"].dtype == np.float32
        assert r["best_idx"].shape == (g,)
        assert r["best_idx"].dtype == np.int32
        assert all(isinstance(i, int) for i in r["indices"])
    assert np.asarray(out).shape == (_ENV_NUM, H, AD)
    assert np.asarray(out).dtype == np.float32


def test_real_hook_record_carries_more_than_the_four_documented_fields(awr):
    """BLAST-RADIUS.md lists indices/candidates/scores/best_idx. The real record
    also carries `state` and `prefix` (and `task_index` under per-task critics).
    Harmless for this probe -- gather_records ignores them -- but the spec's
    field list is incomplete, and a future gather that iterates the dict would
    trip on it."""
    cfg = _make_cfg(awr.cfg_mod)
    obj = _stub_learner(awr.mod, cfg)
    _, rec = _run(awr.mod, obj)
    assert set(rec[0]) == {"indices", "candidates", "scores", "best_idx", "state", "prefix"}


def test_real_returned_chunk_is_bit_exactly_the_gathered_argmax_candidate(awr):
    """The probe's rollout equality check, run against the real producer.

    Exact equality (not approx): both sides are the same float32 buffer indexed
    two different ways -- AWR:619 takes group_actions[arange, best_idx] and
    AWR:650 scatters it into all_best_actions; the record stores the same
    group_actions. Any cast or recompute on either side would break this.
    """
    cfg = _make_cfg(awr.cfg_mod)
    obj = _stub_learner(awr.mod, cfg)
    out, rec = _run(awr.mod, obj)
    cands, scores, best_idx = probe.gather_records(rec, _ENV_NUM)
    best = np.asarray(out, dtype=np.float32)
    assert np.array_equal(cands[np.arange(_ENV_NUM), best_idx], best)
    assert scores.shape == (_ENV_NUM, M)


def test_real_candidate_zero_is_the_first_tiled_noise_draw_for_that_env(awr):
    """The BoN=0 arm executes candidates[:, 0]. That is only "one iid policy
    draw" if the (g*M) tiled batch reshapes to (g, M) with candidate k of group
    row r at flat row r*M + k. _sample_action here returns its own row index, so
    the placement is readable off the values."""
    cfg = _make_cfg(awr.cfg_mod)
    obj = _stub_learner(awr.mod, cfg)
    _, rec = _run(awr.mod, obj)
    for r in rec:
        for row in range(len(r["indices"])):
            for k in range(M):
                assert np.all(r["candidates"][row, k] == float(row * M + k)), (
                    f"group row {row}, candidate {k} is not tiled row {row * M + k}"
                )
    # One _sample_action call per prompt group, each of size g*M.
    assert obj.sample_batches == [2 * M, 1 * M]


@pytest.mark.parametrize(
    "kw,steps",
    [({"n_samples": 1}, 5), ({}, 0)],
    ids=["n_samples=1", "training_steps<inference_start_step"],
)
def test_real_single_sample_path_records_nothing_and_the_probe_says_why(awr, kw, steps):
    """Both fallback conditions at AWR:420 leave the hook empty. gather_records
    must then raise naming the two knobs -- this is what makes the probe's two
    main() guards necessary AND sufficient."""
    cfg = _make_cfg(awr.cfg_mod, **kw)
    obj = _stub_learner(awr.mod, cfg, training_steps=steps)
    out, rec = _run(awr.mod, obj)
    assert rec == [], "the fallback path must not append a record"
    assert np.asarray(out).shape == (_ENV_NUM, H, AD)
    with pytest.raises(RuntimeError, match="single-sample"):
        probe.gather_records(rec, _ENV_NUM)


def test_real_tuple_return_under_store_prefix_rep_is_unwrapped_by_policy_chunk(awr):
    """The ref recipe passes --collect.store_prefix_rep unconditionally
    (scripts/ogpo_multitask_4task.sh:304), so sample_actions returns a tuple on
    every real run of this probe."""
    cfg = _make_cfg(awr.cfg_mod, store_prefix_rep=True)
    obj = _stub_learner(awr.mod, cfg)
    out, rec = _run(awr.mod, obj)
    assert isinstance(out, tuple) and len(out) == 2
    agent = types.SimpleNamespace(sample_actions=lambda o, **kw: out)
    best = probe.policy_chunk(agent, {}, {"task_description": list(_DESC)}, ["t"] * _ENV_NUM)
    cands, _, best_idx = probe.gather_records(rec, _ENV_NUM)
    assert np.array_equal(cands[np.arange(_ENV_NUM), best_idx], best)


def test_real_scores_are_the_configured_reduction_not_a_hardcoded_one(awr):
    """The y-axis is "variance of the reduced Q the run selected on". Two heads
    with mean == 0 and min == -|base| make the two reductions distinguishable:
    under `mean` every candidate scores 0 (variance 0), under `min` it does not.
    """
    obs = {"state": np.arange(_ENV_NUM * SD, dtype=np.float32).reshape(_ENV_NUM, SD)}

    cfg_mean = _make_cfg(awr.cfg_mod, reduction="mean")
    _, rec_mean = _run(awr.mod, _stub_learner(awr.mod, cfg_mean), obs)
    _, s_mean, _ = probe.gather_records(rec_mean, _ENV_NUM)
    # exact: mean of (x, -x) is 0 in float32 for every x
    assert np.all(s_mean == 0.0)
    assert np.all(probe.candidate_q_variance(s_mean) == 0.0)

    cfg_min = _make_cfg(awr.cfg_mod, reduction="min")
    _, rec_min = _run(awr.mod, _stub_learner(awr.mod, cfg_min), obs)
    _, s_min, _ = probe.gather_records(rec_min, _ENV_NUM)
    assert probe.candidate_q_variance(s_min).min() > 0.0, (
        "min-reduction scores must differ across candidates here; if they do not, "
        "the probe is not reading rl.critic.reduction"
    )


# =========================================================================== #
# 2. Probe helpers -- edges the implementer's tests do not reach
# =========================================================================== #
def test_variance_is_computed_in_float64_from_float32_scores():
    """The hook hands back float32. A large common offset with a small spread
    loses ~7 significant digits in a float32 accumulation; the helper's
    np.asarray(..., float64) up-cast is what keeps the answer usable."""
    base = np.float32(1.0e6)
    s = np.array([[base, base + np.float32(1.0)]], dtype=np.float32)
    var = probe.candidate_q_variance(s)
    assert var.dtype == np.float64
    # float32 can represent 1e6 and 1e6+1 exactly (spacing at 1e6 is 0.0625),
    # so the population variance is exactly ((0.5)^2 + (0.5)^2)/2 = 0.25.
    assert var[0] == pytest.approx(0.25, abs=1e-12)


def test_variance_propagates_nan_rather_than_raising():
    """A NaN Q-score (a diverged critic head) silently yields a NaN point on the
    plot and a NaN in the JSON -- it is not caught anywhere."""
    s = np.array([[0.0, np.nan, 1.0]])
    assert np.isnan(probe.candidate_q_variance(s)[0])


def test_gather_rejects_a_candidate_count_mismatch_between_groups():
    """M is read from records[0] only; a second group with a different M must
    not be silently truncated or broadcast."""
    def rec(indices, m):
        g = len(indices)
        return {
            "indices": list(indices),
            "candidates": np.zeros((g, m, 2, 3), np.float32),
            "scores": np.zeros((g, m), np.float32),
            "best_idx": np.zeros(g, np.int32),
        }

    with pytest.raises(RuntimeError, match="shapes disagree"):
        probe.gather_records([rec([0], 4), rec([1], 5)], env_num=2)


def test_gather_accepts_numpy_indices_not_only_python_lists():
    """The real hook writes `[int(i) for i in indices]` (AWR:624), but the
    partition check goes through np.concatenate, so an ndarray must work too --
    pinning it means a future hook change to ndarray indices cannot break the
    probe silently."""
    r = {
        "indices": np.array([1, 0]),
        "candidates": np.arange(2 * 3 * 1 * 1, dtype=np.float32).reshape(2, 3, 1, 1),
        "scores": np.array([[0.0, 1.0, 2.0], [5.0, 4.0, 3.0]], np.float32),
        "best_idx": np.array([2, 0], np.int32),
    }
    cands, scores, best = probe.gather_records([r], env_num=2)
    assert best.tolist() == [0, 2]                       # placed by index, not order
    assert scores[1].tolist() == [0.0, 1.0, 2.0]
    assert np.all(cands[0] == r["candidates"][1])


def test_mean_over_alive_tolerates_a_zero_length_trace():
    """An episode with no recorded chunk cannot happen today (every env is live
    at n=0), but the helper must not divide by zero if it ever does."""
    mean, alive = probe.mean_over_alive([np.zeros(0), np.array([2.0, 4.0])])
    assert alive.tolist() == [1, 1]
    assert mean.tolist() == [2.0, 4.0]


# =========================================================================== #
# 3. rollout -- paths the implementer's tests do not take
# =========================================================================== #
class _Agent:
    """Minimal `_bon_record` producer: one group holding every env."""

    def __init__(self, env_num, m=3):
        self._bon_record = None
        self.env_num, self.m = env_num, m
        self.query = 0
        self.seen_obs = []

    def _cands(self, q):
        c = np.zeros((self.env_num, self.m, 2, 3), np.float32)
        for e in range(self.env_num):
            for k in range(self.m):
                c[e, k] = 1000 * e + 10 * q + k
        return c

    def _scores(self, q):
        # argmax is candidate m-1 for every env, so BoN and BoN=0 differ.
        return np.tile(np.arange(self.m, dtype=np.float32), (self.env_num, 1)) + q

    def sample_actions(self, obs, task_description=None, task_id=None):
        q, self.query = self.query, self.query + 1
        self.seen_obs.append(np.array(obs["x"], copy=True))
        c, s = self._cands(q), self._scores(q)
        bi = s.argmax(axis=1)
        if self._bon_record is not None:
            self._bon_record.append(
                {
                    "indices": list(range(self.env_num)),
                    "candidates": c,
                    "scores": s,
                    "best_idx": bi.astype(np.int32),
                }
            )
        return c[np.arange(self.env_num), bi]


class _Env:
    """(live, replan) step layout. `plan[e]` is a list of (chunk, substep, kind)
    with kind in {"term", "trunc"}; an env not listed never ends."""

    def __init__(self, env_num, plan, replan=3):
        self.env_num, self.replan, self.plan = env_num, replan, plan
        self.chunks = np.zeros(env_num, int)
        self.calls = []

    def step(self, act, id):
        live = list(id)
        self.calls.append(list(live))
        term = np.zeros((len(live), self.replan), bool)
        trunc = np.zeros((len(live), self.replan), bool)
        for row, e in enumerate(live):
            n = self.chunks[e]
            self.chunks[e] += 1
            for (c, j, kind) in self.plan.get(e, []):
                if c == n:
                    (term if kind == "term" else trunc)[row, j] = True
        nobs = {"x": np.array([[100.0 + e] for e in live])}
        return nobs, np.zeros((len(live), self.replan)), term, trunc, {}


def _drive(agent, env, *, max_chunks, use_bon=True):
    G = env.env_num
    obs = {"x": np.zeros((G, 1))}
    info = {"task_description": ["t"] * G}
    return probe.rollout(
        env, agent, obs, info, ["task"] * G, max_chunks=max_chunks, use_bon=use_bon
    )


def test_rollout_truncates_at_max_chunks_and_reports_failure():
    """An env that never terminates: exactly max_chunks score rows, succ False,
    and steps == replan * max_chunks. main() sets max_chunks from the suite's
    TimeLimit + 2, so in a real run the env truncates first -- but a wrong
    suite name or replan_steps would land here silently."""
    env = _Env(2, plan={})
    agent = _Agent(2)
    succ, steps, mats, ex = _drive(agent, env, max_chunks=4)
    assert succ.tolist() == [False, False]
    assert [m.shape for m in mats] == [(4, 3), (4, 3)]
    assert steps.tolist() == [12, 12]
    assert agent.query == 4


def test_rollout_disarms_the_hook_when_sample_actions_raises():
    """Verifier finding 1 (fixed): rollout now arms `agent._bon_record` under
    try/finally. If sample_actions raises mid-wave the exception propagates
    unchanged AND the hook is disarmed, so the next rollout on that agent dies
    on its own cause rather than on the 'already armed' guard."""
    class _Boom(_Agent):
        def sample_actions(self, obs, task_description=None, task_id=None):
            if self.query == 1:
                raise ValueError("policy forward blew up")
            return super().sample_actions(obs, task_description, task_id)

    env = _Env(2, plan={})
    agent = _Boom(2)
    with pytest.raises(ValueError, match="blew up"):
        _drive(agent, env, max_chunks=5)
    assert agent._bon_record is None, "the hook must be disarmed on the way out"
    # ... and the next call dies on the REAL cause (the fake keeps raising),
    # not on the 'already armed' guard, and leaves the hook disarmed again.
    with pytest.raises(ValueError, match="blew up"):
        _drive(agent, env, max_chunks=1)
    assert agent._bon_record is None


def test_rollout_keeps_querying_dead_envs_but_never_records_them():
    """env 0 ends at chunk 0, env 1 at chunk 2. sample_actions is still called
    with all G envs on every query (cost, not correctness), but only live envs
    get a score row."""
    env = _Env(2, plan={0: [(0, 0, "term")], 1: [(2, 2, "term")]})
    agent = _Agent(2)
    succ, steps, mats, ex = _drive(agent, env, max_chunks=10, use_bon=True)
    assert succ.tolist() == [True, True]
    assert [m.shape[0] for m in mats] == [1, 3]
    assert steps.tolist() == [1, 9]
    assert agent.query == 3, "one sample_actions per chunk, over ALL envs"
    assert env.calls == [[0, 1], [1], [1]], "dead envs stop being stepped"


def test_rollout_scatters_only_live_rows_into_obs():
    """The jax.tree.map scatter must leave a finished env's observation frozen;
    overwriting it with a live env's row would feed the policy a scrambled batch
    (and, under BoN=0, score candidates at the wrong state)."""
    env = _Env(3, plan={0: [(0, 0, "term")]})
    agent = _Agent(3)
    _drive(agent, env, max_chunks=3, use_bon=True)
    # query 0 sees the reset obs; queries 1..2 see 100+e for the live envs and
    # the frozen reset value for env 0.
    assert agent.seen_obs[0].ravel().tolist() == [0.0, 0.0, 0.0]
    assert agent.seen_obs[1].ravel().tolist() == [100.0, 101.0, 102.0]
    assert agent.seen_obs[2].ravel().tolist() == [100.0, 101.0, 102.0]


def test_rollout_counts_terminated_as_success_when_both_flags_fire():
    """TimeLimit truncation and task success can land on the same sub-step.
    `succ = bool(term[row, j])` makes terminated win, matching evaluate_policy
    (src/training/collect.py:74-76)."""
    env = _Env(1, plan={0: [(1, 1, "term"), (1, 1, "trunc")]})
    succ, steps, mats, _ = _drive(_Agent(1), env, max_chunks=5, use_bon=True)
    assert succ.tolist() == [True]
    assert steps.tolist() == [5]      # 3 + 2 sub-steps
    assert mats[0].shape[0] == 2


def test_rollout_reports_a_nan_chunk_as_non_finite_not_misaligned():
    """Verifier finding 2 (fixed): np.array_equal is False on any NaN, so a
    policy emitting a NaN action with a perfectly aligned record used to fail
    as "misaligned". A separate finiteness check now names the real cause and
    the offending env rows."""
    class _NaNAgent(_Agent):
        def sample_actions(self, obs, task_description=None, task_id=None):
            best = super().sample_actions(obs, task_description, task_id)
            self._bon_record[0]["candidates"][0, :, 0, 0] = np.nan
            best = np.array(best)
            best[0, 0, 0] = np.nan
            return best

    env = _Env(2, plan={})
    with pytest.raises(RuntimeError, match=r"non-finite chunk at query 0 pass 0 \(rows \[0\]\)"):
        _drive(_NaNAgent(2), env, max_chunks=2, use_bon=True)


def test_rollout_outputs_dump_to_json_the_way_main_writes_them():
    """main() puts score_mats[e].tolist(), exec_per_env[e] and the variance
    trace straight into the summary dict and dumps with default=float. Pin that
    nothing numpy-typed survives into a key or a bare value."""
    env = _Env(2, plan={0: [(1, 0, "term")]})
    succ, steps, mats, ex = _drive(_Agent(2), env, max_chunks=3, use_bon=False)
    payload = {
        "episodes": [
            {
                "episode": e,
                "success": bool(succ[e]),
                "steps": int(steps[e]),
                "chunks": int(mats[e].shape[0]),
                "executed_idx": ex[e],
                "scores": mats[e].tolist(),
                "q_variance": probe.candidate_q_variance(mats[e]).tolist(),
            }
            for e in range(2)
        ]
    }
    round_tripped = json.loads(json.dumps(payload, indent=2, default=float))
    assert round_tripped["episodes"][0]["executed_idx"] == [0, 0]
    assert round_tripped["episodes"][0]["chunks"] == 2
    assert isinstance(round_tripped["episodes"][1]["success"], bool)


# =========================================================================== #
# 4. plots
# =========================================================================== #
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_plots_are_real_pngs_including_a_single_point_trace(tmp_path):
    """A one-chunk episode (success on the first chunk) is the common case on a
    good checkpoint; matplotlib must not choke on a length-1 line, and the file
    must be a PNG, not a zero-byte stub."""
    ep = tmp_path / "libero_90_38" / "ep00_seed0_succ.png"
    probe.plot_episode(ep, np.array([0.5]), "t", 8)
    assert ep.read_bytes()[:8] == _PNG_MAGIC

    mean, alive = probe.mean_over_alive([np.array([1.0, 2.0, 3.0]), np.array([4.0])])
    mp = tmp_path / "libero_90_38" / "mean_trace.png"
    probe.plot_mean(mp, mean, alive, "t", 8)
    assert mp.read_bytes()[:8] == _PNG_MAGIC
    assert alive.tolist() == [2, 1, 1]


# =========================================================================== #
# 4. Multi-pass candidates (QSPREAD_PASSES) -- added 2026-09-07 for the
#    stab-wrapper change, which ships passes=4 as its DEFAULT arm.
# =========================================================================== #
class _PassAgent:
    """`_bon_record` producer driven by a per-call score table.

    Call c returns scores ``table[c]`` (env_num, m) and candidates whose every
    element is ``100 * c + k`` for candidate k, so the executed chunk names the
    (call, candidate) it came from.
    """

    def __init__(self, table):
        self._bon_record = None
        self.table = [np.asarray(t, dtype=np.float32) for t in table]
        self.env_num, self.m = self.table[0].shape
        self.query = 0
        self.rngs = []

    def _cands(self, c):
        out = np.zeros((self.env_num, self.m, 2, 3), np.float32)
        for k in range(self.m):
            out[:, k] = 100 * c + k
        return out

    def sample_actions(self, obs, task_description=None, task_id=None):
        c, self.query = self.query, self.query + 1
        s = self.table[c % len(self.table)]
        cands = self._cands(c)
        bi = s.argmax(axis=1)
        if self._bon_record is not None:
            self._bon_record.append(
                {
                    "indices": list(range(self.env_num)),
                    "candidates": cands,
                    "scores": s,
                    "best_idx": bi.astype(np.int32),
                }
            )
        return cands[np.arange(self.env_num), bi]


class _RecEnv(_Env):
    """_Env plus a record of the action arrays actually stepped."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.acted = []

    def step(self, act, id):
        self.acted.append(np.array(act, copy=True))
        return super().step(act, id)


def _drive_passes(agent, *, passes, use_bon=True, max_chunks=1):
    G = agent.env_num
    env = _RecEnv(G, {e: [(0, 0, "term")] for e in range(G)})
    obs = {"x": np.zeros((G, 1))}
    info = {"task_description": ["t"] * G}
    return probe.rollout(
        env, agent, obs, info, ["task"] * G, max_chunks=max_chunks,
        use_bon=use_bon, passes=passes,
    ), env


def test_multipass_union_index_resolves_to_the_right_pass_and_candidate():
    """The winners sit in DIFFERENT passes (env0 in the first, env1 in the
    last), so a reversed or rotated part order changes both answers. The
    implementer's test puts the argmax in the last candidate of the last pass,
    which several wrong concatenations still satisfy.

    passes=3, m=2 -> union index j means pass j//2, candidate j%2.
    """
    table = [[[9.0, 7.0], [0.0, 1.0]],
             [[5.0, 1.0], [2.0, 3.0]],
             [[0.0, 1.0], [8.0, 6.0]]]
    agent = _PassAgent(table)
    (succ, steps, mats, exec_idx), env = _drive_passes(agent, passes=3)
    assert agent.query == 3                       # one query, three calls
    assert [e[0] for e in exec_idx] == [0, 4]
    assert mats[0].shape == (1, 6)
    assert np.array_equal(mats[0][0], np.array([9, 7, 5, 1, 0, 1], np.float32))
    assert np.array_equal(mats[1][0], np.array([0, 1, 2, 3, 8, 6], np.float32))
    # The action actually handed to the env: env0 = call 0 cand 0, env1 = call 2 cand 0.
    acted = env.acted[0]
    assert np.all(acted[0] == 100 * 0 + 0)
    assert np.all(acted[1] == 100 * 2 + 0)
    # Teeth: a reversed part order would answer [4, 0], not [0, 4].
    rev0 = np.concatenate([np.asarray(t, np.float32)[0] for t in table[::-1]])
    assert int(rev0.argmax()) != int(exec_idx[0][0])


def test_multipass_bon_off_executes_pass_zero_candidate_zero():
    """The control arm must stay one iid draw taken BEFORE any scoring, even
    though passes>1 sampled 5 better-scoring candidates after it."""
    table = [[[0.0, 1.0], [0.0, 1.0]],
             [[5.0, 9.0], [9.0, 5.0]],
             [[2.0, 3.0], [3.0, 2.0]]]
    agent = _PassAgent(table)
    (succ, steps, mats, exec_idx), env = _drive_passes(agent, passes=3, use_bon=False)
    assert [e[0] for e in exec_idx] == [0, 0]
    assert np.all(env.acted[0] == 0.0)            # call 0, candidate 0
    assert mats[0].shape == (1, 6)                # still M-wide, so the two arms
                                                  # measure the same spread


def test_split_over_passes_executes_the_same_candidate_as_one_call():
    """The exactness claim, on the probe's side: given the SAME candidate pool
    and scores, 1 call of 6 and 3 calls of 2 select the same chunk.

    (The distributional half of the claim -- that 3x2 iid draws are 6 iid draws
    -- is a property of _sample_action's `jax.random.normal(rng, (batch,...))`,
    not of this loop; see the report.)
    """
    pool_scores = np.array([[0.0, 1.0, 5.0, 9.0, 2.0, 3.0],
                            [0.0, 1.0, 9.0, 5.0, 3.0, 2.0]], np.float32)

    class _PoolAgent(_PassAgent):
        """One call returns the whole pool; `chunk` is the pool slice."""

        def __init__(self, m):
            self._bon_record = None
            self.env_num, self.m = 2, m
            self.query = 0
            self.n_calls = pool_scores.shape[1] // m

        def sample_actions(self, obs, task_description=None, task_id=None):
            c, self.query = self.query, self.query + 1
            s = pool_scores[:, c * self.m:(c + 1) * self.m]
            cands = np.zeros((2, self.m, 2, 3), np.float32)
            for k in range(self.m):
                cands[:, k] = c * self.m + k      # GLOBAL pool position
            bi = s.argmax(axis=1)
            if self._bon_record is not None:
                self._bon_record.append(
                    {"indices": [0, 1], "candidates": cands, "scores": s,
                     "best_idx": bi.astype(np.int32)}
                )
            return cands[np.arange(2), bi]

    (_, _, mats1, exec1), env1 = _drive_passes(_PoolAgent(6), passes=1)
    (_, _, mats3, exec3), env3 = _drive_passes(_PoolAgent(2), passes=3)
    assert np.array_equal(mats1[0], mats3[0]) and np.array_equal(mats1[1], mats3[1])
    assert exec1 == exec3 == [[3], [2]]
    assert np.array_equal(env1.acted[0], env3.acted[0])


def test_passes_one_cross_check_fires_but_is_off_for_passes_gt_one():
    """`exec_idx == best_parts[0]` is asserted ONLY at passes==1. The stab
    wrapper's default arm is passes=4, so that check never runs there --
    recorded here so the gap is deliberate, not assumed absent."""

    class _LyingAgent(_PassAgent):
        def sample_actions(self, obs, task_description=None, task_id=None):
            out = super().sample_actions(obs, task_description, task_id)
            if self._bon_record:                 # claim a best_idx it did not use
                r = self._bon_record[-1]
                r["best_idx"] = np.zeros_like(r["best_idx"])
            return out

    table = [[[0.0, 1.0], [0.0, 1.0]]] * 4
    with pytest.raises(RuntimeError, match="gathered argmax candidates differ"):
        _drive_passes(_LyingAgent(table), passes=1)
    # passes=2 hits the SAME per-pass check first, so the lie is still caught --
    # it is only the union-vs-production comparison that is skipped.
    with pytest.raises(RuntimeError, match="gathered argmax candidates differ"):
        _drive_passes(_LyingAgent(table), passes=2)


def test_real_sample_actions_draws_a_fresh_rng_each_call(awr):
    """Multi-pass is only extra candidates if consecutive calls at the SAME
    observation use different noise. sample_actions splits self._rng per call
    (AWR:423) and hands the fresh half to _sample_action."""
    import jax

    cfg = _make_cfg(awr.cfg_mod)
    obj = _stub_learner(awr.mod, cfg)
    seen = []
    inner = obj._sample_action

    def _spy(observations, rng, train_state, return_prefix_rep=False):
        seen.append(np.asarray(jax.random.key_data(rng)).copy())
        return inner(observations, rng, train_state, return_prefix_rep)

    obj._sample_action = _spy
    rng_before = np.asarray(jax.random.key_data(obj._rng)).copy()
    _run(awr.mod, obj)
    _run(awr.mod, obj)
    rng_after = np.asarray(jax.random.key_data(obj._rng)).copy()
    # 2 prompt groups x 2 calls
    assert len(seen) == 4
    assert np.array_equal(seen[0], seen[1]), (
        "within one call every prompt group reuses the same key (pre-existing); "
        "if this ever changes the multi-pass reasoning is unaffected"
    )
    assert not np.array_equal(seen[0], seen[2]), "call 2 reused call 1's noise key"
    assert not np.array_equal(rng_before, rng_after), "self._rng did not advance"
