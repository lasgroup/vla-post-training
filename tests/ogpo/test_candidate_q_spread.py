"""Tests for the pure helpers and the env-driving loop of
scripts/probe_candidate_q_spread.py (docs/changes/2026-09-07-candidate-q-spread-rollouts/).

The script is loaded with importlib the way Tier B loads Tier A; its env /
learner imports live inside main(), so nothing here needs LIBERO, MuJoCo, a
GPU or real weights. The rollout loop is exercised against a fake agent that
reproduces the `_bon_record` contract of AdvantageWeightedSFTLearner.
sample_actions (per-prompt-group dicts with indices/candidates/scores/best_idx,
returning candidates[best_idx]) and a fake vector env with the
(live, replan) step layout of the wrapped SubprocVectorEnv.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "probe_candidate_q_spread", _ROOT / "scripts" / "probe_candidate_q_spread.py"
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


# --------------------------------------------------------------------------- #
# candidate_q_variance
# --------------------------------------------------------------------------- #
def test_variance_is_population_variance_over_candidates():
    scores = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    var = probe.candidate_q_variance(scores)
    # ddof=0: mean 2.5, squared deviations 2.25+0.25+0.25+2.25 = 5 / 4 = 1.25.
    assert var.shape == (2,)
    assert var[0] == pytest.approx(1.25, abs=1e-12)  # exact in float64
    assert var[1] == 0.0


def test_variance_rejects_single_candidate_and_wrong_rank():
    with pytest.raises(ValueError):
        probe.candidate_q_variance(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        probe.candidate_q_variance(np.zeros(8))


# --------------------------------------------------------------------------- #
# gather_records
# --------------------------------------------------------------------------- #
def _record(indices, m, h=2, act=3, offset=0):
    g = len(indices)
    cands = np.zeros((g, m, h, act), dtype=np.float32)
    scores = np.zeros((g, m), dtype=np.float32)
    for r, e in enumerate(indices):
        for k in range(m):
            cands[r, k] = 100 * e + k + offset
            scores[r, k] = float(k) - e  # argmax is always the last candidate
    return {
        "indices": list(indices),
        "candidates": cands,
        "scores": scores,
        "best_idx": np.full(g, m - 1, dtype=np.int32),
    }


def test_gather_places_groups_by_env_index():
    m = 4
    recs = [_record([3, 0], m), _record([2, 1], m)]
    cands, scores, best = probe.gather_records(recs, env_num=4)
    assert cands.shape == (4, m, 2, 3) and scores.shape == (4, m) and best.shape == (4,)
    for e in range(4):
        assert np.all(cands[e, 0] == 100 * e)
        assert scores[e, 1] == pytest.approx(1.0 - e)
        assert best[e] == m - 1


def test_gather_rejects_missing_duplicate_and_empty():
    with pytest.raises(RuntimeError, match="partition"):
        probe.gather_records([_record([0, 1], 3)], env_num=3)
    with pytest.raises(RuntimeError, match="partition"):
        probe.gather_records([_record([0, 1], 3), _record([1, 2], 3)], env_num=3)
    with pytest.raises(RuntimeError, match="single-sample"):
        probe.gather_records([], env_num=2)


def test_gather_rejects_shape_disagreement_between_groups():
    a = _record([0], 3)
    b = _record([1], 3, h=5)
    with pytest.raises(RuntimeError, match="shapes disagree"):
        probe.gather_records([a, b], env_num=2)


# --------------------------------------------------------------------------- #
# mean_over_alive
# --------------------------------------------------------------------------- #
def test_mean_over_alive_uses_only_episodes_that_reached_the_index():
    traces = [np.array([1.0, 2.0, 3.0]), np.array([3.0]), np.array([5.0, 6.0])]
    mean, alive = probe.mean_over_alive(traces)
    assert alive.tolist() == [3, 2, 1]
    assert mean.tolist() == pytest.approx([3.0, 4.0, 3.0])  # exact sums of small ints


def test_mean_over_alive_rejects_empty():
    with pytest.raises(ValueError):
        probe.mean_over_alive([])
    with pytest.raises(ValueError):
        probe.mean_over_alive([np.zeros(0)])


# --------------------------------------------------------------------------- #
# rollout against a fake agent + env
# --------------------------------------------------------------------------- #
class _FakeAgent:
    """Reproduces the `_bon_record` contract: one dict per prompt group, and the
    returned chunk is candidates[best_idx]. Candidates and scores are a
    deterministic function of (env, candidate, query) so the test can tell
    which candidate was executed."""

    def __init__(self, env_num: int, m: int, groups: list[list[int]]):
        self._bon_record = None
        self.env_num, self.m, self.groups = env_num, m, groups
        self.query = 0

    def scores_for(self, e: int, q: int) -> np.ndarray:
        k = np.arange(self.m, dtype=np.float32)
        return (k - 2.0) ** 2 * 0.1 + e + q  # argmax at k = m-1 for m >= 5

    def cands_for(self, e: int, q: int) -> np.ndarray:
        c = np.zeros((self.m, 2, 3), dtype=np.float32)
        for k in range(self.m):
            c[k] = 1000 * e + 10 * q + k
        return c

    def sample_actions(self, obs, task_description=None, task_id=None):
        assert len(task_description) == self.env_num and len(task_id) == self.env_num
        q = self.query
        self.query += 1
        best = np.zeros((self.env_num, 2, 3), dtype=np.float32)
        for idx in self.groups:
            cands = np.stack([self.cands_for(e, q) for e in idx])
            scores = np.stack([self.scores_for(e, q) for e in idx])
            bi = scores.argmax(axis=1)
            if self._bon_record is not None:
                self._bon_record.append(
                    {"indices": list(idx), "candidates": cands, "scores": scores,
                     "best_idx": bi.astype(np.int32)}
                )
            best[idx] = cands[np.arange(len(idx)), bi]
        return best, np.zeros((self.env_num, 4))  # store_prefix_rep tuple form


class _FakeEnv:
    """(live, replan) step layout of the wrapped SubprocVectorEnv; env e
    terminates (success) at chunk `ends[e]` on sub-step 1, or truncates
    (failure) if ends[e] < 0 at chunk -ends[e]."""

    def __init__(self, ends: list[int], replan: int = 3):
        self.env_num, self.replan, self.ends = len(ends), replan, ends
        self.executed: list[list[np.ndarray]] = [[] for _ in ends]
        self.chunks = np.zeros(self.env_num, int)

    def step(self, act, id):
        live = list(id)
        assert act.shape[0] == len(live)
        term = np.zeros((len(live), self.replan), bool)
        trunc = np.zeros((len(live), self.replan), bool)
        for row, e in enumerate(live):
            self.executed[e].append(np.array(act[row]))
            n = self.chunks[e]
            self.chunks[e] += 1
            end = self.ends[e]
            if end >= 0 and n == end:
                term[row, 1] = True
            if end < 0 and n == -end:
                trunc[row, 1] = True
        nobs = {"x": np.full((len(live), 1), float(self.chunks.sum()))}
        rew = np.zeros((len(live), self.replan))
        return nobs, rew, term, trunc, {}


@pytest.mark.parametrize("use_bon", [True, False])
@pytest.mark.parametrize("passes", [1, 2])
def test_rollout_executes_the_chosen_candidate_and_records_scores(use_bon, passes):
    """The fake's `query` counter advances per sample_actions CALL, so with
    `passes` calls per query the k-th query's candidates come from calls
    passes*k .. passes*k+passes-1, and the fake's scores rise by 1 per call, so
    the union argmax is the LAST candidate of the LAST pass."""
    ends = [0, 2, -1, 3]  # env 0 succeeds at chunk 0, env 2 truncates at chunk 1, ...
    m = 6
    env = _FakeEnv(ends)
    agent = _FakeAgent(env_num=4, m=m, groups=[[2, 0], [1, 3]])
    obs = {"x": np.zeros((4, 1))}
    info = {"task_description": ["t"] * 4}
    succ, steps, mats, exec_idx = probe.rollout(
        env, agent, obs, info, ["task"] * 4, max_chunks=50, use_bon=use_bon, passes=passes
    )
    assert agent._bon_record is None  # disarmed after every query
    assert succ.tolist() == [True, True, False, True]
    # sub-step 1 of the terminating chunk: replan*chunks + 2 env steps.
    assert steps.tolist() == [2, 8, 5, 11]
    assert agent.query == passes * 4  # longest env: 4 chunks, `passes` calls each
    for e, end in enumerate(ends):
        n_chunks = abs(end) + 1
        assert mats[e].shape == (n_chunks, m * passes)
        assert len(env.executed[e]) == n_chunks
        for q in range(n_chunks):
            calls = [passes * q + p for p in range(passes)]
            expected = np.concatenate([agent.scores_for(e, c) for c in calls])
            assert mats[e][q] == pytest.approx(expected)
            if use_bon:
                expected_k = m * passes - 1              # last candidate, last pass
                src_call, src_k = calls[-1], m - 1
            else:
                expected_k = 0                           # candidate 0 of the first pass
                src_call, src_k = calls[0], 0
            assert exec_idx[e][q] == expected_k
            assert np.all(env.executed[e][q] == 1000 * e + 10 * src_call + src_k)


def test_rollout_rejects_zero_passes():
    env = _FakeEnv([1])
    with pytest.raises(ValueError, match="passes"):
        probe.rollout(env, _FakeAgent(1, 4, [[0]]), {"x": np.zeros((1, 1))},
                      {"task_description": ["t"]}, ["task"], max_chunks=2, use_bon=True, passes=0)


def test_single_pass_argmax_disagreement_is_caught():
    """If the recorded scores were not the ones sample_actions selected on,
    the union argmax would silently diverge from production; pin the check."""
    class _LyingAgent(_FakeAgent):
        def sample_actions(self, obs, task_description=None, task_id=None):
            best, prefix = super().sample_actions(obs, task_description, task_id)
            for r in self._bon_record:
                r["scores"] = -r["scores"]  # argmax moves; best_idx/candidates don't
            return best, prefix

    env = _FakeEnv([1, 1])
    with pytest.raises(RuntimeError, match="best_idx"):
        probe.rollout(env, _LyingAgent(2, 5, [[0, 1]]), {"x": np.zeros((2, 1))},
                      {"task_description": ["t"] * 2}, ["task"] * 2, max_chunks=2, use_bon=True)


def test_rollout_raises_when_the_hook_records_nothing():
    class _NoRecordAgent(_FakeAgent):
        def sample_actions(self, obs, task_description=None, task_id=None):
            return np.zeros((self.env_num, 2, 3), np.float32)

    env = _FakeEnv([1, 1])
    agent = _NoRecordAgent(env_num=2, m=4, groups=[[0, 1]])
    with pytest.raises(RuntimeError, match="single-sample"):
        probe.rollout(env, agent, {"x": np.zeros((2, 1))}, {"task_description": ["t"] * 2},
                      ["task"] * 2, max_chunks=5, use_bon=True)


def test_rollout_raises_on_misaligned_record():
    """A record whose candidates are in the wrong env order must not silently
    execute another env's chunk under BoN=0."""

    class _SwappedAgent(_FakeAgent):
        def sample_actions(self, obs, task_description=None, task_id=None):
            best, prefix = super().sample_actions(obs, task_description, task_id)
            for r in self._bon_record:
                r["indices"] = list(reversed(r["indices"]))
            return best, prefix

    env = _FakeEnv([1, 1])
    agent = _SwappedAgent(env_num=2, m=5, groups=[[0, 1]])
    with pytest.raises(RuntimeError, match="misaligned"):
        probe.rollout(env, agent, {"x": np.zeros((2, 1))}, {"task_description": ["t"] * 2},
                      ["task"] * 2, max_chunks=5, use_bon=False)


def test_plots_render_to_png(tmp_path):
    var = np.array([0.5, 0.7, 0.2])
    probe.plot_episode(tmp_path / "a" / "ep.png", var, "t", 8)
    mean, alive = probe.mean_over_alive([var, var[:2]])
    probe.plot_mean(tmp_path / "a" / "mean.png", mean, alive, "t", 8)
    assert (tmp_path / "a" / "ep.png").stat().st_size > 0
    assert (tmp_path / "a" / "mean.png").stat().st_size > 0
