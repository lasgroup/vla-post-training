"""ResumeStore: replay-buffer chunks + agent checkpoints round-trip through a directory standing in for S3."""
import json
import os

import jax
import numpy as np
import pytest

from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.log_utils import CsvLogger
from ogpo.utils.resume import ResumeStore, restore_tree, truncate_csvs

UPR = 100  # updates per round


def _buffer(size):
    ex = dict(observations=np.zeros(3, np.float32), images=np.zeros((4, 4, 3), np.uint8), actions=np.zeros(2, np.float32),
              rewards=np.array(0.0, np.float32), terminals=np.array(0.0, np.float32), masks=np.array(1.0, np.float32),
              next_observations=np.zeros(3, np.float32), next_images=np.zeros((4, 4, 3), np.uint8),
              mc_returns=np.array(0.0, np.float32), task_ids=np.array(0, np.int32))
    return ReplayBuffer.create(ex, size, numpy_rng=np.random.default_rng(0))


def _add(rb, n, tag):
    for i in range(n):
        rb.add_transition(dict(observations=np.full(3, tag + i, np.float32), images=np.full((4, 4, 3), (tag + i) % 255, np.uint8),
                               actions=np.full(2, tag, np.float32), rewards=np.array(-1.0, np.float32),
                               terminals=np.array(float(i == n - 1), np.float32), masks=np.array(1.0, np.float32),
                               next_observations=np.full(3, tag + i + 1, np.float32),
                               next_images=np.full((4, 4, 3), (tag + i + 1) % 255, np.uint8),
                               mc_returns=np.array(float(tag), np.float32), task_ids=np.array(tag % 3, np.int32),
                               is_success=float(tag % 2)))


def _store(tmp_path, name="s"):
    return ResumeStore(tmp_path / f"{name}_local", tmp_path / "remote", UPR, ckpt_interval=50)


def _agent(step):
    return {"step": step, "w": np.full(4, step, np.float32)}


def _csv(tmp_path, step):
    p = tmp_path / "csv_src"
    p.mkdir(exist_ok=True)
    (p / "eval.csv").write_text("success,step\n" + "".join(f"0.5,{s}\n" for s in range(0, step + 1, 50)))
    (p / "env.csv").write_text("success,step\n" + "".join(f"0.5,{s}\n" for s in range(0, step + 1, UPR)))
    return [str(p / "eval.csv"), str(p / "env.csv")]


def test_chunks_roundtrip_through_ring_wraparound(tmp_path):
    rb, st = _buffer(10), _store(tmp_path)
    st.save_agent(_agent(0), 0, _csv(tmp_path, 0))
    for r, tag in enumerate((10, 20, 30, 40)):  # 4 rows per round into a ring of 10 -> wraps in round 2
        p0, t0 = rb.pointer, rb.total_added
        _add(rb, 4, tag)
        st.save_chunk(rb, r, p0, t0, {"episodes": 4 * (r + 1), "env_steps": 40 * (r + 1)})
        st.save_agent(_agent((r + 1) * UPR), (r + 1) * UPR, _csv(tmp_path, (r + 1) * UPR))
    assert st.wait() is None
    man = _store(tmp_path, "new").manifest()
    assert (man["round"], man["step"]) == (3, 400)
    assert man["chunks"] == [1, 2, 3]  # 12 rows cover the 10-slot ring; chunk 0 is pruned from the remote
    assert not os.path.exists(tmp_path / "remote" / "chunks" / "c0000")

    fresh = _store(tmp_path, "new")
    state, csv_dir, dirs = fresh.fetch(man)
    rb2 = _buffer(10)
    meta = fresh.load_buffer(rb2, dirs)
    assert state["step"] == 400 and meta["episodes"] == 16
    assert (rb2.pointer, rb2.size, rb2.total_added) == (rb.pointer, rb.size, rb.total_added)
    for k in rb._dict:
        np.testing.assert_array_equal(rb2._dict[k], rb._dict[k], err_msg=k)
    np.testing.assert_array_equal(rb2._traj_success_mask, rb._traj_success_mask)


def test_manifest_only_advances_to_a_consistent_pair(tmp_path):
    rb, st = _buffer(100), _store(tmp_path)
    st.save_agent(_agent(0), 0, _csv(tmp_path, 0))
    _add(rb, 4, 10)
    st.save_chunk(rb, 0, 0, 0, {"episodes": 1, "env_steps": 4})
    st.save_agent(_agent(50), 50, _csv(tmp_path, 50))
    st.wait()
    assert (st.manifest()["round"], st.manifest()["step"]) == (0, 50)
    st.save_agent(_agent(150), 150, _csv(tmp_path, 150))  # past round 0's updates: chunk 1 is not there yet
    st.wait()
    assert (st.manifest()["round"], st.manifest()["step"]) == (0, 50)
    p0, t0 = rb.pointer, rb.total_added
    _add(rb, 4, 20)
    st.save_chunk(rb, 1, p0, t0, {"episodes": 2, "env_steps": 8})
    st.wait()
    assert (st.manifest()["round"], st.manifest()["step"]) == (1, 150)
    assert not os.path.exists(tmp_path / "remote" / "agent" / "step0000050.pkl")  # superseded checkpoints are removed


def test_truncate_csvs_and_append_logger(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "eval.csv").write_text("success,step\n0.1,0\n0.2,50\n0.3,100\n0.4,150\n")
    (src / "env.csv").write_text("success,step\n0.5,0\n0.6,100\n0.7,200\n")
    truncate_csvs(str(src), str(dst), 100, keep_at_step=False)  # round 1 will be re-collected
    assert (dst / "eval.csv").read_text() == "success,step\n0.1,0\n0.2,50\n0.3,100\n"
    assert (dst / "env.csv").read_text() == "success,step\n0.5,0\n"
    lg = CsvLogger(str(dst / "eval.csv"), append=True)
    lg.log({"success": 0.9}, 150)
    lg.close()
    assert (dst / "eval.csv").read_text().splitlines()[-1] == "0.9,150"
    fresh = CsvLogger(str(dst / "none.csv"), append=True)  # nothing to continue: starts a file with a header
    fresh.log({"success": 1.0}, 1)
    fresh.close()
    assert (dst / "none.csv").read_text() == "success,step\n1.0,1\n"


def test_restore_tree_keeps_dtype_and_rejects_mismatches():
    cur = {"a": jax.numpy.zeros((2, 3), jax.numpy.bfloat16), "b": [jax.numpy.zeros(4, jax.numpy.float32)]}
    out = restore_tree(cur, {"a": np.ones((2, 3), np.float32), "b": [np.arange(4, dtype=np.float32)]})
    assert out["a"].dtype == jax.numpy.bfloat16 and float(out["b"][0][3]) == 3.0
    with pytest.raises(ValueError, match="shape"):
        restore_tree(cur, {"a": np.ones((3, 3), np.float32), "b": [np.zeros(4, np.float32)]})
    with pytest.raises(ValueError, match="tree"):
        restore_tree(cur, {"a": np.ones((2, 3), np.float32)})


# ----------------------------------------------------------------------------- the round runner, killed and restarted
class _Crash(Exception):
    pass


def _run_rounds(monkeypatch, store, crash_at=None, save_dir=None, num_rounds=3, upr=None):
    """3 rounds x 4 updates on the stub agent; `crash_at=(prefix, step)` raises when the logger gets that row."""
    upr = upr or UPR_RUN
    import ogpo.runners.pi05_online_runner as R
    from tests._pi05_stubs import Logger, StubAgent, TaskEnv, reg

    monkeypatch.setattr(R.T, "resize_images", lambda x: x)
    monkeypatch.setattr(R, "_agent_state", lambda agent, step: {"step": step})

    def fake_resume(resume, man, agent, rb, save_dir, upr, seed):  # the agent trees need the real pi05 agent
        state, csv_dir, dirs = resume.fetch(man)
        meta = resume.load_buffer(rb, dirs)
        rnd = (state["step"] - resume.step_offset) // upr
        truncate_csvs(csv_dir, save_dir, state["step"], keep_at_step=rnd == man["round"])
        return agent, state["step"], meta["episodes"], meta["env_steps"]

    monkeypatch.setattr(R, "_resume_state", fake_resume)
    logger, tr = Logger(), reg()
    base_log = logger.log

    def log(data, prefix, step):
        if crash_at == (prefix, step):
            raise _Crash
        base_log(data, prefix, step)

    logger.log = log
    try:
        _, rb = R.run_pi05_round_rl(
            StubAgent([]), TaskEnv(0, tr), TaskEnv(0, reg()), {"ppo_batch_size": 2}, logger, num_rounds=num_rounds,
            rollouts_per_round=2, updates_per_round=upr, eval_interval=0, eval_episodes=1, log_interval=2, batch_size=2, success_buffer_batch_size=2,
            horizon_length=3, bc_horizon=4, discount=0.99, buffer_size=2000, seed=3, save_dir=save_dir, resume=store)
    except _Crash:
        rb = None
    return rb, logger.logs, tr


UPR_RUN = 4


def _resume_store(tmp_path, name):
    return ResumeStore(tmp_path / f"{name}_local", tmp_path / "remote", UPR_RUN, ckpt_interval=2)


@pytest.mark.parametrize("crash_at,resumed_round_collects", [(("env", 8), True), (("online_agent", 10), False)])
def test_round_runner_resumes_after_a_crash(tmp_path, monkeypatch, crash_at, resumed_round_collects):
    # uninterrupted reference
    ref_rb, ref_logs, _ = _run_rounds(monkeypatch, None)
    store = _resume_store(tmp_path, "a")
    assert _run_rounds(monkeypatch, store, crash_at, save_dir=str(tmp_path / "save_a"))[0] is None
    assert store.wait() is None
    man = _resume_store(tmp_path, "peek").manifest()
    assert (man["round"], man["step"]) == ((1, 8) if resumed_round_collects else (2, 8))

    store2 = _resume_store(tmp_path, "b")
    rb, logs, tr = _run_rounds(monkeypatch, store2, save_dir=str(tmp_path / "save_b"))
    assert rb.size == ref_rb.size and rb.total_added == ref_rb.total_added
    ref_updates = [s for p, s, _ in ref_logs if p == "online_agent"]
    assert max(s for p, s, _ in logs if p == "online_agent") == max(ref_updates) == 12
    # episodes continue where the restored state stopped: no reset seed is reused, none skipped
    seeds = [s for _, s in tr["resets"]]
    assert len(seeds) == len(set(seeds))
    # collection ran again only for the rounds missing from the restored buffer
    collected_rounds = [d["round"] for p, _, d in logs if p == "env"]
    assert collected_rounds == ([2] if resumed_round_collects else [])


@pytest.mark.parametrize("crash_at", [None, ("env", 20), ("online_agent", 24)])
def test_rebase_rounds_continues_a_finished_run_with_a_new_round_length(tmp_path, monkeypatch, crash_at):
    # 3 rounds x 4 updates, finished: manifest (round 2, step 12) without a step_offset, as the pre-rebase runs wrote
    store = _resume_store(tmp_path, "a")
    _run_rounds(monkeypatch, store, save_dir=str(tmp_path / "save_a"))
    assert store.wait() is None
    man = _resume_store(tmp_path, "peek").manifest()
    assert (man["round"], man["step"]) == (2, 12)
    man.pop("step_offset")
    (tmp_path / "remote" / "MANIFEST.json").write_text(json.dumps(man))

    # continue to 5 rounds with 8 updates per round: rounds 3 and 4 collect, then 8 updates each -> step 28
    def rebased(name):
        return ResumeStore(tmp_path / f"{name}_local", tmp_path / "remote", 8, ckpt_interval=2, rebase_rounds=True)

    if crash_at is not None:  # killed mid-way (after round 3's collection, or mid round 4's updates), then restarted
        crashed = rebased("b")
        assert _run_rounds(monkeypatch, crashed, crash_at, save_dir=str(tmp_path / "save_b"), num_rounds=5, upr=8)[0] is None
        assert crashed.wait() is None
        man = _resume_store(tmp_path, "peek2").manifest()
        assert man["step_offset"] == 12 - 3 * 8
    final = rebased("c")
    rb, logs, _ = _run_rounds(monkeypatch, final, save_dir=str(tmp_path / "save_c"), num_rounds=5, upr=8)
    assert final.wait() is None
    updates = [s for p, s, _ in logs if p == "online_agent"]
    assert max(updates) == 28 and min(updates) > 12
    collected = [d["round"] for p, _, d in logs if p == "env"]
    assert collected == {None: [3, 4], ("env", 20): [4], ("online_agent", 24): []}[crash_at]
    assert [s for p, s, _ in logs if p == "env"] == {None: [12, 20], ("env", 20): [20], ("online_agent", 24): []}[crash_at]
    end = _resume_store(tmp_path, "peek3").manifest()
    assert (end["round"], end["step"], end["step_offset"]) == (4, 28, -12)
