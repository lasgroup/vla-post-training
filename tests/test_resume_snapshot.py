"""run_pi05_round_rl(resume_snapshot_dir=...): continue a run from agent.pkl + buffer dump + CSV logs (docs/changes/2026-10-27-pi05-snapshot-resume).

A stub-agent run is cut inside round 1's update phase; the buffer is dumped (save_buffer_dump) as a live snapshot would be, the
agent state is the one at the start of round 1, and a second run continues from that. It must collect only round 2, finish at
update 12 with the reference run's buffer counters, keep the first two rounds' rows bit-identical and the reset seeds
continuing where the snapshot stopped; inconsistent pairs and flag combinations raise.

Run:  JAX_PLATFORMS=cpu pytest tests/test_resume_snapshot.py -q
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

from ogpo.utils.resume import save_buffer_dump

UPR = 4
KW = dict(num_rounds=3, rollouts_per_round=2, updates_per_round=UPR, eval_interval=0, eval_episodes=1, log_interval=2,
          batch_size=2, success_buffer_batch_size=2, horizon_length=3, bc_horizon=4, discount=0.99, buffer_size=2000, seed=3)


class _Crash(Exception):
    pass


def _run(monkeypatch, crash_at=None, save_dir=None, **extra):
    import ogpo.runners.pi05_online_runner as R
    from _pi05_stubs import Logger, StubAgent, TaskEnv, reg  # tests/ is on sys.path; `tests` is shadowed by a site-packages package in .venv-molmo

    monkeypatch.setattr(R.T, "resize_images", lambda x: x)
    monkeypatch.setattr(R, "_restore_agent", lambda agent, state: agent)  # the agent trees need the real pi05 agent
    made = []
    real_make = R._make_buffer
    monkeypatch.setattr(R, "_make_buffer", lambda *a, **k: made.append(real_make(*a, **k)) or made[-1])
    logger, tr = Logger(), reg()
    base_log = logger.log

    def log(data, prefix, step):
        if crash_at == (prefix, step):
            raise _Crash
        base_log(data, prefix, step)

    logger.log = log
    rb = None
    try:
        _, rb = R.run_pi05_round_rl(StubAgent([]), TaskEnv(0, tr), TaskEnv(0, reg()), {"ppo_batch_size": 2}, logger,
                                    save_dir=save_dir, **KW, **extra)
    except _Crash:
        pass
    return rb, made[-1], logger.logs, tr


def _snapshot(monkeypatch, tmp_path):
    """Cut a run inside round 1's updates and write a live-run snapshot (agent.pkl, buffer/, csv/)."""
    _, rb, logs, tr = _run(monkeypatch, crash_at=("online_agent", 6))
    env_rows = [d for p, _, d in logs if p == "env"]
    assert [d["round"] for d in env_rows] == [0, 1]
    snap = tmp_path / "snap"
    save_buffer_dump(rb, str(snap / "buffer"), dict(round=1, episodes=env_rows[-1]["episodes"], env_steps=env_rows[-1]["env_steps"]))
    with open(snap / "agent.pkl", "wb") as f:
        pickle.dump({"step": UPR}, f)
    (snap / "csv").mkdir()
    (snap / "csv" / "env.csv").write_text("success,step\n" + "".join(f"0.5,{s}\n" for s in (0, UPR)))
    (snap / "csv" / "online_agent.csv").write_text("loss,step\n" + "".join(f"1.0,{s}\n" for s in (2, 4, 6)))  # 6 is past the snapshot
    return snap, rb, tr


def test_snapshot_resume_continues_the_run(tmp_path, monkeypatch):
    ref_rb, _, ref_logs, ref_tr = _run(monkeypatch)
    snap, cut_rb, _ = _snapshot(monkeypatch, tmp_path)
    rb, _, logs, tr = _run(monkeypatch, save_dir=str(tmp_path / "save"), resume_snapshot_dir=str(snap))
    assert (rb.size, rb.total_added, rb.pointer) == (ref_rb.size, ref_rb.total_added, ref_rb.pointer)
    n = cut_rb.size  # rows of rounds 0 and 1 come from the dump, bit for bit
    for k in ref_rb._dict:
        assert np.array_equal(rb._dict[k][:n], cut_rb._dict[k][:n]), k
    assert np.array_equal(rb._traj_success_mask[:n], cut_rb._traj_success_mask[:n])
    assert [d["round"] for p, _, d in logs if p == "env"] == [2]  # only the missing round is collected
    assert sorted({s for p, s, _ in logs if p == "online_agent"}) == [6, 8, 10, 12]  # round 1's updates continue after step 4
    ref_seeds, seeds = [s for _, s in ref_tr["resets"]], [s for _, s in tr["resets"]]
    assert seeds[1:] == ref_seeds[len(ref_seeds) - len(seeds) + 1:]  # round 2's reset seeds, as in the uninterrupted run
    # CSVs: rows past the snapshot are dropped, the env row of the resumed round is kept
    assert (tmp_path / "save" / "env.csv").read_text().splitlines() == ["success,step", "0.5,0", "0.5,4"]
    assert (tmp_path / "save" / "online_agent.csv").read_text().splitlines() == ["loss,step", "1.0,2", "1.0,4"]


def test_inconsistent_pair_and_flag_combinations_raise(tmp_path, monkeypatch):
    snap, _, _ = _snapshot(monkeypatch, tmp_path)
    with open(snap / "agent.pkl", "wb") as f:
        pickle.dump({"step": 2 * UPR}, f)  # agent at round 2, buffer after round 1
    with pytest.raises(ValueError, match="not a consistent pair"):
        _run(monkeypatch, save_dir=str(tmp_path / "s1"), resume_snapshot_dir=str(snap))
    with pytest.raises(ValueError, match="exclusive"):
        _run(monkeypatch, save_dir=str(tmp_path / "s2"), resume_snapshot_dir=str(snap), resume=object())
    with pytest.raises(ValueError, match="buffer_reuse_storage needs"):
        _run(monkeypatch, save_dir=str(tmp_path / "s3"), buffer_reuse_storage=True)
    with pytest.raises(ValueError, match="needs buffer_reuse_storage"):
        _run(monkeypatch, save_dir=str(tmp_path / "s4"), resume_snapshot_dir=str(snap), buffer_storage_dir=str(tmp_path))
    with pytest.raises(ValueError, match="experiment.log"):
        _run(monkeypatch, resume_snapshot_dir=str(snap))


# ----------------------------------------------------------------------------- _restore_agent replaces buffers in place
def _stand_in_agent(with_value=False):
    import flax.struct
    import jax.numpy as jnp

    class Net(flax.struct.PyTreeNode):  # pytrees like the real TrainState / OGPOPi05Agent (flax.struct.PyTreeNode)
        params: dict
        opt_state: dict
        step: object

    class Agent(flax.struct.PyTreeNode):
        actor_params: dict
        target_actor_params: dict
        actor_opt_state: dict
        critic_network: Net
        value_network: object
        frozen_params: dict

    shared = {"w": jnp.full((3,), 7.0)}  # same object in a replaced and in an unreplaced field: must survive
    net = lambda v: Net({"w": jnp.zeros((2,))}, {"mu": jnp.zeros((2,))}, jnp.array(v))
    agent = Agent(actor_params={"w": jnp.zeros((4,)), "shared": shared["w"]}, target_actor_params={"w": jnp.zeros((4,))},
                  actor_opt_state={"mu": jnp.zeros((4,))}, critic_network=net(0), value_network=net(0) if with_value else None,
                  frozen_params=shared)
    state = {"actor_params": {"w": np.ones(4, np.float32), "shared": np.full(3, 9.0, np.float32)},
             "target_actor_params": {"w": np.full(4, 2.0, np.float32)}, "actor_opt_state": {"mu": np.full(4, 3.0, np.float32)},
             "critic_params": {"w": np.full(2, 4.0, np.float32)}, "critic_opt_state": {"mu": np.full(2, 5.0, np.float32)},
             "critic_step": np.array(11)}
    if with_value:
        state.update(value_params={"w": np.full(2, 6.0, np.float32)}, value_opt_state={"mu": np.full(2, 7.0, np.float32)},
                     value_step=np.array(13))
    return agent, state


@pytest.mark.parametrize("with_value", [False, True])
def test_restore_agent_replaces_buffers_in_place_and_keeps_shared_ones(with_value):
    import jax

    import ogpo.runners.pi05_online_runner as R

    agent, state = _stand_in_agent(with_value)
    old = [agent.actor_params["w"], agent.target_actor_params["w"], agent.actor_opt_state["mu"],
           agent.critic_network.params["w"], agent.critic_network.opt_state["mu"], agent.critic_network.step]
    if with_value:
        old += [agent.value_network.params["w"], agent.value_network.opt_state["mu"], agent.value_network.step]
    shared = agent.frozen_params["w"]
    new = R._restore_agent(agent, state)
    assert all(x.is_deleted() for x in old)  # the live copy is gone, not stacked next to the restored one
    assert agent.actor_params["shared"] is shared and not shared.is_deleted()  # frozen_params of `new` still holds it
    assert new.frozen_params["w"] is shared and float(shared[0]) == 7.0
    assert float(new.actor_params["w"][0]) == 1.0 and float(new.actor_params["shared"][0]) == 9.0
    assert float(new.target_actor_params["w"][0]) == 2.0 and float(new.actor_opt_state["mu"][0]) == 3.0
    assert float(new.critic_network.params["w"][0]) == 4.0 and float(new.critic_network.opt_state["mu"][0]) == 5.0
    assert int(new.critic_network.step) == 11 and isinstance(new.critic_network.step, jax.Array)
    assert new.actor_params["w"].dtype == np.float32 and new.critic_network.step.dtype == old[-1 if not with_value else 5].dtype
    if with_value:
        assert float(new.value_network.params["w"][0]) == 6.0 and float(new.value_network.opt_state["mu"][0]) == 7.0
        assert int(new.value_network.step) == 13
    else:
        assert new.value_network is None


@pytest.mark.parametrize("bad", ["critic_params", "actor_opt_state"])
def test_restore_agent_validates_everything_before_deleting_anything(bad):
    import ogpo.runners.pi05_online_runner as R

    agent, state = _stand_in_agent()
    state[bad] = {"w": np.ones(5, np.float32)} if bad == "critic_params" else {"nu": np.ones(4, np.float32)}  # shape / structure
    with pytest.raises(ValueError, match="leaf shape|tree does not match"):
        R._restore_agent(agent, state)
    assert not agent.actor_params["w"].is_deleted() and not agent.critic_network.params["w"].is_deleted()
    assert not agent.actor_opt_state["mu"].is_deleted()
