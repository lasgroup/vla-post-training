# ruff: noqa: F722
"""Tests for the privileged, per-task OGPO critic.

Everything here runs on CPU without pi05 weights: the LIBERO privileged-state
extraction, the observation wrapper that emits it, the per-task critic ensemble,
the replay-buffer schema, the next-action alignment feeding the Q backup, and
the config/dispatch wiring. The pieces that need a real model (the sampler jit's
dict sidecar) are checked structurally, the way the rest of tests/ogpo does.
"""
import ast
import inspect
import pathlib

import flax.nnx as nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import src.training.config as _config
from src.envs.wrappers import Pi0ObservationWrapper
from src.rl.networks.task_ensemble_critic import (
    TASK_ID_KEY,
    TaskEnsembleStateActionCritic,
    TaskEnsembleStateValue,
    _select_task,
    clip_by_global_norm_per_task,
    make_task_ensemble_defs,
    per_task_optimizer,
)
from src.rl.ogpo.privileged.ogpo_privileged_learner import OGPOPrivilegedLearner
from src.rl.privileged_state import (
    PRIVILEGED_STATE_NAME,
    PRIVILEGED_TASK_ID_NAME,
    UNKNOWN_TASK_ID,
    extract_libero_privileged_state,
    privileged_task_ids,
)
from src.rl.replay_buffer import ShardedReplayBuffer


_ROOT = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# 1. Privileged state extraction
# --------------------------------------------------------------------------------------


def _raw_libero_obs(n_objects: int = 3, proprio: int = 39) -> dict:
    """The two robosuite modality concatenations LIBERO's observation carries."""
    rng = np.random.RandomState(0)
    return {
        "robot0_proprio-state": rng.randn(proprio).astype(np.float32),
        "object-state": rng.randn(14 * n_objects).astype(np.float32),
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_extract_concatenates_proprio_then_objects_and_zero_pads():
    obs = _raw_libero_obs(n_objects=3)
    out = extract_libero_privileged_state(obs, dim=128)
    assert out.shape == (128,)
    assert out.dtype == np.float32
    expected = np.concatenate([obs["robot0_proprio-state"], obs["object-state"]])
    np.testing.assert_allclose(out[: expected.size], expected)
    # Everything past the real width is exactly zero, so a per-task critic sees
    # a constant there rather than stale bytes.
    np.testing.assert_array_equal(out[expected.size :], 0.0)


def test_extract_is_width_stable_across_tasks_with_different_object_counts():
    a = extract_libero_privileged_state(_raw_libero_obs(n_objects=2), dim=256)
    b = extract_libero_privileged_state(_raw_libero_obs(n_objects=9), dim=256)
    assert a.shape == b.shape == (256,)


def test_extract_raises_rather_than_silently_truncating_an_oversized_state():
    with pytest.raises(ValueError, match="privileged_state_dim"):
        extract_libero_privileged_state(_raw_libero_obs(n_objects=9), dim=32)


def test_extract_raises_on_a_missing_modality_instead_of_zero_filling_it():
    obs = _raw_libero_obs()
    del obs["object-state"]
    with pytest.raises(KeyError, match="object-state"):
        extract_libero_privileged_state(obs, dim=128)


def test_task_ids_dedupe_but_preserve_collection_order():
    # `collect.tasks` repeats a task to weight collection (the `x4` multiplier);
    # the critic gets one head per DISTINCT task.
    assert privileged_task_ids(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


# --------------------------------------------------------------------------------------
# 2. The observation wrapper
# --------------------------------------------------------------------------------------


class _FakeLiberoEnv(gym.Env):
    """Emits a raw LIBERO-shaped observation and records the reset options."""

    def __init__(self, n_objects: int = 3):
        self._n_objects = n_objects
        self.last_options = None

    def _obs(self):
        obs = _raw_libero_obs(self._n_objects)
        obs["robot0_eef_pos"] = np.zeros(3, dtype=np.float32)
        obs["robot0_eef_quat"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs["robot0_gripper_qpos"] = np.zeros(2, dtype=np.float32)
        obs["robot0_eye_in_hand_image"] = np.zeros((8, 8, 3), dtype=np.uint8)
        return obs

    def reset(self, *, seed=None, options=None):
        self.last_options = options
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}


def test_wrapper_is_off_by_default_and_emits_only_the_policy_keys():
    env = Pi0ObservationWrapper(env=_FakeLiberoEnv(), env_class="libero")
    obs, _ = env.reset()
    assert set(obs) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
    }


def test_wrapper_emits_privileged_state_and_the_reset_task_index():
    env = Pi0ObservationWrapper(
        env=_FakeLiberoEnv(),
        env_class="libero",
        privileged_tasks=["libero_90_79", "libero_90_31"],
        privileged_state_dim=128,
    )
    obs, _ = env.reset(options={"task_id": "libero_90_31"})
    assert obs[f"observation/{PRIVILEGED_STATE_NAME}"].shape == (128,)
    np.testing.assert_array_equal(
        obs[f"observation/{PRIVILEGED_TASK_ID_NAME}"], np.array([1.0], np.float32)
    )
    # The id is latched at reset, so it survives stepping.
    step_obs, *_ = env.step(np.zeros(7))
    np.testing.assert_array_equal(
        step_obs[f"observation/{PRIVILEGED_TASK_ID_NAME}"], np.array([1.0], np.float32)
    )


def test_wrapper_marks_a_task_outside_the_train_list_as_unknown():
    # The held-out eval block has no critic head; -1 must reach the critic so it
    # can one-hot to zero rather than borrow task 0's head.
    env = Pi0ObservationWrapper(
        env=_FakeLiberoEnv(),
        env_class="libero",
        privileged_tasks=["libero_90_79"],
        privileged_state_dim=128,
    )
    obs, _ = env.reset(options={"task_id": "libero_90_4"})
    assert obs[f"observation/{PRIVILEGED_TASK_ID_NAME}"][0] == UNKNOWN_TASK_ID


def test_wrapper_rejects_a_non_libero_domain_rather_than_emitting_junk():
    with pytest.raises(NotImplementedError):
        Pi0ObservationWrapper(
            env=_FakeLiberoEnv(),
            env_class="molmo",
            privileged_tasks=["a"],
            privileged_state_dim=128,
        )


# --------------------------------------------------------------------------------------
# 3. The per-task critic ensemble
# --------------------------------------------------------------------------------------


def test_select_task_picks_each_sample_s_own_task_row():
    T, N, B = 3, 2, 5
    per_task = jnp.asarray(np.arange(T * N * B, dtype=np.float32).reshape(T, N, B))
    ids = jnp.asarray([0.0, 2.0, 1.0, 0.0, 2.0])
    got = _select_task(per_task, ids, T)
    assert got.shape == (N, B)
    for b, t in enumerate([0, 2, 1, 0, 2]):
        np.testing.assert_allclose(np.asarray(got)[:, b], np.asarray(per_task)[t, :, b])


def test_select_task_returns_zero_for_an_out_of_range_task_id():
    per_task = jnp.ones((3, 2, 4))
    got = _select_task(per_task, jnp.asarray([-1.0, 0.0, -1.0, 2.0]), 3)
    np.testing.assert_allclose(np.asarray(got)[:, [0, 2]], 0.0)
    np.testing.assert_allclose(np.asarray(got)[:, [1, 3]], 1.0)


def _small_critic_obs(batch: int, state_dim: int = 16, ids=None):
    ids = np.zeros(batch, dtype=np.float32) if ids is None else np.asarray(ids, np.float32)
    return {
        "state": jnp.asarray(np.random.RandomState(1).randn(batch, state_dim), jnp.float32),
        TASK_ID_KEY: jnp.asarray(ids.reshape(batch, 1)),
    }


def test_task_ensemble_q_matches_the_single_task_output_shape():
    obs = _small_critic_obs(batch=6, ids=[0, 1, 2, 0, 1, 2])
    act = jnp.asarray(np.random.RandomState(2).randn(6, 30), jnp.float32)
    critic = TaskEnsembleStateActionCritic(
        observation=obs, action=act, hidden_dim=8, depth=1, num_qs=2, num_tasks=3,
        rngs=nnx.Rngs(0),
    )
    out = critic(obs, act)
    assert out.shape == (2, 6)  # (num_qs, B), exactly the single-task convention


def test_task_ensemble_value_matches_the_single_task_output_shape():
    obs = _small_critic_obs(batch=4, ids=[0, 1, 1, 0])
    critic = TaskEnsembleStateValue(
        observation=obs, hidden_dim=8, depth=1, num_vs=3, num_tasks=2, rngs=nnx.Rngs(0)
    )
    assert critic(obs).shape == (3, 4)


def test_only_the_selected_task_s_parameters_receive_gradient():
    """The whole point of per-task heads: task 1's transitions must not move
    task 0's critic. The T-fold forward is a compute cost, not a coupling."""
    obs = _small_critic_obs(batch=4, ids=[1, 1, 1, 1])
    act = jnp.asarray(np.random.RandomState(3).randn(4, 30), jnp.float32)
    critic = TaskEnsembleStateActionCritic(
        observation=obs, action=act, hidden_dim=8, depth=1, num_qs=1, num_tasks=3,
        rngs=nnx.Rngs(0),
    )

    def loss_fn(m):
        return jnp.sum(m(obs, act) ** 2)

    grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(critic)
    flat = dict(grads.flat_state())
    per_task_norm = {0: 0.0, 1: 0.0, 2: 0.0}
    for path, leaf in flat.items():
        # path[0] == "nets", path[1] == task index
        per_task_norm[int(path[1])] += float(jnp.sum(jnp.abs(leaf.value)))
    assert per_task_norm[1] > 0.0
    assert per_task_norm[0] == 0.0
    assert per_task_norm[2] == 0.0


def test_task_heads_are_independent_parameter_sets():
    obs = _small_critic_obs(batch=2)
    act = jnp.asarray(np.zeros((2, 30)), jnp.float32)
    critic = TaskEnsembleStateActionCritic(
        observation=obs, action=act, hidden_dim=8, depth=1, num_qs=1, num_tasks=4,
        rngs=nnx.Rngs(0),
    )
    task_indices = {int(path[1]) for path in dict(nnx.state(critic).flat_state())}
    assert task_indices == {0, 1, 2, 3}
    # Distinct rng draws per head, so the same state maps to different Qs.
    ones = {"state": jnp.ones((1, 16)), TASK_ID_KEY: jnp.zeros((1, 1))}
    a = float(critic(ones, jnp.zeros((1, 30)))[0, 0])
    b = float(
        critic({**ones, TASK_ID_KEY: jnp.ones((1, 1))}, jnp.zeros((1, 30)))[0, 0]
    )
    assert a != b


def test_make_task_ensemble_defs_matches_the_learner_calling_convention():
    q_def, v_def = make_task_ensemble_defs(
        hidden_dim=8, depth=1, num_qs=2, num_vs=2, num_tasks=2
    )
    obs = _small_critic_obs(batch=2)
    act = jnp.zeros((2, 30), jnp.float32)
    assert q_def(obs, act, nnx.Rngs(0))(obs, act).shape == (2, 2)
    assert v_def(obs, nnx.Rngs(0))(obs).shape == (2, 2)


def test_task_ensemble_rejects_a_zero_task_run():
    with pytest.raises(ValueError, match="num_tasks"):
        TaskEnsembleStateValue(
            observation=_small_critic_obs(batch=1), hidden_dim=8, depth=1, num_vs=1,
            num_tasks=0, rngs=nnx.Rngs(0),
        )


# --------------------------------------------------------------------------------------
# 4. Buffer schema and the next-action alignment
# --------------------------------------------------------------------------------------


def test_next_actions_are_the_action_window_at_the_next_observation():
    """`next_obs_index == obs_index + act_h`, so `next_actions[i]` must be the
    action window whose observation index is `i + act_h`. Getting this wrong
    would silently make the backup off-by-one — a plausible-looking Q that
    bootstraps through the wrong state's action."""
    act_h, n_windows = 3, 5
    # actions_out spans n_windows + act_h windows; row j is the window at obs j.
    actions_out = np.arange(n_windows + act_h, dtype=np.float32)[:, None, None] * np.ones(
        (1, act_h, 2), dtype=np.float32
    )
    got = _FakeLearner("next_action_q")._extra_transition_fields(
        actions_out=actions_out, n_windows=n_windows, act_h=act_h
    )["next_actions"]
    assert got.shape == (n_windows, act_h, 2)
    for i in range(n_windows):
        np.testing.assert_allclose(got[i], actions_out[i + act_h])


def test_base_learner_adds_no_extra_transition_fields():
    from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner

    assert (
        FilteredSFTLearner._extra_transition_fields(
            None, actions_out=np.zeros((4, 2, 2)), n_windows=2, act_h=2
        )
        == {}
    )


def test_replay_buffer_round_trips_the_privileged_schema():
    n, act_h, adim, sdim = 6, 2, 4, 16
    payload = {
        "observations": {
            "state": np.random.rand(n + act_h, 8).astype(np.float32),
            PRIVILEGED_STATE_NAME: np.random.rand(n + act_h, sdim).astype(np.float32),
            PRIVILEGED_TASK_ID_NAME: np.ones((n + act_h, 1), np.float32),
        },
        "obs_index": np.arange(n, dtype=np.int64),
        "next_obs_index": np.arange(n, dtype=np.int64) + act_h,
        "actions": np.random.rand(n, act_h, adim).astype(np.float32),
        "next_actions": np.random.rand(n, act_h, adim).astype(np.float32),
        "reward": np.zeros(n, np.float32),
        "mc_return": np.zeros(n, np.float32),
        "discount": np.zeros(n, np.float32),
        "is_success": np.zeros(n, np.float32),
    }
    buf = ShardedReplayBuffer(dummy_data=payload, max_capacity=32, seed=0, freeze_dict=False)
    buf.insert(payload)
    batch = buf.sample(batch_size=4)
    assert batch["observation"][PRIVILEGED_STATE_NAME].shape == (4, sdim)
    assert batch["observation"][PRIVILEGED_TASK_ID_NAME].shape == (4, 1)
    assert batch["next_observation"][PRIVILEGED_STATE_NAME].shape == (4, sdim)
    assert batch["next_actions"].shape == (4, act_h, adim)
    # The critic drops images by key; privileged columns must survive that.
    dropped = buf.sample(batch_size=4, drop_obs_keys=("image", "image_mask"))
    assert PRIVILEGED_STATE_NAME in dropped["observation"]


def test_privileged_learner_declares_both_privileged_columns_as_passthrough():
    """`_policy_transforms`' repack drops any key it does not map, so a critic-only
    observation column has to bypass it."""
    assert set(OGPOPrivilegedLearner._buffer_obs_passthrough_keys) == {
        PRIVILEGED_STATE_NAME,
        PRIVILEGED_TASK_ID_NAME,
    }


# --------------------------------------------------------------------------------------
# 5. Config + wiring
# --------------------------------------------------------------------------------------


def test_registered_privileged_config_is_the_baseline_plus_the_critic_swap():
    priv = _config.get_config("pi05_libero_online_ogpo_privileged")
    base = _config.get_config("pi05_libero_online_ogpo_sft")
    assert isinstance(priv.rl, _config.OGPOPrivilegedLearnerConfig)
    assert priv.collect.store_privileged_state is True
    assert base.collect.store_privileged_state is False
    # The actor-side recipe is untouched, so a paired run isolates the critic.
    for field in (
        "group_num_samples", "clip_epsilon", "bc_coeff", "num_sde_steps",
        "noise_level", "adv_strategy", "discount", "online_ratio", "n_samples",
    ):
        assert getattr(priv.rl, field) == getattr(base.rl, field), field
    assert priv.rl.policy == base.rl.policy
    assert priv.freeze_filter is not None


def test_privileged_backup_defaults_to_the_baseline_v_bootstrap():
    """The default arm holds the TD target fixed against the baseline so the
    experiment isolates the critic's inputs and parameterization."""
    priv = _config.get_config("pi05_libero_online_ogpo_privileged")
    assert priv.rl.privileged_backup == "value"


class _FakeLearner:
    """Just enough of the learner to exercise the backup-gated hooks."""

    def __init__(self, backup):
        import dataclasses as dc

        self._config = dc.replace(
            _config.get_config("pi05_libero_online_ogpo_privileged"),
            rl=dc.replace(
                _config.get_config("pi05_libero_online_ogpo_privileged").rl,
                privileged_backup=backup,
            ),
        )
        self._privileged_state_dim = 16

    _uses_next_action_backup = OGPOPrivilegedLearner._uses_next_action_backup
    _extra_transition_fields = OGPOPrivilegedLearner._extra_transition_fields
    _privileged_critic_obs = OGPOPrivilegedLearner._privileged_critic_obs
    _online_batch_to_critic_batch = OGPOPrivilegedLearner._online_batch_to_critic_batch


def test_the_default_backup_writes_no_next_actions_column():
    """`next_actions` is one action chunk per transition (~640 MB at capacity
    500k). The V-bootstrap default must not allocate or write it."""
    learner = _FakeLearner("value")
    assert learner._uses_next_action_backup is False
    assert learner._extra_transition_fields(
        actions_out=np.zeros((6, 2, 4)), n_windows=3, act_h=3
    ) == {}


def test_the_opt_in_backup_writes_the_next_actions_column():
    learner = _FakeLearner("next_action_q")
    assert learner._uses_next_action_backup is True
    got = learner._extra_transition_fields(
        actions_out=np.zeros((6, 2, 4)), n_windows=3, act_h=3
    )
    assert set(got) == {"next_actions"}


def _fake_buffer_batch(with_next_actions: bool):
    obs = {
        PRIVILEGED_STATE_NAME: np.zeros((4, 16), np.float32),
        PRIVILEGED_TASK_ID_NAME: np.zeros((4, 1), np.float32),
    }
    batch = {
        "observation": obs,
        "next_observation": obs,
        "actions": np.zeros((4, 2, 4), np.float32),
        "reward": np.zeros(4, np.float32),
        "discount": np.zeros(4, np.float32),
        "mc_return": np.zeros(4, np.float32),
    }
    if with_next_actions:
        batch["next_actions"] = np.zeros((4, 2, 4), np.float32)
    return batch


def test_the_default_backup_builds_the_shared_six_element_critic_batch():
    """So the inherited AWR train steps consume it unchanged — the default
    privileged arm runs the SAME numeric critic code as the baseline."""
    learner = _FakeLearner("value")
    batch = learner._online_batch_to_critic_batch(_fake_buffer_batch(False), None)
    assert len(batch) == 6
    obs, actions, next_obs, reward, discount, mc_return = batch
    assert set(obs) == {"state", TASK_ID_KEY}
    assert np.asarray(obs["state"]).shape == (4, 16)


def test_the_opt_in_backup_builds_the_seven_element_critic_batch():
    learner = _FakeLearner("next_action_q")
    batch = learner._online_batch_to_critic_batch(_fake_buffer_batch(True), None)
    assert len(batch) == 7
    assert np.asarray(batch[3]).shape == (4, 2, 4)  # next_actions


def test_the_default_backup_delegates_the_digestion_burst_to_the_base_learner():
    """The base MC-target burst builds its jit from the AWR steps, which is
    correct for the 6-element batch and would crash on the 7-element one."""
    src = inspect.getsource(OGPOPrivilegedLearner._burst_critic_update_fn)
    assert "if not self._uses_next_action_backup:" in src
    assert "super()._burst_critic_update_fn()" in src


def test_privileged_backup_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="privileged_backup"):
        _config.OGPOPrivilegedLearnerConfig(privileged_backup="td3")


def test_privileged_config_dispatches_before_the_plain_ogpo_config():
    """OGPOPrivilegedLearnerConfig subclasses OGPOSFTLearnerConfig, so an
    isinstance chain that checks the parent first would silently run the
    baseline learner on a privileged config."""
    src = (_ROOT / "scripts" / "exp.py").read_text()
    priv = src.index("OGPOPrivilegedLearnerConfig")
    plain = src.index("_config.OGPOSFTLearnerConfig")
    assert priv < plain
    assert "OGPOPrivilegedLearner" in src


def test_sampler_uses_a_dict_sidecar_verbatim_as_the_critic_observation():
    """The privileged critic's state comes from the simulator, so it cannot be
    rebuilt inside jit-1 from the policy observation; it arrives as a dict
    sidecar in the slot the prefix rep normally occupies."""
    from src.rl.ogpo.update_actor import sample_and_advantage

    tree = ast.parse(inspect.getsource(sample_and_advantage))
    branch = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and "isinstance(critic_prefix, dict)" in ast.unparse(n.test)
    )
    assert "critic_observation = critic_prefix" in ast.unparse(branch.body)


def test_privileged_learner_supplies_that_dict_sidecar():
    sidecar_src = inspect.getsource(OGPOPrivilegedLearner._online_batch_to_sft_batch)
    assert "_privileged_critic_obs" in sidecar_src


def test_buffer_columns_are_remapped_onto_the_critic_obs_schema():
    """The buffer names them privileged_*; the critic modules read `state` and
    `task_id` (so they can reuse the single-task BroNet input assembly)."""
    buffer_obs = {
        PRIVILEGED_STATE_NAME: np.arange(6, dtype=np.float32).reshape(2, 3),
        PRIVILEGED_TASK_ID_NAME: np.array([[0.0], [1.0]], dtype=np.float32),
        "state": np.zeros((2, 8), dtype=np.float32),  # the POLICY state, ignored
        "image": {"base": np.zeros((2, 4, 4, 3), dtype=np.uint8)},
    }
    got = OGPOPrivilegedLearner._privileged_critic_obs(None, buffer_obs)
    assert set(got) == {"state", TASK_ID_KEY}
    np.testing.assert_allclose(np.asarray(got["state"]), buffer_obs[PRIVILEGED_STATE_NAME])
    np.testing.assert_allclose(
        np.asarray(got[TASK_ID_KEY]), buffer_obs[PRIVILEGED_TASK_ID_NAME]
    )


def test_privileged_critic_drops_images_unconditionally():
    """The base gate keeps images when store_prefix_rep is off (for the prefix
    recompute); this critic never performs one, so the 1024-row critic sample
    must not carry 224x224 tensors onto the GPU."""
    keys = OGPOPrivilegedLearner._critic_drop_obs_keys(None)
    assert set(keys) == {"image", "image_mask"}


# --------------------------------------------------------------------------------------
# 6. The critic train steps, end to end on CPU (no pi05 weights needed)
# --------------------------------------------------------------------------------------


def _tiny_privileged_config(**rl_overrides):
    """The registered privileged config shrunk to something CPU-sized."""
    import dataclasses as dc

    cfg = _config.get_config("pi05_libero_online_ogpo_privileged")
    critic = dc.replace(
        cfg.rl.critic,
        bronet_hidden_dim=16,
        bronet_depth=1,
        num_qs=2,
        num_vs=2,
        batch_size=8,
        use_bronet=True,
        td_weight_schedule=_config.StepSchedule(
            init_value=1.0, end_value=1.0, switch_step=1
        ),
    )
    return dc.replace(cfg, rl=dc.replace(cfg.rl, critic=critic, **rl_overrides))


_CRITIC_STATE_CACHE: dict = {}


def _build_privileged_critic_states(cfg, *, num_tasks, state_dim, act_h, act_dim):
    """Build (q_state, v_state) once and reuse them.

    The train steps are functional — they return new states and never mutate
    their inputs — so sharing one initialization across tests is safe, and it
    keeps the mesh/jit churn (and its thread usage) to a single cycle.
    """
    key = (num_tasks, state_dim, act_h, act_dim, cfg.rl.critic.bronet_hidden_dim,
           cfg.rl.critic.bronet_depth, cfg.rl.critic.num_qs, cfg.rl.critic.num_vs)
    if key in _CRITIC_STATE_CACHE:
        return _CRITIC_STATE_CACHE[key]
    states = _init_privileged_critic_states(
        cfg, num_tasks=num_tasks, state_dim=state_dim, act_h=act_h, act_dim=act_dim
    )
    _CRITIC_STATE_CACHE[key] = states
    return states


def _init_privileged_critic_states(cfg, *, num_tasks, state_dim, act_h, act_dim):
    import openpi.training.sharding as sharding
    from src.rl.advantage_weighted_sft.update_critic import (
        init_state_action_critic_train_state,
        init_state_value_train_state,
    )

    mesh = sharding.make_mesh(1)
    tx = per_task_optimizer(cfg.rl.critic.optimizer, cfg.rl.critic.lr_schedule, num_tasks)
    q_def, v_def = make_task_ensemble_defs(
        hidden_dim=cfg.rl.critic.bronet_hidden_dim,
        depth=cfg.rl.critic.bronet_depth,
        num_qs=cfg.rl.critic.num_qs,
        num_vs=cfg.rl.critic.num_vs,
        num_tasks=num_tasks,
        num_bins=cfg.rl.critic.num_value_bins,
    )
    dummy_obs = {
        "state": jnp.zeros((1, state_dim), jnp.float32),
        TASK_ID_KEY: jnp.zeros((1, 1), jnp.float32),
    }
    dummy_act = jnp.zeros((1, act_h, act_dim), jnp.float32)
    q_state, _ = init_state_action_critic_train_state(
        cfg, jax.random.key(0), mesh, critic_def=q_def,
        dummy_obs=dummy_obs, dummy_act=dummy_act, tx=tx,
    )
    v_state, _ = init_state_value_train_state(
        cfg, jax.random.key(1), mesh, critic_def=v_def, dummy_obs=dummy_obs, tx=tx
    )
    return q_state, v_state


def _privileged_batch(batch=8, state_dim=12, act_h=2, act_dim=4, num_tasks=3, seed=0):
    rng = np.random.RandomState(seed)
    ids = (np.arange(batch) % num_tasks).astype(np.float32).reshape(batch, 1)
    obs = {
        "state": jnp.asarray(rng.randn(batch, state_dim), jnp.float32),
        TASK_ID_KEY: jnp.asarray(ids),
    }
    next_obs = {
        "state": jnp.asarray(rng.randn(batch, state_dim), jnp.float32),
        TASK_ID_KEY: jnp.asarray(ids),
    }
    return (
        obs,
        jnp.asarray(rng.randn(batch, act_h, act_dim), jnp.float32),
        next_obs,
        jnp.asarray(rng.randn(batch, act_h, act_dim), jnp.float32),
        jnp.asarray(-np.ones(batch), jnp.float32),          # reward
        jnp.asarray(np.full(batch, 0.99), jnp.float32),     # discount
        jnp.asarray(rng.randn(batch) * 10.0, jnp.float32),  # mc_return
    )


@pytest.mark.parametrize("backup", ["next_action_q", "value"])
def test_privileged_q_step_runs_and_moves_the_parameters(backup):
    from src.rl.ogpo.privileged.update_critic import train_q_step

    cfg = _tiny_privileged_config(privileged_backup=backup)
    q_state, v_state = _build_privileged_critic_states(
        cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
    )
    batch = _privileged_batch()
    new_q, info = train_q_step(cfg, jax.random.key(0), q_state, v_state, batch)
    assert new_q.step == q_state.step + 1
    for key in ("loss", "td_loss", "mc_loss", "mc_corr", "bootstrap_mean", "grad_norm"):
        assert key in info, key
    assert np.isfinite(float(info["loss"]))
    assert float(info["grad_norm"]) > 0.0


def test_the_two_backups_produce_different_q_targets():
    """`next_action_q` is the whole point of the privileged critic: if it
    resolved to the same target as the V bootstrap, the experiment would be
    measuring only the state swap."""
    from src.rl.ogpo.privileged.update_critic import train_q_step

    batch = _privileged_batch()
    losses = {}
    for backup in ("next_action_q", "value"):
        cfg = _tiny_privileged_config(privileged_backup=backup)
        q_state, v_state = _build_privileged_critic_states(
            cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
        )
        _, info = train_q_step(cfg, jax.random.key(0), q_state, v_state, batch)
        losses[backup] = float(info["bootstrap_mean"])
    assert losses["next_action_q"] != losses["value"]


def test_privileged_value_step_runs_on_the_seven_element_batch():
    from src.rl.ogpo.privileged.update_critic import train_value_step

    cfg = _tiny_privileged_config(privileged_backup="next_action_q")
    q_state, v_state = _build_privileged_critic_states(
        cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
    )
    new_v, info = train_value_step(
        cfg, jax.random.key(0), v_state, q_state, _privileged_batch()
    )
    assert new_v.step == v_state.step + 1
    assert np.isfinite(float(info["loss"]))


def test_a_terminal_transition_ignores_the_bootstrap_entirely():
    """discount == 0 on true terminations, so a garbage next-action bootstrap
    (the repeated-tail fallback at the end of an episode) cannot leak in."""
    from src.rl.ogpo.privileged.update_critic import train_q_step

    cfg = _tiny_privileged_config(privileged_backup="next_action_q")
    q_state, v_state = _build_privileged_critic_states(
        cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
    )
    base = list(_privileged_batch())
    base[5] = jnp.zeros_like(base[5])  # discount = 0 everywhere
    a = train_q_step(cfg, jax.random.key(0), q_state, v_state, tuple(base))[1]["loss"]
    perturbed = list(base)
    perturbed[3] = perturbed[3] + 100.0  # wildly different next actions
    b = train_q_step(cfg, jax.random.key(0), q_state, v_state, tuple(perturbed))[1]["loss"]
    np.testing.assert_allclose(float(a), float(b), rtol=1e-6)


def test_pure_td_loss_tracks_the_reward_plus_discounted_bootstrap():
    """Sanity-check the target algebra rather than just that it runs: with
    td_weight pinned to 1 and a Gaussian (num_bins == 1) critic, the loss is the
    mean squared error against r + gamma * bootstrap."""
    from src.rl.advantage_weighted_sft.update_critic import create_critic
    from src.rl.ogpo.privileged.update_critic import train_q_step
    from src.rl.advantage_weighted_sft.update_critic import (
        flatten_action_horizon,
        summarize_critic_values,
    )

    # Pinned to the opt-in backup: this checks the Q(s', a') target algebra,
    # which the "value" default does not use.
    cfg = _tiny_privileged_config(privileged_backup="next_action_q")
    assert cfg.rl.critic.num_value_bins == 1
    q_state, v_state = _build_privileged_critic_states(
        cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
    )
    obs, act, next_obs, next_act, reward, discount, mc_return = _privileged_batch()
    _, info = train_q_step(
        cfg, jax.random.key(0), q_state, v_state,
        (obs, act, next_obs, next_act, reward, discount, mc_return),
    )

    live = nnx.merge(q_state.model_def, q_state.params)
    live.eval()
    target = create_critic(q_state, cfg)
    target.eval()
    boot = summarize_critic_values(
        target(next_obs, flatten_action_horizon(next_act)),
        cfg,
        critic_reduction=cfg.rl.critic.reduction,
    )
    td_targets = reward + discount * boot
    q = live(obs, flatten_action_horizon(act))
    expected = float(jnp.mean(jnp.square(q - td_targets)))
    np.testing.assert_allclose(float(info["td_loss"]), expected, rtol=1e-5)
    np.testing.assert_allclose(float(info["bootstrap_mean"]), float(jnp.mean(boot)), rtol=1e-5)


# --------------------------------------------------------------------------------------
# 7. Per-task gradient clipping (docs/changes/2026-08-21-per-task-critics, D5)
# --------------------------------------------------------------------------------------


def _two_task_grad_tree(scale_task0: float, scale_task1: float):
    """A parameter-shaped tree with the per-task path layout the clip reads."""
    return {
        "nets": {
            0: {"w": jnp.asarray([3.0, 4.0]) * scale_task0},   # norm 5 * scale
            1: {"w": jnp.asarray([6.0, 8.0]) * scale_task1},   # norm 10 * scale
        }
    }


def test_per_task_clip_scales_each_task_by_its_own_norm():
    tx = clip_by_global_norm_per_task(max_norm=1.0, num_tasks=2)
    grads = _two_task_grad_tree(1.0, 1.0)
    out, _ = tx.update(grads, tx.init(grads))
    np.testing.assert_allclose(np.asarray(out["nets"][0]["w"]), [3 / 5, 4 / 5], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out["nets"][1]["w"]), [6 / 10, 8 / 10], rtol=1e-6)


def test_per_task_clip_leaves_one_task_untouched_when_another_explodes():
    """The decision this implements: a global clip would shrink task 0's update
    because task 1 blew up. Here task 0's output must not move at all."""
    tx = clip_by_global_norm_per_task(max_norm=100.0, num_tasks=2)
    calm = _two_task_grad_tree(1.0, 1.0)
    exploded = _two_task_grad_tree(1.0, 1e6)
    calm_out, _ = tx.update(calm, tx.init(calm))
    exploded_out, _ = tx.update(exploded, tx.init(exploded))
    np.testing.assert_allclose(
        np.asarray(calm_out["nets"][0]["w"]), np.asarray(exploded_out["nets"][0]["w"])
    )
    # Sanity: a global clip WOULD have moved it.
    global_out, _ = optax.clip_by_global_norm(100.0).update(
        exploded, optax.clip_by_global_norm(100.0).init(exploded)
    )
    assert not np.allclose(
        np.asarray(global_out["nets"][0]["w"]), np.asarray(calm_out["nets"][0]["w"])
    )


def test_per_task_clip_is_a_no_op_below_the_threshold():
    tx = clip_by_global_norm_per_task(max_norm=100.0, num_tasks=2)
    grads = _two_task_grad_tree(1.0, 1.0)
    out, _ = tx.update(grads, tx.init(grads))
    for t in (0, 1):
        np.testing.assert_allclose(
            np.asarray(out["nets"][t]["w"]), np.asarray(grads["nets"][t]["w"])
        )


def test_per_task_clip_keeps_an_absent_task_at_exactly_zero():
    """A task with no samples in the batch has an all-zero gradient; a naive
    g/||g|| would hand it NaN and destroy its parameters on the next step."""
    tx = clip_by_global_norm_per_task(max_norm=1.0, num_tasks=2)
    grads = _two_task_grad_tree(0.0, 1.0)
    out, _ = tx.update(grads, tx.init(grads))
    np.testing.assert_array_equal(np.asarray(out["nets"][0]["w"]), [0.0, 0.0])


def test_per_task_optimizer_mirrors_adamw_without_the_global_clip():
    cfg = _tiny_privileged_config()
    tx = per_task_optimizer(cfg.rl.critic.optimizer, cfg.rl.critic.lr_schedule, 2)
    grads = _two_task_grad_tree(1.0, 1e6)
    state = tx.init(grads)
    updates, _ = tx.update(grads, state, grads)
    # AdamW normalizes per parameter, so both tasks take an ~lr-sized step even
    # though their raw gradients differ by six orders of magnitude — which only
    # holds because the clip did not couple them.
    lr = cfg.rl.critic.lr_schedule.value
    for t in (0, 1):
        step = np.abs(np.asarray(updates["nets"][t]["w"]))
        np.testing.assert_allclose(step, lr, rtol=1e-3)


def test_per_task_clip_rejects_a_tree_that_is_not_a_task_ensemble():
    tx = clip_by_global_norm_per_task(max_norm=1.0, num_tasks=2)
    bad = {"proj": {"kernel": jnp.ones((2,))}}
    with pytest.raises(ValueError, match="per-task critic parameter path"):
        tx.update(bad, tx.init(bad))


def test_the_critic_train_states_use_the_per_task_optimizer():
    cfg = _tiny_privileged_config(privileged_backup="next_action_q")
    q_state, v_state = _build_privileged_critic_states(
        cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
    )
    # optax.chain's state is a tuple; the per-task clip contributes an EmptyState
    # where openpi's AdamW would contribute clip_by_global_norm's (also empty),
    # so identify it by the transformation actually wired in.
    for state in (q_state, v_state):
        assert isinstance(state.tx, optax.GradientTransformation)
    # A gradient concentrated on one task must leave the other tasks' params put.
    from src.rl.ogpo.privileged.update_critic import train_q_step

    batch = list(_privileged_batch(num_tasks=3))
    ids = jnp.zeros((batch[0]["state"].shape[0], 1), jnp.float32)  # all task 0
    batch[0] = {**batch[0], TASK_ID_KEY: ids}
    batch[2] = {**batch[2], TASK_ID_KEY: ids}
    new_q, _ = train_q_step(cfg, jax.random.key(0), q_state, v_state, tuple(batch))
    before = dict(nnx.filter_state(q_state.params, nnx.Param).flat_state())
    after = dict(nnx.filter_state(new_q.params, nnx.Param).flat_state())
    moved = {int(path[1]) for path in before
             if not np.allclose(np.asarray(before[path].value), np.asarray(after[path].value))}
    assert moved == {0}, moved


# --------------------------------------------------------------------------------------
# 8. The DEFAULT path: the shared AWR critic steps over the privileged observation
# --------------------------------------------------------------------------------------


def _default_privileged_batch(batch=8, state_dim=12, act_h=2, act_dim=4, num_tasks=3, seed=0):
    """The shared 6-element CriticBatch the default arm builds."""
    obs, act, next_obs, _next_act, reward, discount, mc = _privileged_batch(
        batch=batch, state_dim=state_dim, act_h=act_h, act_dim=act_dim,
        num_tasks=num_tasks, seed=seed,
    )
    return (obs, act, next_obs, reward, discount, mc)


def test_default_arm_runs_the_shared_awr_critic_steps_on_the_privileged_obs():
    """The default privileged arm swaps NO numeric critic code — it feeds the
    per-task critic to the same steps the baseline arm uses."""
    from src.rl.advantage_weighted_sft.update_critic import train_q_step, train_value_step

    cfg = _tiny_privileged_config()
    assert cfg.rl.privileged_backup == "value"
    q_state, v_state = _build_privileged_critic_states(
        cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
    )
    batch = _default_privileged_batch()
    new_q, q_info = train_q_step(cfg, jax.random.key(0), q_state, v_state, batch)
    new_v, v_info = train_value_step(cfg, jax.random.key(0), v_state, q_state, batch)
    assert new_q.step == q_state.step + 1 and new_v.step == v_state.step + 1
    # Same key schema the baseline arm logs, so the two series are comparable.
    for key in ("loss", "td_loss", "mc_loss", "td_weight", "mc_corr", "value_mean"):
        assert key in q_info, key
    assert np.isfinite(float(q_info["loss"])) and np.isfinite(float(v_info["loss"]))


def test_default_arm_still_routes_gradients_to_one_task_only():
    """Per-task isolation is a property of the critic module, not of the train
    step — it must hold on the default (shared-step) path too."""
    from src.rl.advantage_weighted_sft.update_critic import train_q_step

    cfg = _tiny_privileged_config()
    q_state, v_state = _build_privileged_critic_states(
        cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
    )
    batch = list(_default_privileged_batch())
    ids = jnp.full((batch[0]["state"].shape[0], 1), 2.0, jnp.float32)  # all task 2
    batch[0] = {**batch[0], TASK_ID_KEY: ids}
    batch[2] = {**batch[2], TASK_ID_KEY: ids}
    new_q, _ = train_q_step(cfg, jax.random.key(0), q_state, v_state, tuple(batch))
    before = dict(nnx.filter_state(q_state.params, nnx.Param).flat_state())
    after = dict(nnx.filter_state(new_q.params, nnx.Param).flat_state())
    moved = {int(path[1]) for path in before
             if not np.allclose(np.asarray(before[path].value), np.asarray(after[path].value))}
    assert moved == {2}, moved


def test_default_arm_q_target_is_the_v_bootstrap():
    """Pin the target algebra of the arm that will actually run: pure TD, a
    Gaussian critic, so the loss is the MSE against r + gamma * V(s')."""
    from src.rl.advantage_weighted_sft.update_critic import (
        create_critic,
        flatten_action_horizon,
        summarize_critic_values,
        train_q_step,
    )

    cfg = _tiny_privileged_config()
    q_state, v_state = _build_privileged_critic_states(
        cfg, num_tasks=3, state_dim=12, act_h=2, act_dim=4
    )
    obs, act, next_obs, reward, discount, mc_return = _default_privileged_batch()
    _, info = train_q_step(
        cfg, jax.random.key(0), q_state, v_state, (obs, act, next_obs, reward, discount, mc_return)
    )
    value_model = create_critic(v_state, cfg)
    value_model.eval()
    boot = summarize_critic_values(
        value_model(next_obs), cfg, critic_reduction=cfg.rl.critic.reduction
    )
    live = nnx.merge(q_state.model_def, q_state.params)
    live.eval()
    expected = float(
        jnp.mean(jnp.square(live(obs, flatten_action_horizon(act)) - (reward + discount * boot)))
    )
    np.testing.assert_allclose(float(info["td_loss"]), expected, rtol=1e-5)
