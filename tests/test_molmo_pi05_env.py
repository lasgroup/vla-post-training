"""
Tests for the MolmoSpaces pi05 env family (docs/changes/2026-10-14-molmo-pi05/), CPU, no simulator:

  - `molmo_pi05-<id>` parsing and `get_env_maker` dispatch; make_env's guards (vector only, post_success_steps=0,
    MLSPACES_* / NLTK_DATA required) and laziness (gymnasium builds slot 0 in the trainer: no molmo_spaces import).
  - MolmoPi05Env over a fake MolmoSpaces task: obs reduction (exo / wrist / 7 joints + gripper / 0.824033),
    action adapter (gripper > 0.5 -> 255), vla-post-training's time-to-success reward (-1, 0 on success),
    terminate on success, 450-step truncation, terminal return / length, prompt fixed across resets.
  - main_pi05: robot / family / vector_envs guards; molmo prompts come from the workers (`vector_prompts`).

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_molmo_pi05_env.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace

import numpy as np
import pytest

import envs.molmo_pi05_utils as MU
from envs.molmo_pi05_utils import MolmoPi05Env, _parse_env_name, is_molmo_pi05_env

ENV_VARS = {'MLSPACES_ASSETS_DIR': '/x/assets', 'MLSPACES_CACHE_DIR': '/x/cache', 'NLTK_DATA': '/x/nltk'}


@pytest.fixture
def env_vars(monkeypatch):
    for k, v in ENV_VARS.items():
        monkeypatch.setenv(k, v)


# --- naming / dispatch / make_env -----------------------------------------------------------------------------------

def test_parse_env_name():
    assert _parse_env_name('molmo_pi05-164') == 164
    assert is_molmo_pi05_env('molmo_pi05-0')
    for bad in ('molmo_164', 'molmo_pi05-', 'molmo_pi05-x1', 'molmo_pi05--3', 'libero_pi05-libero_90-1', None):
        assert not is_molmo_pi05_env(bad)
    with pytest.raises(ValueError, match="molmo_pi05-<episode_id>"):
        _parse_env_name('molmo_164')


def test_get_env_maker_dispatch():
    from ogpo.utils.main_helper import get_env_maker
    assert get_env_maker('molmo_pi05-164') is MU.make_env


def test_make_env_guards(monkeypatch, env_vars):
    with pytest.raises(ValueError, match='vector_envs=true'):
        MU.make_env('molmo_pi05-164')
    with pytest.raises(ValueError, match='post_success_steps=1'):
        MU.make_env('molmo_pi05-164', post_success_steps=1, vector=True)
    monkeypatch.delenv('MLSPACES_CACHE_DIR')
    with pytest.raises(ValueError, match="MLSPACES_CACHE_DIR"):
        MU.make_env('molmo_pi05-164', vector=True)


def test_make_env_is_lazy(env_vars):
    """gymnasium's AsyncVectorEnv builds slot 0 in the trainer to read the spaces: nothing may load there."""
    env = MU.make_env('molmo_pi05-164', seed=3, vector=True)
    assert 'molmo_spaces' not in sys.modules
    inner = env.unwrapped
    assert isinstance(inner, MolmoPi05Env) and inner._sampler is None and inner.episode_id == 164
    sp = env.observation_space
    assert sp['image'].shape == sp['wrist_image'].shape == (224, 224, 3)  # VectorSlotWrapper resizes in the worker
    assert sp['state'].shape == (8,) and env.action_space.shape == (8,)
    env.close()


# --- MolmoPi05Env over a fake task ----------------------------------------------------------------------------------

def _raw_obs(k):
    return {'qpos': {'arm': np.arange(7) * 0.1 + k, 'gripper': np.array([0.412, 0.412]), 'base': np.zeros(0)},
            'exo_camera_1': np.full((352, 624, 3), 7, np.uint8), 'wrist_camera': np.full((352, 624, 3), 9, np.uint8),
            'env_states': {'big': np.zeros(31)}}


class _FakeTask:
    def __init__(self, succeed_at=None):
        self.env = SimpleNamespace(n_batch=1)
        self._sensor_suite = SimpleNamespace(sensors={'object_image_points': 1, 'exo_camera_1': 2})
        self.succeed_at, self.k, self.actions = succeed_at, 0, []

    def register_policy(self, policy):
        self.policy = policy

    def reset(self):
        self.k = 0
        return [_raw_obs(0)], [{'success': False}]

    def step(self, action):
        self.actions.append(action)
        self.k += 1
        succ = self.k == self.succeed_at
        return [_raw_obs(self.k)], np.array([0.37]), np.array([succ]), np.array([False]), [{'success': succ}]


def _env(monkeypatch, succeed_at=None, prompts=('Pick Up The Box.',)):
    task = _FakeTask(succeed_at)
    prompt_iter = iter(list(prompts) * 100)

    def fake_load(self):
        if self._sampler is None:
            self._sampler = SimpleNamespace(sample_task=lambda **kw: task, close=lambda: None)
            self._episode = SimpleNamespace(house_index=244)
            self._prompt_sampler = SimpleNamespace(next=lambda: None, get_prompt=lambda t: next(prompt_iter))

    monkeypatch.setattr(MolmoPi05Env, '_load', fake_load)
    return MolmoPi05Env(164), task


def test_reset_obs_and_prompt(monkeypatch):
    env, task = _env(monkeypatch)
    obs, info = env.reset(seed=0)
    assert info == {'task_description': 'pick up the box.'} and env.prompt == 'pick up the box.'
    assert set(obs) == {'image', 'wrist_image', 'state', 'full_state'}
    assert obs['image'].shape == (352, 624, 3) and obs['image'].dtype == np.uint8 and obs['image'][0, 0, 0] == 7
    assert obs['wrist_image'][0, 0, 0] == 9
    np.testing.assert_allclose(obs['state'][:7], np.arange(7) * 0.1, atol=1e-6)
    np.testing.assert_allclose(obs['state'][7], 0.412 / 0.824033, atol=1e-6)
    assert obs['state'].dtype == np.float32 and np.array_equal(obs['state'], obs['full_state'])
    assert 'object_image_points' not in task._sensor_suite.sensors and 'exo_camera_1' in task._sensor_suite.sensors
    assert env.get_prompt() == 'pick up the box.'


def test_action_adapter(monkeypatch):
    env, task = _env(monkeypatch)
    env.reset(seed=0)
    env.step(np.array([1, 2, 3, 4, 5, 6, 7, 0.6]))
    env.step(np.array([1, 2, 3, 4, 5, 6, 7, 0.4]))
    a0, a1 = task.actions
    np.testing.assert_array_equal(a0['arm'], np.arange(1, 8, dtype=np.float32))
    assert a0['arm'].dtype == np.float32 and a0['gripper'].tolist() == [255.0] and a1['gripper'].tolist() == [0.0]


def test_time_to_success_reward_and_termination(monkeypatch):
    env, _ = _env(monkeypatch, succeed_at=4)
    env.reset(seed=0)
    rewards = []
    for _ in range(4):
        obs, r, term, trunc, info = env.step(np.zeros(8))
        rewards.append(r)
    assert rewards == [-1.0, -1.0, -1.0, 0.0]  # no success bonus
    assert term and not trunc and info['success'] == 1
    assert info['length'] == 4 and info['return'] == -3.0 and env.t == 4


def test_truncation_at_450(monkeypatch):
    env, _ = _env(monkeypatch)
    env.reset(seed=0)
    for t in range(450):
        obs, r, term, trunc, info = env.step(np.zeros(8))
        if t < 449:
            assert not (term or trunc) and 'length' not in info and info['success'] == 0
    assert trunc and not term and info['length'] == 450 and info['return'] == -450.0


def test_reset_restarts_counters_and_rejects_prompt_change(monkeypatch):
    env, _ = _env(monkeypatch, prompts=('pick up the box.', 'pick up the box.', 'pick up the mug.'))
    env.reset(seed=0)
    env.step(np.zeros(8))
    env.reset(seed=1)
    assert env.t == 0 and env.episode_return == 0.0
    with pytest.raises(RuntimeError, match='prompt changed'):
        env.reset(seed=2)


# --- runner helper / main_pi05 --------------------------------------------------------------------------------------

def test_vector_prompts_reads_workers():
    from ogpo.runners.pi05_online_runner import VectorTasks, vector_prompts
    calls = []
    vec = SimpleNamespace(call_async=lambda name: calls.append(name),
                          call_wait=lambda timeout: ('p a', 'p a', 'p b', 'p b'))
    assert vector_prompts(VectorTasks(vec, ['a', 'b'], [0, 1], 2, 4)) == {'a': 'p a', 'b': 'p b'}
    assert calls == ['get_prompt']
    vec.call_wait = lambda timeout: ('p a', 'p x', 'p b', 'p b')
    with pytest.raises(RuntimeError, match='disagree'):
        vector_prompts(VectorTasks(vec, ['a', 'b'], [0, 1], 2, 4))


class _Built(Exception):
    pass


def _main(monkeypatch, *extra):
    import ogpo.main_pi05 as M
    monkeypatch.setattr(sys, 'argv', ['main_pi05.py', '--algo=ogpo', '--experiment.log=false',
                                      '--agent.pi05_action_horizon=15', '--training.collection_mode=rounds', *extra])
    built = {}

    def fail(*a, **k):
        raise AssertionError('no libero env / prompt may be built for molmo')

    def make_vec(names, num_envs, **kw):
        built.setdefault('vec', []).append((list(names), kw))
        return SimpleNamespace(close=lambda terminate=False: None)

    def create(cls, seed, prompt, config):
        built['prompt'] = prompt
        raise _Built()

    monkeypatch.setattr(M, 'make_env', fail)
    monkeypatch.setattr(M, 'get_prompt', fail)
    monkeypatch.setattr(M, 'make_vectorized_env', make_vec)
    monkeypatch.setattr(M, 'vector_prompts', lambda vt: {n: f'prompt:{n}' for n in vt.names})
    monkeypatch.setattr(M.OGPOPi05Agent, 'create', classmethod(create))
    return M, built


@pytest.mark.parametrize('extra,match', [
    (['--env.name=molmo_pi05-164', '--env.vector_envs=true'], "pi05_robot='libero'.*need pi05_robot='droid'"),
    (['--env.name=libero_pi05-libero_90-1', '--agent.pi05_robot=droid'], "need pi05_robot='libero'"),
    (['--env.name=molmo_pi05-164', '--agent.pi05_robot=droid'], 'vector_envs=true'),
    (['--env.name=molmo_pi05-164,libero_pi05-libero_90-1', '--agent.pi05_robot=droid', '--env.vector_envs=true'],
     'mix libero_pi05 and molmo_pi05'),
])
def test_main_pi05_molmo_guards(monkeypatch, extra, match):
    M, built = _main(monkeypatch, *extra)
    with pytest.raises(ValueError, match=match):
        M.main()
    assert 'vec' not in built


def test_main_pi05_molmo_prompts_from_workers(monkeypatch):
    M, built = _main(monkeypatch, '--env.name=molmo_pi05-164,molmo_pi05-86', '--env.eval_name=molmo_pi05-164,molmo_pi05-70',
                     '--agent.pi05_robot=droid', '--env.vector_envs=true', '--env.post_success_steps=0')
    with pytest.raises(_Built):
        M.main()
    assert [names for names, _ in built['vec']] == [['molmo_pi05-164', 'molmo_pi05-86'], ['molmo_pi05-164', 'molmo_pi05-70']]
    assert all(kw['env_kwargs'] == {'vector': True} and kw['context'] == 'spawn' for _, kw in built['vec'])
    # prompt table: train tasks first, then eval-only tasks
    assert built['prompt'] == ['prompt:molmo_pi05-164', 'prompt:molmo_pi05-86', 'prompt:molmo_pi05-70']


@pytest.mark.parametrize('visible,pinned,expect', [
    ('2,3', None, 2),      # no pin: first visible GPU (every slot on one GPU)
    ('2,3', '3', 3),       # a pin that is one of the run's GPUs wins (worker_egl_devices / launcher)
    ('2,3', '5', 2),       # a pin outside CUDA_VISIBLE_DEVICES is ignored: molmo asserts it is visible
    ('', '4', 4),          # nothing visible-listed: the pin
    ('', None, 0),
    ('0,1,2,3', 'x', 0),   # a non-numeric pin is ignored
])
def test_render_device_follows_the_workers_egl_pin(monkeypatch, visible, pinned, expect):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    if pinned is None:
        monkeypatch.delenv('MUJOCO_EGL_DEVICE_ID', raising=False)
    else:
        monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', pinned)
    assert MU._render_device() == expect

