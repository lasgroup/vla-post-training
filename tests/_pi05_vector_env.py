"""Numpy-only gym envs for tests/test_pi05_vector.py (not collected: leading underscore).

Importable from spawned AsyncVectorEnv workers, so it imports nothing heavy at module level.

SlotEnv        TaskEnv dynamics (tests/_pi05_stubs.py) as one vector slot: freeze-after-done (as
               envs.libero_pi05_utils.VectorSlotWrapper), info['length'] / info['return'] only on the terminal
               step and a str 'task_description' every step (as LiberoImageWrapper). Never-reset slots step
               like an un-reset sim (counted, never done), as a constructed-but-unreset LIBERO env would.
FakeLiberoEnv  256x256 LIBERO-like obs dict, for VectorSlotWrapper (episode ends after `horizon` steps).
"""
import time

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

IMG = 4


def ep_len(seed):  # = tests/_pi05_stubs.ep_len
    return 3 + seed % 4


def _reg():
    return {'built': [], 'resets': [], 'actions': [], 'frozen_steps': 0, 'unreset_steps': 0}


class SlotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task, reg=None, slow_on=None):
        self.task, self.reg = task, (reg if reg is not None else _reg())
        self.t, self.s, self._final = 0, None, None
        self.slow_on = slow_on
        img = Box(0, 255, (IMG, IMG, 3), np.uint8)
        self.observation_space = Dict({'image': img, 'wrist_image': img,
                                       'state': Box(-np.inf, np.inf, (8,), np.float32)})
        self.action_space = Box(-np.inf, np.inf, (7,), np.float32)
        self.reg['built'].append(self)

    def _obs(self):
        st = np.zeros(8, np.float32)
        s = 0 if self.s is None else self.s
        st[0], st[1], st[2] = self.task, s, self.t
        img = np.full((IMG, IMG, 3), (7 * s + self.t) % 256, np.uint8)
        return {'image': img, 'wrist_image': img.copy(), 'state': st}

    def reset(self, *, seed=None, options=None):
        assert seed is not None, "every reset passes a seed"
        assert options is None or 'reset_mask' not in options
        self.reg['resets'].append((self.task, int(seed)))
        self.s, self.t, self._final = int(seed), 0, None
        return self._obs(), {'task_description': f'task{self.task}'}

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (7,)
        if self.slow_on is not None and action[0] == self.slow_on:
            time.sleep(30)
        if self._final is not None:
            self.reg['frozen_steps'] += 1
            assert not action.any(), "a finished slot got a non-zero action"
            return self._final
        if self.s is None:  # constructed but never reset: the sim steps anyway
            self.reg['unreset_steps'] += 1
            return self._obs(), 0.0, False, False, {'success': 0, 'task_description': f'task{self.task}'}
        assert self.t < ep_len(self.s), "stepped a finished episode"
        self.reg['actions'].append((self.task, self.s, self.t, action.copy()))
        self.t += 1
        end = self.t >= ep_len(self.s)
        succ = int(self.s % 2 == 0) if end else 0
        info = {'success': succ, 'task_description': f'task{self.task}'}
        if end:
            info['length'] = self.t
            info['return'] = -float(self.t)
        out = (self._obs(), -1.0, bool(end and succ), bool(end and not succ), info)
        if end:
            self._final = out
        return out


def make_slot_env(task, slow_on=None):
    return SlotEnv(task, slow_on=slow_on)


class FakeLiberoEnv(gym.Env):
    """LiberoPi05Wrapper-shaped obs: 256x256 images, 8-D state / full_state. Counts inner steps."""
    metadata = {"render_modes": []}

    def __init__(self, horizon=3, size=256):
        img = Box(0, 255, (size, size, 3), np.uint8)
        st = Box(-np.inf, np.inf, (8,), np.float32)
        self.observation_space = Dict({'image': img, 'wrist_image': img, 'state': st, 'full_state': st})
        self.action_space = Box(-1.0, 1.0, (7,), np.float32)
        self.horizon, self.size = horizon, size
        self.inner_steps, self.t, self.last_reset = 0, 0, None

    def _obs(self):
        rng = np.random.default_rng(1000 * (self.last_reset or 0) + self.t)
        img = rng.integers(0, 256, (self.size, self.size, 3), dtype=np.uint8)
        st = np.full(8, self.t, np.float32)
        return {'image': img, 'wrist_image': img[::-1].copy(), 'state': st, 'full_state': st.copy()}

    def reset(self, *, seed=None, options=None):
        self.last_reset, self.t = seed, 0
        return self._obs(), {'seed': seed}

    def step(self, action):
        self.inner_steps += 1
        self.t += 1
        done = self.t >= self.horizon
        info = {'success': int(done), 'inner_steps': self.inner_steps}
        if done:
            info['length'] = self.t
        return self._obs(), -1.0, done, False, info


def make_wrapped_fake(horizon=3):
    from envs.libero_pi05_utils import VectorSlotWrapper
    return VectorSlotWrapper(FakeLiberoEnv(horizon))


def check_env_maker():
    """get_env_maker / make_vectorized_env checks. Run in a subprocess by test_pi05_vector.py: get_env_maker imports
    envs.env_utils (torch), and torch + a later XLA compile in the same pytest process aborts."""
    import envs.libero_pi05_utils as L
    import ogpo.utils.main_helper as MH
    from gymnasium.vector import AutoresetMode

    assert MH.get_env_maker('libero_pi05-libero_90-82') is L.make_env
    for bad in ('libero_pi05-nosuite-3', 'nope-env'):
        try:
            MH.get_env_maker(bad)
            raise AssertionError(f'{bad} resolved')
        except ValueError as e:
            assert 'Unknown environment' in str(e)
    for args, msg in (((['libero_pi05-libero_90-1'] * 3, 4), 'env names for num_envs'),
                      ((['libero_pi05-libero_90-1', 'nope-env'], 2), 'Unknown environment')):
        try:
            MH.make_vectorized_env(*args)
            raise AssertionError(f'{args} accepted')
        except ValueError as e:
            assert msg in str(e), e

    calls, made = {}, []

    class FakeAsync:
        def __init__(self, fns, autoreset_mode, context):
            calls.update(fns=fns, autoreset_mode=autoreset_mode, context=context)

    def maker(name, seed, post_success_steps, **kw):
        made.append((name, seed, post_success_steps, kw))
    MH.AsyncVectorEnv, MH.get_env_maker = FakeAsync, lambda name: maker
    names = ['a', 'a', 'b', 'b']
    MH.make_vectorized_env(names, 4, seed=42, post_success_steps=3, context='spawn', env_kwargs={'vector': True})
    assert calls['context'] == 'spawn' and calls['autoreset_mode'] == AutoresetMode.DISABLED
    for f in calls['fns']:
        f()
    assert made == [(n, 42 + i, 3, {'vector': True}) for i, n in enumerate(names)], made
    made.clear()
    MH.make_vectorized_env('a', 2, seed=0)  # old single-name call shape (ogpo/main.py)
    for f in calls['fns']:
        f()
    assert made == [('a', 0, 0, {}), ('a', 1, 0, {})] and calls['context'] is None, made
    print('CHECK_ENV_MAKER_OK')


class EglProbeEnv(SlotEnv):
    """SlotEnv whose obs also carries the process's EGL device env vars: state[3] = MUJOCO_EGL_DEVICE_ID,
    state[4] = EGL_DEVICE_ID (-1 if unset), read at obs time (so inside the worker for a spawned slot)."""

    def _obs(self):
        import os
        o = super()._obs()
        o['state'][3] = float(os.environ.get('MUJOCO_EGL_DEVICE_ID', -1))
        o['state'][4] = float(os.environ.get('EGL_DEVICE_ID', -1))
        return o


def make_egl_probe_env(name, seed=0, post_success_steps=0, **kw):
    """make_env-shaped factory for make_vectorized_env (via a patched get_env_maker); picklable by reference."""
    return EglProbeEnv(0)
