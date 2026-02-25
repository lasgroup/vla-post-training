# Taken from
# https://github.com/denisyarats/dmc2gym
# and modified to exclude duplicated code.

import copy
from typing import Dict, Optional, OrderedDict

import jax
import dm_env
import numpy as np
from dm_control import suite
from gymnasium import core, spaces


from src.envs.venv import BaseVectorEnv


def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(low=spec.minimum,
                          high=spec.maximum,
                          shape=spec.shape,
                          dtype=spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        return spaces.Box(low=-float('inf'),
                          high=float('inf'),
                          shape=spec.shape,
                          dtype=spec.dtype)
    else:
        raise NotImplementedError


class DMCEnv(core.Env):

    def __init__(self,
                 domain_name: Optional[str] = None,
                 task_name: Optional[str] = None,
                 env: Optional[dm_env.Environment] = None,
                 task_kwargs: Optional[Dict] = {},
                 environment_kwargs=None,
                 height: int = 84,
                 width: int = 84,
                 camera_id: int = 0,
                 ):
        assert 'random' in task_kwargs, 'Please specify a seed, for deterministic behaviour.'
        assert (
            env is not None
            or (domain_name is not None and task_name is not None)
        ), 'You must provide either an environment or domain and task names.'

        if env is None:
            env = suite.load(domain_name=domain_name,
                             task_name=task_name,
                             task_kwargs=task_kwargs,
                             environment_kwargs=environment_kwargs)

        self._env = env
        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        self.observation_space = dmc_spec2gym_space(
            self._env.observation_spec())

        self.seed(seed=task_kwargs['random'])
        self.render_mode = 'rgb_array'
        self.height = height
        self.width = width
        self.camera_id = camera_id

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        obs = time_step.observation

        termination = False  # we never reach a goal
        truncation = time_step.last()
        info = {"discount": time_step.discount}
        return obs, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        self.seed(seed)
        timestep = self._env.reset()
        observation = timestep.observation
        info = {}
        return observation, info

    def seed(self, seed):
        if seed is not None:
            if not isinstance(seed, np.random.RandomState):
                seed = np.random.RandomState(seed)
            self._env.task._random = seed

    def render(self):
        return self._env.physics.render(height=self.height,
                                        width=self.width,
                                        camera_id=self.camera_id)

# Pick some domain/task combos to cycle through
# tasks = [
#     ("cartpole", "swingup"),
#     ("reacher", "easy"),
#     ("cheetah", "run"),
#     ("walker", "walk"),
#     ("hopper", "stand"),
# ]

# seed = 42
# num_episodes = 3
# max_steps = 100

# for domain, task in tasks:
#     print(f"\n=== Running {domain}/{task} ===")
#     env = DMCEnv(
#         domain_name=domain,
#         task_name=task,
#         task_kwargs={"random": seed},
#     )

#     for ep in range(num_episodes):
#         obs, info = env.reset(seed=seed + ep)
#         total_reward = 0.0

#         for step in range(max_steps):
#             action = env.action_space.sample()
#             obs, reward, termination, truncation, info = env.step(action)
#             total_reward += reward

#             if termination or truncation:
#                 break

#         print(f"  Episode {ep + 1}: steps={step + 1}, reward={total_reward:.2f}")

#     env.close()