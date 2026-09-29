"""Minimal synchronous vector envs (adapted from tianshou).

Only the features used by data collection are kept: seeding, (partial) resets
with per-env kwargs, synchronous stepping and closing.
"""

import sys
import traceback
import faulthandler
import os
from multiprocessing import Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, List, Optional, Union

import cloudpickle
import gymnasium as gym
import jax
import numpy as np


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


################################################################################
#
# Workers
#
################################################################################


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent.close()
    # Dump C-level stack trace on segfault/abort to stderr
    faulthandler.enable(file=sys.stderr, all_threads=True)
    pid = os.getpid()
    print(f"[ENV WORKER pid={pid}] starting", file=sys.stderr, flush=True)
    try:
        env = env_fn_wrapper.data()
    except Exception:
        print(f"[ENV WORKER pid={pid}] env creation failed:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        p.close()
        return
    print(f"[ENV WORKER pid={pid}] env created successfully", file=sys.stderr, flush=True)
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                p.send(env.step(data))
            elif cmd == "reset":
                p.send(env.reset(**data))
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            else:
                p.close()
                raise NotImplementedError(cmd)
    except KeyboardInterrupt:
        p.close()
    except Exception:
        print(f"[ENV WORKER pid={pid}] unhandled exception:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        p.close()
    finally:
        print(f"[ENV WORKER pid={pid}] exiting", file=sys.stderr, flush=True)


class DummyEnvWorker:
    """Runs the env in the main process."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        self.is_closed = False

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)

    def recv(self):
        return self.result

    def seed(self, seed: Optional[int] = None):
        try:
            return self.env.seed(seed)
        except (AttributeError, NotImplementedError):
            self.env.reset(seed=seed)
            return [seed]

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        self.env.close()


class SubprocEnvWorker:
    """Runs the env in a subprocess, communicating through a pipe."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.parent_remote, self.child_remote = Pipe()
        args = (self.parent_remote, self.child_remote, CloudpickleWrapper(env_fn))
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        self.is_closed = False

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.parent_remote.send(["reset", kwargs])
        else:
            self.parent_remote.send(["step", action])

    def recv(self):
        return self.parent_remote.recv()

    def seed(self, seed: Optional[int] = None):
        self.parent_remote.send(["seed", seed])
        return self.parent_remote.recv()

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        try:
            self.parent_remote.send(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        self.process.terminate()


################################################################################
#
# VecEnvs
#
################################################################################


class BaseVectorEnv(object):
    """Synchronous vectorized environment over a list of env factories.

    Envs must follow the gymnasium API: ``reset`` returns ``(obs, info)`` and
    ``step`` returns ``(obs, reward, terminated, truncated, info)``. Observations,
    rewards, flags and infos are stacked along a leading env axis.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], Any],
    ) -> None:
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.env_num = len(env_fns)
        self.is_closed = False

    def _assert_is_not_closed(self) -> None:
        assert not self.is_closed, f"Methods of {self.__class__.__name__} cannot be called after close."

    def _wrap_id(self, id: Optional[Union[int, List[int], np.ndarray]] = None) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id

    @staticmethod
    def _stack_infos(infos: List[dict]) -> dict:
        return {k: np.array([info[k] for info in infos]) for k in infos[0].keys()}

    def reset(self, id: Optional[Union[int, List[int], np.ndarray]] = None, **kwargs: Any):
        """Reset the envs in ``id`` (all envs if None).

        Any list found in ``kwargs`` (at any nesting depth) is indexed by env
        position, so ``options={"task_id": [...]}`` assigns one task per env.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        for i in id:
            local_kwargs = jax.tree_util.tree_map(
                lambda v: v[i] if isinstance(v, list) else v,
                kwargs,
                is_leaf=lambda x: isinstance(x, list),
            )
            self.workers[i].send(None, **local_kwargs)
        ret_list = [self.workers[i].recv() for i in id]
        obs = jax.tree.map(lambda *xs: np.stack(xs), *[r[0] for r in ret_list])
        return obs, self._stack_infos([r[1] for r in ret_list])

    def step(self, action: np.ndarray):
        """Step all envs synchronously with a batch of actions."""
        self._assert_is_not_closed()
        assert len(action) == self.env_num
        for worker, act in zip(self.workers, action):
            worker.send(act)
        result = [worker.recv() for worker in self.workers]
        obs_list, *others, info_list = zip(*result)
        obs = jax.tree.map(lambda *xs: np.stack(xs), *obs_list)
        return (obs, *map(np.stack, others), self._stack_infos(info_list))

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> list:
        """Seed all envs; an int ``s`` expands to ``[s, s + 1, ...]``."""
        self._assert_is_not_closed()
        if seed is None:
            seed_list = [None] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def close(self) -> None:
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True


class DummyVectorEnv(BaseVectorEnv):
    """Vectorized env that steps all envs sequentially in the main process."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns, DummyEnvWorker)


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized env with one subprocess per env."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns, SubprocEnvWorker)
