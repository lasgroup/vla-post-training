import cloudpickle
import ctypes
import gymnasium as gym
import numpy as np
import warnings
import time

from abc import ABC, abstractmethod
from collections import OrderedDict
from multiprocessing import Array, Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, List, Optional, Tuple, Union
import jax

gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)
_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


def deprecation(msg: str) -> None:
    """Deprecation warning wrapper."""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


GYM_RESERVED_KEYS = [
    "metadata",
    "reward_range",
    "spec",
    "action_space",
    "observation_space",
]


################################################################################
#
# Workers
#
################################################################################


class EnvWorker(ABC):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Union[
            gym_old_venv_step_type,
            gym_new_venv_step_type,
            Tuple[np.ndarray, dict],
            np.ndarray,
        ]
        # self.action_space = self.get_env_attr("action_space")  # noqa: B009
        self.is_reset = False

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        pass

    def send(self, action: Optional[np.ndarray]) -> None:
        """Send action signal to low-level worker.

        When action is None, it indicates sending "reset" signal; otherwise
        it indicates "step" signal. The paired return value from "recv"
        function is determined by such kind of different signal.
        """
        if hasattr(self, "send_action"):
            deprecation(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if action is None:
                self.is_reset = True
                self.result = self.reset()
            else:
                self.is_reset = False
                self.send_action(action)

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        """Receive result from low-level worker.

        If the last "send" function sends a NULL action, it only returns a
        single observation; otherwise it returns a tuple of (obs, rew, done,
        info) or (obs, rew, terminated, truncated, info), based on whether
        the environment is using the old step API or the new one.
        """
        if hasattr(self, "get_result"):
            deprecation(
                "get_result will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if not self.is_reset:
                self.result = self.get_result()
        return self.result

    @abstractmethod
    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        pass

    def step(
        self, action: np.ndarray
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Perform one timestep of the environment's dynamic.

        "send" and "recv" are coupled in sync simulation, so users only call
        "step" function. But they can be called separately in async
        simulation, i.e. someone calls "send" first, and calls "recv" later.
        """
        self.send(action)
        return self.recv()  # type: ignore

    @staticmethod
    def wait(
        workers: List["EnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        # return self.action_space.seed(seed)  # issue 299
        pass

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """Render the environment."""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()


class ShArray:
    """Wrapper of multiprocessing Array."""

    def __init__(self, dtype: np.generic, shape: Tuple[int]) -> None:
        self.arr = Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))  # type: ignore
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(
            self.shape
        )  # type: ignore
        np.copyto(dst_np, ndarray)

    def get(self) -> np.ndarray:
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)  # type: ignore


def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        return ShArray(space.dtype, space.shape)  # type: ignore


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                p.send(obs)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env.unwrapped, key, value)

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return self.env.reset(**kwargs)

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)  # type: ignore

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        try:
            return self.env.seed(seed)  # type: ignore
        except (AttributeError, NotImplementedError):
            self.env.reset(seed=seed)
            return [seed]  # type: ignore

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()

    def check_success(self):
        return self.env.check_success()

    def get_segmentation_of_interest(self, segmentation_image):
        return self.env.get_segmentation_of_interest(segmentation_image)

    def get_sim_state(self):
        return self.env.get_sim_state()

    def set_init_state(self, init_state):
        return self.env.set_init_state(init_state)


class SubprocEnvWorker(EnvWorker):
    """Subprocess worker used in SubprocVectorEnv and ShmemVectorEnv."""

    def __init__(
        self, env_fn: Callable[[], gym.Env], share_memory: bool = False
    ) -> None:
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_remote.send(["setattr", {"key": key, "value": value}])

    def _decode_obs(self) -> Union[dict, tuple, np.ndarray]:
        def decode_obs(
            buffer: Optional[Union[dict, tuple, ShArray]]
        ) -> Union[dict, tuple, np.ndarray]:
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError

        return decode_obs(self.buffer)

    @staticmethod
    def wait(  # type: ignore
        workers: List["SubprocEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None,
    ) -> List["SubprocEnvWorker"]:
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: List[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]
        return [workers[conns.index(con)] for con in ready_conns]

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            if "seed" in kwargs:
                super().seed(kwargs["seed"])
            self.parent_remote.send(["reset", kwargs])
        else:
            self.parent_remote.send(["step", action])

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            if len(result) == 2:
                obs, info = result
                if self.share_memory:
                    obs = self._decode_obs()
                return obs, info
            obs = result[0]
            if self.share_memory:
                obs = self._decode_obs()
            return (obs, *result[1:])  # type: ignore
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        self.parent_remote.send(["reset", kwargs])

        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            obs, info = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs, info
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        ret = self.parent_remote.recv()
        return ret

    def render(self, **kwargs: Any) -> Any:
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        try:
            self.parent_remote.send(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        self.process.terminate()

    def check_success(self):
        self.parent_remote.send(["check_success", None])
        return self.parent_remote.recv()

    def get_segmentation_of_interest(self, segmentation_image):
        self.parent_remote.send(["get_segmentation_of_interest", segmentation_image])
        return self.parent_remote.recv()

    def get_sim_state(self):
        self.parent_remote.send(["get_sim_state", None])
        return self.parent_remote.recv()

    def set_init_state(self, init_state):
        self.parent_remote.send(["set_init_state", init_state])
        obs = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs


################################################################################
#
# VecEnvs
#
################################################################################


class BaseVectorEnv(object):
    """Base class for vectorized environments.

    Usage:
    ::

        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments

    .. warning::

        If you use your own environment, please make sure the ``seed`` method
        is set up properly, e.g.,
        ::

            def seed(self, seed):
                np.random.seed(seed)

        Otherwise, the outputs of these envs may be the same with each other.

    :param env_fns: a list of callable envs, ``env_fns[i]()`` generates the i-th env.
    :param worker_fn: a callable worker, ``worker_fn(env_fns[i])`` generates a
        worker which contains the i-th env.
    :param int wait_num: use in asynchronous simulation if the time cost of
        ``env.step`` varies with time and synchronously waiting for all
        environments to finish a step is time-wasting. In that case, we can
        return when ``wait_num`` environments finish a step and keep on
        simulation in these environments. If ``None``, asynchronous simulation
        is disabled; else, ``1 <= wait_num <= env_num``.
    :param float timeout: use in asynchronous simulation same as above, in each
        vectorized step it only deal with those environments spending time
        within ``timeout`` seconds.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
        stack_obs: bool = True,
    ) -> None:
        self._env_fns = env_fns
        self._stack_obs = stack_obs
        # A VectorEnv contains a pool of EnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert (
            1 <= self.wait_num <= len(env_fns)
        ), f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        self.timeout = timeout
        assert (
            self.timeout is None or self.timeout > 0
        ), f"timeout is {timeout}, it should be positive if provided!"
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: List[EnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: List[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

    def _assert_is_not_closed(self) -> None:
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.

        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> List[Any]:
        """Get an attribute from the underlying environments.

        If id is an int, retrieve the attribute denoted by key from the environment
        underlying the worker at index id. The result is returned as a list with one
        element. Otherwise, retrieve the attribute for all workers at indices id and
        return a list that is ordered correspondingly to id.

        :param str key: The key of the desired attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.

        :return list: The list of environment attributes.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        return [self.workers[j].get_env_attr(key) for j in id]

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> None:
        """Set an attribute in the underlying environments.

        If id is an int, set the attribute denoted by key from the environment
        underlying the worker at index id to value.
        Otherwise, set the attribute for all workers at indices id.

        :param str key: The key of the desired attribute.
        :param Any value: The new value of the attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def _wrap_id(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: Union[List[int], np.ndarray]) -> None:
        for i in id:
            assert (
                i not in self.waiting_id
            ), f"Cannot interact with environment {i} which is stepping now."
            assert (
                i in self.ready_id
            ), f"Can only interact with ready environments {self.ready_id}."

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        for i in id:
            self.workers[i].send(None, **kwargs)
        ret_list = [self.workers[i].recv() for i in id]

        reset_returns_info = (
            isinstance(ret_list[0], (tuple, list))
            and len(ret_list[0]) == 2
            and isinstance(ret_list[0][1], dict)
        )
        if reset_returns_info:
            obs_list = [r[0] for r in ret_list]
        else:
            obs_list = ret_list

        if isinstance(obs_list[0], tuple):
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        try:
            obs = jax.tree.map(lambda *args: np.stack(args), *obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)

        if reset_returns_info:
            infos = [r[1] for r in ret_list]
            return obs, infos  # type: ignore
        else:
            return obs

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple consisting of either:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

            or:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``terminated`` a numpy.ndarray, whether these episodes have been \
                terminated
            * ``truncated`` a numpy.ndarray, whether these episodes have been truncated
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

            The case distinction is made based on whether the underlying environment
            uses the old step API (first case) or the new step API (second case).

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            for i, j in enumerate(id):
                self.workers[j].send(action[i])
            result = []
            for j in id:
                env_return = self.workers[j].recv()
                env_return[-1]["env_id"] = j
                result.append(env_return)
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id):
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                # env_return can be (obs, reward, done, info) or
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()
                env_return[-1]["env_id"] = env_id  # Add `env_id` to info
                result.append(env_return)
                self.ready_id.append(env_id)
        return_lists = tuple(zip(*result))
        obs_list = return_lists[0]
        try:
            obs_stack = jax.tree.map(lambda *args: np.stack(args), *obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)

        other_stacks = map(np.stack, return_lists[1:])
        return (obs_stack, *other_stacks)  # type: ignore

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> List[Optional[List[int]]]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.

        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True


class DummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, DummyEnvWorker, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs


import functools
import gc
import logging
import os
import weakref
from typing import Any, Dict

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from src.training.config import _CONFIGS_DICT
# from openpi.training.config import (
#     _CONFIGS,
#     TrainConfig,
#     DataConfig,
#     pi0_config,
#     LeRobotLiberoDataConfig,
# )
import openpi.transforms as _transforms
from openpi.policies import policy_config
from openpi_client import image_tools
from src.rl.agent import Agent
from src.rl.filtered_sft_agent.update import train_step
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig
from src.training.data_loader import create_data_loader
from src.envs.wrappers import Pi0ObservationWrapper, QueryFrequencyWrapper
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.agent import Agent, EnvFn

def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True
    )

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )

@at.typecheck
def init_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            nnx.replace_by_pure_dict(state, partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, nnx.to_pure_dict(train_state_shape.params)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding

class DSRLVectorEnv(SubprocVectorEnv):
    """Vectorized environment wrapper based on subprocess for DSRL."""
    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, **kwargs)
        self._config = _CONFIGS_DICT["pi05_libero_online"]
        self._rng = jax.random.key(self._config.seed)
        init_rng, self._rng = jax.random.split(self._rng, 2)

        # set up sharding
        self._mesh = sharding.make_mesh(self._config.fsdp_devices)
        self._policy_sharding_spec = jax.sharding.NamedSharding(
            jax.sharding.Mesh(
                mesh_utils.create_device_mesh((len(jax.devices()),)),
                axis_names=("batch",),
            ),
            jax.sharding.PartitionSpec(
                "batch",
            ),
        )
        # Initialize checkpoint manager.
        self._checkpoint_manager, self._resuming = (
            _checkpoints.initialize_checkpoint_dir(
                self._config.checkpoint_dir,
                keep_period=self._config.keep_period,
                overwrite=self._config.overwrite,
                resume=self._config.resume,
            )
        )
        # Initialize train state.
        self._train_state, self._train_state_sharding = init_train_state(
            self._config, init_rng, self._mesh, resume=self._resuming
        )
        jax.block_until_ready(self._train_state)
        logging.info(
            f"Initialized train state:\n{training_utils.array_tree_to_info(self._train_state.params)}"
        )
        if self._resuming:
            self._train_state = _checkpoints.restore_state(
                self._checkpoint_manager, self._train_state, self._data_loader
            )
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        self.model = nnx.merge(self._train_state.model_def, params)
        
        policy_checkpoint_dir = os.environ.get("OPENPI_POLICY_CHECKPOINT_DIR")
        if policy_checkpoint_dir is None and isinstance(
            self._config.weight_loader,
            _weight_loaders.CheckpointWeightLoader,
        ):
            params_path = self._config.weight_loader.params_path
            if params_path.endswith("/params"):
                policy_checkpoint_dir = params_path[: -len("/params")]
            else:
                policy_checkpoint_dir = params_path
        if policy_checkpoint_dir is None:
            policy_checkpoint_dir = self._checkpoint_manager._directory
            if not (policy_checkpoint_dir / "params").exists():
                raise FileNotFoundError(
                    "Policy checkpoint not found. Set OPENPI_POLICY_CHECKPOINT_DIR to a checkpoint "
                    "containing 'params' (e.g. .../openpi-assets/checkpoints/pi05_libero)."
                )
        self._policy = policy_config.create_trained_policy(
            self._config,
            policy_checkpoint_dir,
        )
        # This learner always calls `infer_with_model(...)` with the current train-state model.
        # Drop policy-owned model references to avoid keeping an extra model copy in memory.
        self._drop_policy_model()

    def _drop_policy_model(self):
        # For PyTorch policies `infer_with_model` ignores the provided model and uses internal state,
        # so we cannot safely drop the internal model there.
        if getattr(self._policy, "_is_pytorch_model", False):
            return

        model = getattr(self._policy, "_model", None)
        model_ref = None
        if model is not None:
            try:
                model_ref = weakref.ref(model)
            except TypeError:
                model_ref = None

        self._policy._model = None
        # These JAX callables are created from bound model methods and can capture model state.
        if hasattr(self._policy, "_sample_actions"):
            self._policy._sample_actions = None
        if hasattr(self._policy, "_get_prefix_rep"):
            self._policy._get_prefix_rep = None

        del model
        gc.collect()

        if model_ref is not None and model_ref() is not None:
            logging.warning(
                "Policy model object is still alive after cleanup; other references remain."
            )

    def _process_obs_for_pi0(
        self,
        observations: Dict,
        task_description: str | None = None,
    ) -> Dict[str, Any]:
        # With per-step collection enabled, each env step contains a short chunk of
        # observations. Use the most recent one for policy inference.
        if self._config.collect.add_per_step_data:
            current_obs = jax.tree_util.tree_map(
                lambda x: x[:, -1], observations["observation"]
            )
        else:
            current_obs = observations["observation"]

        processed_obs = {}
        prompt_in_obs = False
        for key, val in current_obs.items():
            # Extract all observations relevant for the policy
            if self._config.collect.obs_prefix_key in key:
                obs_key = key.split(f"{self._config.collect.obs_prefix_key}/")[-1]
                if obs_key == "prompt":
                    prompt_in_obs = True
                    processed_obs[obs_key] = val
                else:
                    if "image" in obs_key and self._config.collect.resize_image > 0:
                        # Rescale images
                        val = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                val,
                                self._config.collect.resize_image,
                                self._config.collect.resize_image,
                            )
                        )
                    obs_key = f"observation/{obs_key}"
                    processed_obs[obs_key] = val
        # If prompt is not stored in obs, we add the default prompt here.
        if not prompt_in_obs:
            assert task_description is not None, "No task description is provided"
            processed_obs["prompt"] = task_description
        return processed_obs

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.
        Perform a dummy step to get the prefix representation.
        """        
        reset_returns = super().reset(id, **kwargs)
        if isinstance(reset_returns, tuple):
            obs, info = reset_returns
            processed_obs = self._process_obs_for_pi0(
                obs, task_description="dummy"
            )
            dummy_noise = jnp.zeros((len(obs["observation"]["state"]), self._policy.action_horizon, self._policy.action_dim))
            outputs = self._policy.infer_with_model(
                model=self.model,
                obs=processed_obs, # Use processed obs
                noise=dummy_noise,
                return_prefix_rep=True,
                sharding_spec=self._policy_sharding_spec,
            )
            prefix_rep = outputs["prefix_rep"]
            
            if isinstance(obs, dict):
                obs["prefix_rep"] = np.array(prefix_rep)
                self._last_obs = obs # Track last obs
                return obs, info
            self._last_obs = np.concatenate([obs, prefix_rep], axis=1) # Track last obs
            return self._last_obs, info
        else:
            obs = reset_returns
            processed_obs = self._process_obs_for_pi0(
                obs, task_description="dummy"
            )
            dummy_noise = jnp.zeros((len(obs["observation"]["state"]), self._policy.action_horizon, self._policy.action_dim))
            outputs = self._policy.infer_with_model(
                model=self.model,
                obs=processed_obs, # Use processed obs
                noise=dummy_noise,
                return_prefix_rep=True,
                sharding_spec=self._policy_sharding_spec,
            )
            prefix_rep = outputs["prefix_rep"]
            
            if isinstance(obs, dict):
                 obs["prefix_rep"] = np.array(prefix_rep)
                 self._last_obs = obs # Track last obs
                 return obs
            self._last_obs = np.concatenate([obs, prefix_rep], axis=1) # Track last obs
            return self._last_obs

    def step(
        self,
        noise: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Run one timestep of the environment with the given the DSRL policy noise.
        The noise is passed to the base policy's infer_with_model method.

        Args:
            noise: The noise to take in the environment.
            id: The id of the environment to take the action in.

        Returns:
            The observations with prefix representation, rewards, dones, and infos.
        """
        if id is not None:
             raise NotImplementedError("Partially stepping DSRLVectorEnv is not supported yet because of state tracking complexity.")

        # Use the stored _last_obs
        processed_obs = self._process_obs_for_pi0(
             self._last_obs, task_description="dummy"
        )
        
        # Vector envs expect a batch dimension for actions. Policy inference
        # unbatches when batch_size == 1, so add it back for single-env runs.
        outputs = self._policy.infer_with_model(
            model=self.model,
            obs=processed_obs,
            noise=noise,
            return_prefix_rep=True,
            sharding_spec=self._policy_sharding_spec,
        )
        actions = outputs["actions"]
        prefix_rep = outputs["prefix_rep"]
        
        return_stacks = super().step(actions, id)
        obs_stack = return_stacks[0]
        
        if isinstance(obs_stack, dict):
             obs_stack["prefix_rep"] = np.array(prefix_rep)
             self._last_obs = obs_stack
             return (obs_stack, *return_stacks[1:]) # type: ignore
             
        obs_stack = np.concatenate([obs_stack, prefix_rep], axis=1)
        self._last_obs = obs_stack
        return (obs_stack, *return_stacks[1:])  # type: ignore
