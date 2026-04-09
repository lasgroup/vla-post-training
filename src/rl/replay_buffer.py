from typing import Union
from typing import Iterable, Optional, Any, Callable
import h5py
import jax
import gymnasium as gym
import numpy as np
import pickle
import logging
import json
import os
from pathlib import Path
import tempfile

import copy

from src.rl.dataset import Dataset, DatasetDict, read_nested, write_nested
import collections
from flax.core import frozen_dict

# Type alias for clarity
NestedData = Any  # Can be Dict, List, Tuple, etc. containing Arrays


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


class ReplayBuffer(Dataset):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity

        print("making replay buffer of capacity ", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        next_observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty(
            (self.capacity, *self.action_space.shape), dtype=self.action_space.dtype
        )
        next_actions = np.empty(
            (self.capacity, *self.action_space.shape), dtype=self.action_space.dtype
        )
        rewards = np.empty((self.capacity,), dtype=np.float32)
        masks = np.empty((self.capacity,), dtype=np.float32)
        discount = np.empty((self.capacity,), dtype=np.float32)

        self.data = {
            "observations": observations,
            "next_observations": next_observations,
            "actions": actions,
            "next_actions": next_actions,
            "rewards": rewards,
            "masks": masks,
            "discount": discount,
        }

        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = dict()
        self.streaming_buffer_size = None  # this is for streaming the online data

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    def increment_traj_counter(self):
        self.traj_bounds[self._traj_counter] = (self._start, self.size)  # [start, end)
        self._start = self.size
        self._traj_counter += 1

    def get_random_trajs(self, num_trajs: int):
        self.which_trajs = np.random.randint(0, self._traj_counter, num_trajs)
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []

        for i in self.which_trajs:
            start, end = self.traj_bounds[i]

            # handle this as a dictionary
            obs_dict_curr_traj = dict()
            for k in self.data["observations"]:
                obs_dict_curr_traj[k] = self.data["observations"][k][start:end]
            observations_list.append(obs_dict_curr_traj)

            next_obs_dict_curr_traj = dict()
            for k in self.data["next_observations"]:
                next_obs_dict_curr_traj[k] = self.data["next_observations"][k][
                    start:end
                ]
            next_observations_list.append(next_obs_dict_curr_traj)

            actions_list.append(self.data["actions"][start:end])
            rewards_list.append(self.data["rewards"][start:end])
            terminals_list.append(1 - self.data["masks"][start:end])
            masks_list.append(self.data["masks"][start:end])

        batch = {
            "observations": observations_list,
            "next_observations": next_observations_list,
            "actions": actions_list,
            "rewards": rewards_list,
            "terminals": terminals_list,
            "masks": masks_list,
        }
        return batch

    def insert(self, data_dict: DatasetDict):
        if self.size == self.capacity:
            # Double the capacity
            observations = _init_replay_dict(self.observation_space, self.capacity)
            next_observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty(
                (self.capacity, *self.action_space.shape), dtype=self.action_space.dtype
            )
            next_actions = np.empty(
                (self.capacity, *self.action_space.shape), dtype=self.action_space.dtype
            )
            rewards = np.empty((self.capacity,), dtype=np.float32)
            masks = np.empty((self.capacity,), dtype=np.float32)
            discount = np.empty((self.capacity,), dtype=np.float32)

            data_new = {
                "observations": observations,
                "next_observations": next_observations,
                "actions": actions,
                "next_actions": next_actions,
                "rewards": rewards,
                "masks": masks,
                "discount": discount,
            }

            for x in data_new:
                if isinstance(self.data[x], np.ndarray):
                    self.data[x] = np.concatenate((self.data[x], data_new[x]), axis=0)
                elif isinstance(self.data[x], dict):
                    for y in self.data[x]:
                        self.data[x][y] = np.concatenate(
                            (self.data[x][y], data_new[x][y]), axis=0
                        )
                else:
                    raise TypeError()
            self.capacity *= 2

        for x in data_dict:
            if x in self.data:
                if isinstance(data_dict[x], dict):
                    for y in data_dict[x]:
                        self.data[x][y][self.size] = data_dict[x][y]
                else:
                    self.data[x][self.size] = data_dict[x]
        self.size += 1

    def compute_action_stats(self):
        # Only compute stats over populated transitions.
        actions = self.data["actions"][: self.size]
        return {"mean": actions.mean(axis=0), "std": actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        # Avoid mutating the caller's dict.
        action_stats = copy.deepcopy(action_stats)
        action_stats["mean"][-1] = 0
        action_stats["std"][-1] = 1
        self.data["actions"] = (
            self.data["actions"] - action_stats["mean"]
        ) / action_stats["std"]
        self.data["next_actions"] = (
            self.data["next_actions"] - action_stats["mean"]
        ) / action_stats["std"]

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        if self.streaming_buffer_size:
            indices = np.random.randint(0, self.streaming_buffer_size, batch_size)
        else:
            indices = np.random.randint(0, self.size, batch_size)
        data_dict = {}
        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                data_dict[x] = self.data[x][indices]
            elif isinstance(self.data[x], dict):
                data_dict[x] = {}
                for y in self.data[x]:
                    data_dict[x][y] = self.data[x][y][indices]
            else:
                raise TypeError()

        return frozen_dict.freeze(data_dict)

    def sample_with_indices(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> tuple[frozen_dict.FrozenDict, np.ndarray]:
        if self.streaming_buffer_size:
            indices = np.random.randint(0, self.streaming_buffer_size, batch_size)
        else:
            indices = np.random.randint(0, self.size, batch_size)
        data_dict = {}
        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                data_dict[x] = self.data[x][indices]
            elif isinstance(self.data[x], dict):
                data_dict[x] = {}
                for y in self.data[x]:
                    data_dict[x][y] = self.data[x][y][indices]
            else:
                raise TypeError()
        return frozen_dict.freeze(data_dict), indices

    def get_iterator(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        queue_size: int = 2,
    ):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def get_iterator_with_indices(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        queue_size: int = 2,
    ):
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data, indices = self.sample_with_indices(batch_size, keys, indx)
                queue.append((jax.device_put(data), indices))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def save(self, filename):
        save_dict = dict(
            data=self.data,
            size=self.size,
            _traj_counter=self._traj_counter,
            _start=self._start,
            traj_bounds=self.traj_bounds,
        )
        with open(filename, "wb") as f:
            pickle.dump(save_dict, f, protocol=4)

    def restore(self, filename):
        # `save()` uses pickle, so restore must read via pickle too.
        with open(filename, "rb") as f:
            save_dict = pickle.load(f)

        self.data = save_dict["data"]
        self.size = int(save_dict["size"])
        self._traj_counter = int(save_dict["_traj_counter"])
        self._start = int(save_dict["_start"])
        self.traj_bounds = save_dict["traj_bounds"]

        # Keep capacity in sync with the underlying storage.
        def _capacity_from_storage(storage: NestedData) -> int:
            if isinstance(storage, np.ndarray):
                return int(storage.shape[0])
            if isinstance(storage, dict):
                # Grab first leaf deterministically.
                for v in storage.values():
                    return _capacity_from_storage(v)
            raise TypeError(f"Unsupported storage type: {type(storage)}")

        self.capacity = _capacity_from_storage(self.data["actions"])


class ShardedReplayBuffer:
    def __init__(
        self,
        dummy_data: NestedData,
        max_capacity: int,
        batch_size: int | None = None,
        data_sharding: jax.sharding.NamedSharding | None = None,
        seed: Optional[int] = None,
        preprocess_fn: Callable[[NestedData], NestedData] | None = None,
        postprocess_fn: Callable[[NestedData], Any] | None = None,
        freeze_dict: bool = True,
    ):
        """
        Args:
            dummy_data: A sample dictionary to define shapes/dtypes.
            max_capacity: Total number of transitions to store in RAM.
            batch_size: The global batch size for sampling.
            data_sharding: JAX sharding spec for the output batches.
        """
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.data_sharding = data_sharding
        self.ptr = 0
        self.size = 0
        self._preprocess_fn = preprocess_fn
        self._postprocess_fn = postprocess_fn
        self._freeze_dict = freeze_dict

        # 1. Pre-allocate the entire buffer in Host RAM (NumPy)
        # This prevents memory fragmentation during long training runs.
        self.storage = self._allocate_storage(dummy_data, max_capacity)
        self._refresh_storage_views()

        # Seeding
        self._rng = np.random.default_rng(seed)
        self.total_inserted = 0
        self._persisted_total_inserted = 0
        self._latest_saved_shard_path: Path | None = None

    @staticmethod
    def _allocate_storage(dummy_data: NestedData, capacity: int) -> NestedData:
        def create_buffer(leaf_array):
            # leaf_array shape: (batch, features...) -> storage shape: (capacity, features...)
            buffer_shape = (capacity,) + leaf_array.shape[1:]
            return np.zeros(buffer_shape, dtype=leaf_array.dtype)

        return jax.tree_util.tree_map(create_buffer, dummy_data)

    def _refresh_storage_views(self) -> None:
        self._storage_leaves, self._storage_treedef = jax.tree_util.tree_flatten(
            self.storage
        )

    def _ordered_indices(self, count: int) -> np.ndarray:
        if count < 0 or count > self.size:
            raise ValueError(f"Requested {count} active transitions from buffer size {self.size}.")
        start = (self.ptr - self.size) % self.max_capacity
        return (np.arange(count, dtype=np.int32) + start) % self.max_capacity

    def _recent_indices(self, count: int) -> np.ndarray:
        if count < 0 or count > self.size:
            raise ValueError(f"Requested {count} recent transitions from buffer size {self.size}.")
        start = (self.ptr - count) % self.max_capacity
        return (np.arange(count, dtype=np.int32) + start) % self.max_capacity

    def _slice_storage(self, indices: np.ndarray) -> NestedData:
        return jax.tree_util.tree_map(lambda leaf: leaf[indices].copy(), self.storage)

    def insert(self, data: NestedData):
        """
        Inserts nested data into the buffer.
        Assumes data structure matches the initialized dummy_data.
        """

        if self._preprocess_fn is not None:
            data = self._preprocess_fn(data)
        data_leaves, data_treedef = jax.tree_util.tree_flatten(data)

        # Sanity check: ensure structures match
        if self._storage_treedef != data_treedef:
            raise ValueError("Insert data structure does not match buffer structure")

        # Get the number of new items from the first leaf
        num_new = int(data_leaves[0].shape[0])
        if num_new <= 0:
            return

        # Calculate circular buffer indices
        indices = (np.arange(self.ptr, self.ptr + num_new) % self.max_capacity).astype(
            np.int32
        )

        # Update every leaf array in the storage
        for storage_leaf, new_data_leaf in zip(self._storage_leaves, data_leaves):
            storage_leaf[indices] = new_data_leaf

        # Update pointers
        self.ptr = int((self.ptr + num_new) % self.max_capacity)
        self.size = int(min(self.size + num_new, self.max_capacity))
        self.total_inserted += num_new

    def sample(self, batch_size=None) -> NestedData:
        """
        Samples a nested batch and shards every leaf.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if batch_size is None:
            assert self.batch_size is not None, "Batch size must be specified for sampling"
            batch_size = self.batch_size
        indices = self._rng.integers(0, self.size, size=batch_size)

        def fetch_and_shard(buffer_leaf):
            batch_slice = buffer_leaf[indices]
            return (
                jax.device_put(batch_slice, self.data_sharding)
                if self.data_sharding
                else batch_slice
            )

        leaves = [fetch_and_shard(leaf) for leaf in self._storage_leaves]
        batch = self._storage_treedef.unflatten(leaves)

        if self._postprocess_fn is not None:
            return self._postprocess_fn(batch)
        if self._freeze_dict and isinstance(batch, dict):
            return frozen_dict.freeze(batch)
        return batch

    def __len__(self):
        return self.size

    def rng_state_json(self) -> str:
        return json.dumps(self._rng.bit_generator.state)

    def set_rng_state_json(self, rng_state_json: str | None) -> None:
        if rng_state_json is None:
            return
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = json.loads(rng_state_json)

    def save_shard(self, path: str | Path) -> dict[str, int | str | None]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        delta_count = self.total_inserted - self._persisted_total_inserted
        if delta_count < 0:
            raise ValueError("Replay buffer total_inserted went backwards.")
        if delta_count == 0:
            logging.info(
                "No new replay transitions to save; reusing latest shard %s",
                self._latest_saved_shard_path,
            )
            return {
                "size": int(self.size),
                "total_inserted": int(self.total_inserted),
                "path": (
                    None
                    if self._latest_saved_shard_path is None
                    else str(self._latest_saved_shard_path)
                ),
            }

        shard_data = self._slice_storage(self._recent_indices(delta_count))
        num_transitions = int(delta_count)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                write_nested(f.create_group("transitions"), shard_data)
                metadata = f.create_group("metadata")
                metadata.attrs["num_transitions"] = num_transitions
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        self._persisted_total_inserted = int(self.total_inserted)
        self._latest_saved_shard_path = path
        logging.info(
            "Saved replay buffer shard to %s (new transitions=%d, replay size=%d)",
            path,
            num_transitions,
            self.size,
        )
        return {
            "size": int(self.size),
            "total_inserted": int(self.total_inserted),
            "path": str(path),
        }

    def restore_shards(
        self,
        shard_dir: str | Path,
        *,
        step: int | None = None,
        total_inserted: int | None = None,
        latest_shard_path: str | Path | None = None,
        rng_state_json: str | None = None,
    ) -> dict[str, int | str | None]:
        shard_dir = Path(shard_dir)
        if not shard_dir.exists():
            raise FileNotFoundError(f"Replay shard directory does not exist: {shard_dir}")

        dummy_data = jax.tree_util.tree_map(lambda leaf: leaf[:1].copy(), self.storage)
        self.storage = self._allocate_storage(dummy_data, self.max_capacity)
        self.ptr = 0
        self.size = 0
        self.total_inserted = 0
        self._persisted_total_inserted = 0
        self._latest_saved_shard_path = None
        self._refresh_storage_views()
        shard_paths = sorted(shard_dir.glob("step_*.h5"))
        if step is not None:
            shard_paths = [p for p in shard_paths if int(p.stem.split("_")[-1]) <= int(step)]

        for shard_path in shard_paths:
            with h5py.File(shard_path, "r") as f:
                restored_data = read_nested(f["transitions"])
            self.insert(restored_data)
            self._latest_saved_shard_path = shard_path

        if total_inserted is not None:
            self.total_inserted = int(total_inserted)
        self._persisted_total_inserted = int(self.total_inserted)
        self._latest_saved_shard_path = (
            None if latest_shard_path is None else Path(latest_shard_path)
        )
        self.set_rng_state_json(rng_state_json)

        logging.info(
            "Restored replay buffer from shards in %s (step=%s, transitions=%d, latest shard=%s)",
            shard_dir,
            step,
            self.size,
            self._latest_saved_shard_path,
        )
        return {
            "step": (None if step is None else int(step)),
            "size": int(self.size),
            "total_inserted": int(self.total_inserted),
            "path": (
                None
                if self._latest_saved_shard_path is None
                else str(self._latest_saved_shard_path)
            ),
        }


if __name__ == "__main__":
    # 1. Complex Nested Structure (Dicts of Dicts of Arrays)
    dummy_data = {
        "observations": {
            "camera_front": np.zeros((1, 64, 64, 3), dtype=np.uint8),
            "camera_wrist": np.zeros((1, 64, 64, 3), dtype=np.uint8),
            "proprioception": {
                "joints": np.zeros((1, 7), dtype=np.float32),
                "gripper": np.zeros((1, 1), dtype=np.float32),
            },
        },
        "actions": np.zeros((1, 7), dtype=np.float32),
        "rewards": np.zeros((1, 1), dtype=np.float32),
    }

    # 2. Initialize
    buffer = ShardedReplayBuffer(
        dummy_data=dummy_data, max_capacity=100_000, batch_size=256, data_sharding=None
    )

    # 3. Insert Nested Data
    # (Assuming `get_step()` returns a dict matching the structure above)
    fake_data = {
        "observations": {
            "camera_front": np.zeros((256, 64, 64, 3), dtype=np.uint8),
            "camera_wrist": np.zeros((256, 64, 64, 3), dtype=np.uint8),
            "proprioception": {
                "joints": np.zeros((256, 7), dtype=np.float32),
                "gripper": np.zeros((256, 1), dtype=np.float32),
            },
        },
        "actions": np.zeros((256, 7), dtype=np.float32),
        "rewards": np.zeros((256, 1), dtype=np.float32),
    }
    buffer.insert(fake_data)

    # 4. Sample
    # `batch` will have the exact same structure as `dummy_data`,
    # but every leaf will be a Sharded JAX Array of size 256.
    batch = buffer.sample()

    print(batch["observations"]["proprioception"]["joints"].shape)
    # Output: (256, 7) on device
