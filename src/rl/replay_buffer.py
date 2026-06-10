from typing import Optional, Any
import h5py
import jax
import numpy as np
import logging
import json
import os
from pathlib import Path
import tempfile

from src.rl.dataset import read_nested, write_nested
from flax.core import frozen_dict


class ShardedReplayBuffer:
    def __init__(
        self,
        dummy_data: Any,
        max_capacity: int,
        batch_size: int | None = None,
        data_sharding: jax.sharding.NamedSharding | None = None,
        seed: Optional[int] = None,
        freeze_dict: bool = True,
    ):
        """
        Args:
            dummy_data: A sample dictionary to define shapes/dtypes.
            max_capacity: Total number of transitions to store in RAM.
            batch_size: The global batch size for sampling.
            data_sharding: JAX sharding spec for the output batches.
            seed: Random seed for sampling.
            freeze_dict: Whether to freeze the output dicts
        """
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.data_sharding = data_sharding
        self.freeze_dict = freeze_dict
        self.ptr = 0
        self.size = 0
        self.total_inserted = 0
        self.persisted_total_inserted = 0

        self._allocate_storage(dummy_data, max_capacity)
        self._dummy = jax.tree_util.tree_map(lambda l: np.asarray(l)[:1], dummy_data)

        self._rng = np.random.default_rng(seed)

    def _allocate_storage(self, dummy_data: Any, capacity: int) -> Any:
        def create_buffer(leaf_array):
            return np.zeros((capacity,) + leaf_array.shape[1:], dtype=leaf_array.dtype)
        self.storage = jax.tree_util.tree_map(create_buffer, dummy_data)
        self._storage_leaves, self._storage_treedef = jax.tree_util.tree_flatten(self.storage)

    def insert(self, data: Any):

        data_leaves, data_treedef = jax.tree_util.tree_flatten(data)
        if self._storage_treedef != data_treedef:
            raise ValueError("Insert data structure does not match buffer structure")
        num_new = int(data_leaves[0].shape[0])
        if num_new <= 0:
            return

        indices = (np.arange(self.ptr, self.ptr + num_new) % self.max_capacity).astype(np.int32)
        for storage_leaf, new_data_leaf in zip(self._storage_leaves, data_leaves):
            storage_leaf[indices] = new_data_leaf

        self.ptr = int((self.ptr + num_new) % self.max_capacity)
        self.size = int(min(self.size + num_new, self.max_capacity))
        self.total_inserted += num_new

    def sample(self, batch_size=None) -> Any:
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

        if self.freeze_dict and isinstance(batch, dict):
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

    def _recent_indices(self, count: int) -> np.ndarray:
        if count < 0 or count > self.size:
            raise ValueError(f"Requested {count} recent transitions from buffer size {self.size}.")
        start = (self.ptr - count) % self.max_capacity
        return (np.arange(count, dtype=np.int32) + start) % self.max_capacity

    def _slice_storage(self, indices: np.ndarray) -> Any:
        return jax.tree_util.tree_map(lambda leaf: leaf[indices].copy(), self.storage)

    def save_shard(self, path: str | Path) -> dict[str, int | str | None]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        delta_count = self.total_inserted - self.persisted_total_inserted
        delta_count = min(delta_count, self.size)  # this can cause non-determinism
        if delta_count < 0:
            raise ValueError("Replay buffer total_inserted went backwards.")
        if delta_count == 0:
            logging.info("No new replay transitions to save; reusing latest shard.")
            return

        shard_data = self._slice_storage(self._recent_indices(delta_count))
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
                metadata.attrs["num_transitions"] = delta_count
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        self.persisted_total_inserted = self.total_inserted
        logging.info(
            "Saved replay buffer shard to %s (new transitions=%d, replay size=%d)",
            path,
            delta_count,
            self.size,
        )

    def restore_shards(
        self,
        shard_dir: str | Path,
        *,
        rng_state_json: str | None = None,
    ) -> dict[str, int | str | None]:
        shard_dir = Path(shard_dir)
        if not shard_dir.exists():
            raise FileNotFoundError(f"Replay shard directory does not exist: {shard_dir}")

        self.ptr = 0
        self.size = 0
        self.total_inserted = 0
        shard_paths = sorted(shard_dir.glob("step_*.h5"))

        for shard_path in shard_paths:
            with h5py.File(shard_path, "r") as f:
                restored_data = read_nested(f["transitions"])
            self.insert(restored_data)

        self.persisted_total_inserted = self.total_inserted
        self.set_rng_state_json(rng_state_json)

        logging.info(
            "Restored replay buffer from shards in %s (transitions=%d)",
            shard_dir,
            self.size,
        )
