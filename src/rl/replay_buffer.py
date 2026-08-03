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


# Keys that, when present in `dummy_data`, switch the buffer into
# "linked observation" mode (see ShardedReplayBuffer docstring).
_LINKED_OBS_KEY = "observations"
_LINKED_INDEX_KEYS = ("obs_index", "next_obs_index")


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

        self._allocate_linked_storage(dummy_data)

        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _zeros_like_tree(template: Any, capacity: int) -> Any:
        def create_buffer(leaf_array):
            leaf_array = np.asarray(leaf_array)
            return np.zeros((capacity,) + leaf_array.shape[1:], dtype=leaf_array.dtype)
        return jax.tree_util.tree_map(create_buffer, template)

    def _allocate_linked_storage(self, dummy_data: dict) -> None:
        observations = dummy_data[_LINKED_OBS_KEY]
        transition_dummy = {
            k: v for k, v in dummy_data.items()
            if k != _LINKED_OBS_KEY and k not in _LINKED_INDEX_KEYS
        }

        self.obs_storage = self._zeros_like_tree(observations, self.max_capacity)
        self._obs_leaves, self._obs_treedef = jax.tree_util.tree_flatten(self.obs_storage)
        self.obs_ptr = 0
        self.obs_total = 0

        self.storage = self._zeros_like_tree(transition_dummy, self.max_capacity)
        self._storage_leaves, self._storage_treedef = jax.tree_util.tree_flatten(self.storage)
        self.obs_pos = np.zeros((self.max_capacity,), dtype=np.int64)
        self.next_obs_pos = np.zeros((self.max_capacity,), dtype=np.int64)

        self.valid_start = 0

    def insert(self, data: Any):
        observations = data[_LINKED_OBS_KEY]
        obs_index = np.asarray(data["obs_index"])
        next_obs_index = np.asarray(data["next_obs_index"])

        obs_leaves, obs_treedef = jax.tree_util.tree_flatten(observations)
        if obs_treedef != self._obs_treedef:
            raise ValueError("Insert observation structure does not match buffer structure")
        transition = {
            k: v for k, v in data.items()
            if k != _LINKED_OBS_KEY and k not in _LINKED_INDEX_KEYS
        }
        txn_leaves, txn_treedef = jax.tree_util.tree_flatten(transition)
        if txn_treedef != self._storage_treedef:
            raise ValueError("Insert transition structure does not match buffer structure")

        # Some leaves are 0-d (LiberoInputs emits image_mask as np.True_/np.False_
        # scalars), so take the batch size from the first batched leaf rather than
        # leaf 0 — which is only an image while "image" is present in the tree.
        num_obs = next(
            (int(leaf.shape[0]) for leaf in obs_leaves if np.ndim(leaf) >= 1), 0
        )
        num_new = int(obs_index.shape[0])
        if num_new <= 0:
            return
        if num_obs > self.max_capacity:
            raise ValueError(
                f"Single insert writes {num_obs} observations into a ring of capacity "
                f"{self.max_capacity}; increase max_capacity."
            )

        # Write the unique observations into the observation ring.
        base = self.obs_total
        obs_ring_idx = (np.arange(base, base + num_obs) % self.max_capacity).astype(np.int64)
        for storage_leaf, new_leaf in zip(self._obs_leaves, obs_leaves):
            storage_leaf[obs_ring_idx] = new_leaf
        self.obs_total += num_obs
        self.obs_ptr = int((self.obs_ptr + num_obs) % self.max_capacity)

        # Write the transitions + absolute observation pointers.
        txn_idx = (np.arange(self.ptr, self.ptr + num_new) % self.max_capacity).astype(np.int64)
        for storage_leaf, new_leaf in zip(self._storage_leaves, txn_leaves):
            storage_leaf[txn_idx] = new_leaf
        self.obs_pos[txn_idx] = base + obs_index.astype(np.int64)
        self.next_obs_pos[txn_idx] = base + next_obs_index.astype(np.int64)

        self.ptr = int((self.ptr + num_new) % self.max_capacity)
        self.total_inserted += num_new
        self._advance_valid_start()

    def _advance_valid_start(self) -> None:
        self.valid_start = max(self.valid_start, self.total_inserted - self.max_capacity)
        oldest_resident_obs = self.obs_total - self.max_capacity
        while (
            self.valid_start < self.total_inserted
            and int(self.obs_pos[self.valid_start % self.max_capacity]) < oldest_resident_obs
        ):
            self.valid_start += 1
        self.size = int(self.total_inserted - self.valid_start)

    def sample(self, batch_size=None) -> Any:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if batch_size is None:
            assert self.batch_size is not None, "Batch size must be specified for sampling"
            batch_size = self.batch_size

        ordinals = self._rng.integers(self.valid_start, self.total_inserted, size=batch_size)
        ring = (ordinals % self.max_capacity).astype(np.int64)

        transition = self._storage_treedef.unflatten([leaf[ring] for leaf in self._storage_leaves])
        obs_idx = (self.obs_pos[ring] % self.max_capacity)
        next_idx = (self.next_obs_pos[ring] % self.max_capacity)
        observation = jax.tree_util.tree_map(lambda leaf: leaf[obs_idx], self.obs_storage)
        next_observation = jax.tree_util.tree_map(lambda leaf: leaf[next_idx], self.obs_storage)

        batch = {"observation": observation, "next_observation": next_observation}
        batch.update(transition)
        # device_put broadcasts a single sharding across the whole pytree, so we
        # shard the assembled batch in one call rather than leaf by leaf.
        if self.data_sharding is not None:
            batch = jax.device_put(batch, self.data_sharding)
        if self.freeze_dict:
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
        delta_count = self.total_inserted - self.persisted_total_inserted
        delta_count = min(delta_count, self.size)  # this can cause non-determinism
        if delta_count < 0:
            raise ValueError("Replay buffer total_inserted went backwards.")
        if delta_count == 0:
            logging.info("No new replay transitions to save; reusing latest shard.")
            return

        # Transitions to persist (the most recent `delta_count`).
        ordinals = np.arange(self.total_inserted - delta_count, self.total_inserted)
        ring = (ordinals % self.max_capacity).astype(np.int64)

        # Observations referenced by these transitions form a contiguous ordinal
        # range; persist that range once and rebase the pointers to it.
        obs_lo = int(self.obs_pos[ring].min())
        obs_hi = int(self.next_obs_pos[ring].max()) + 1  # exclusive
        obs_ordinals = np.arange(obs_lo, obs_hi)
        obs_ring_idx = (obs_ordinals % self.max_capacity).astype(np.int64)
        obs_slice = jax.tree_util.tree_map(
            lambda leaf: leaf[obs_ring_idx].copy(), self.obs_storage
        )
        txn_slice = jax.tree_util.tree_map(lambda leaf: leaf[ring].copy(), self.storage)
        rel_obs_index = (self.obs_pos[ring] - obs_lo).astype(np.int64)
        rel_next_obs_index = (self.next_obs_pos[ring] - obs_lo).astype(np.int64)

        def write_fn(f):
            write_nested(f.create_group("observations"), obs_slice)
            write_nested(f.create_group("transitions"), txn_slice)
            links = f.create_group("links")
            links.create_dataset("obs_index", data=rel_obs_index)
            links.create_dataset("next_obs_index", data=rel_next_obs_index)

        self._write_h5_atomic(path, write_fn)
        self.persisted_total_inserted = self.total_inserted
        logging.info(
            "Saved replay shard to %s (transitions=%d, observations=%d, replay size=%d)",
            path, delta_count, int(obs_hi - obs_lo), self.size,
        )

    def _write_h5_atomic(self, path: Path, write_fn) -> None:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                write_fn(f)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

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
        self.obs_ptr = 0
        self.obs_total = 0
        self.valid_start = 0
        shard_paths = sorted(shard_dir.glob("step_*.h5"))

        for shard_path in shard_paths:
            with h5py.File(shard_path, "r") as f:
                restored = {
                    _LINKED_OBS_KEY: read_nested(f["observations"]),
                    "obs_index": f["links/obs_index"][()],
                    "next_obs_index": f["links/next_obs_index"][()],
                }
                restored.update(read_nested(f["transitions"]))
            self.insert(restored)

        self.persisted_total_inserted = self.total_inserted
        self.set_rng_state_json(rng_state_json)

        logging.info(
            "Restored replay buffer from shards in %s (transitions=%d)",
            shard_dir, self.size,
        )
