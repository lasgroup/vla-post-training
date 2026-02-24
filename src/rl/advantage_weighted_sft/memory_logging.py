import logging

import jax


def _pytree_size_mb(tree) -> float:
    """Return total size of all arrays in a pytree, in megabytes."""
    leaves = jax.tree.leaves(tree)
    total_bytes = sum(
        leaf.size * leaf.dtype.itemsize for leaf in leaves if hasattr(leaf, "size")
    )
    return total_bytes / (1024 * 1024)


def _pytree_size_gb(tree) -> float:
    return _pytree_size_mb(tree) / 1024


def _pytree_per_device_size_gb(tree) -> float:
    """Return total per-device size of all arrays in a pytree, in GiB."""
    leaves = jax.tree.leaves(tree)
    total_bytes = 0
    for leaf in leaves:
        if not hasattr(leaf, "size"):
            continue
        if hasattr(leaf, "addressable_shards") and leaf.addressable_shards:
            shard = leaf.addressable_shards[0]
            total_bytes += shard.data.size * shard.data.dtype.itemsize
        else:
            total_bytes += leaf.size * leaf.dtype.itemsize
    return total_bytes / (1024 ** 3)


def _log_device_memory(tag: str) -> None:
    """Log live GPU memory for device 0 and count of live arrays."""
    jax.effects_barrier()  # wait for async dispatch to finish
    stats = jax.local_devices()[0].memory_stats()
    if stats is None:
        logging.info(f"[MEM {tag}] memory_stats unavailable")
        return
    live_gb = stats.get("bytes_in_use", 0) / (1024 ** 3)
    peak_gb = stats.get("peak_bytes_in_use", 0) / (1024 ** 3)
    limit_gb = stats.get("bytes_limit", 0) / (1024 ** 3)
    num_live = len(jax.live_arrays())
    logging.info(
        f"[MEM {tag}] live={live_gb:.2f} GiB, peak={peak_gb:.2f} GiB, "
        f"limit={limit_gb:.2f} GiB, num_live_arrays={num_live}"
    )
