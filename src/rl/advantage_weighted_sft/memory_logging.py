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


def log_memory_debug(tag: str, train_state=None, batch=None, training_steps: int = 0, **extra_pytrees):
    """Log memory and array diagnostics at various checkpoints."""
    if tag == "init":
        parts = [f"{name}: {_pytree_size_mb(tree):.2f} MB" for name, tree in extra_pytrees.items()]
        logging.info(f"[OOM-DEBUG init] {', '.join(parts)}")
    elif tag == "step_start":
        live_arrays = jax.live_arrays()
        total_bytes = sum(arr.nbytes for arr in live_arrays)
        total_gb = total_bytes / (1024**3)
        print(f"Step {training_steps + 1} Memory Check")
        print(f"Total live arrays on device: {len(live_arrays)}")
        print(f"Total tracked memory: {total_gb:.2f} GB")
    elif tag == "before_critics":
        _log_device_memory("before_get_policy_model")
        if train_state is not None:
            logging.info(
                f"[SIZE-DEBUG] train_state total (global): "
                f"params={_pytree_size_gb(train_state.params):.2f} GiB, "
                f"ema_params={_pytree_size_gb(train_state.ema_params):.2f} GiB, "
                f"opt_state={_pytree_size_gb(train_state.opt_state):.2f} GiB"
            )
            logging.info(
                f"[SIZE-DEBUG] train_state per-device: "
                f"params={_pytree_per_device_size_gb(train_state.params):.2f} GiB, "
                f"ema_params={_pytree_per_device_size_gb(train_state.ema_params):.2f} GiB, "
                f"opt_state={_pytree_per_device_size_gb(train_state.opt_state):.2f} GiB"
            )
        if batch is not None:
            logging.info(
                f"[SIZE-DEBUG] SFT batch: global={_pytree_size_gb(batch):.2f} GiB, "
                f"per-device={_pytree_per_device_size_gb(batch):.2f} GiB"
            )
    elif tag == "after_update_critics":
        _log_device_memory("after_update_critics")
    elif tag == "before_update_policy":
        _log_device_memory("before_update_policy (after del batch+policy_model)")
