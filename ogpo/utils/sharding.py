import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_mesh(gpu_mode="single"):
    if gpu_mode == "single":
        return None

    devices = jax.devices()
    if len(devices) <= 1:
        return None

    if gpu_mode == "multi":
        mesh_devices = devices
    elif gpu_mode.startswith("reserve:"):
        exclude = int(gpu_mode.split(":")[1])
        mesh_devices = [d for d in devices if d.id != exclude]
        if not mesh_devices:
            raise ValueError(f"No devices left after reserving GPU {exclude}")
    else:
        raise ValueError(
            f"Unknown gpu_mode: {gpu_mode!r}. Use 'single', 'multi', or 'reserve:N'"
        )

    return Mesh(mesh_devices, axis_names=('data',))


def shard_agent(agent, mesh):
    """Replicate all agent params/opt_state across mesh devices."""
    return jax.device_put(agent, NamedSharding(mesh, P()))


def shard_batch(batch, mesh):
    """Shard UTD-reshaped batch (utd_ratio, batch_size, ...) on axis 1."""
    s = NamedSharding(mesh, P(None, 'data'))
    return jax.tree.map(lambda x: jax.device_put(x, s), batch)


def shard_batch_flat(batch, mesh):
    """Shard flat batch (batch_size, ...) on axis 0. For BC/CalQL phases."""
    s = NamedSharding(mesh, P('data'))
    return jax.tree.map(lambda x: jax.device_put(x, s), batch)


def zero_spec(shape, n_dev):
    """ZeRO-1 PartitionSpec for one leaf: shard the first axis divisible by n_dev over 'data', else replicate."""
    for i, d in enumerate(shape):
        if d >= n_dev and d % n_dev == 0:
            return P(*([None] * i), 'data')
    return P()


def zero_shardings(tree, mesh):
    """Per-leaf `zero_spec` NamedShardings for a pytree of arrays (or tracers)."""
    n_dev = mesh.devices.size
    return jax.tree.map(lambda x: NamedSharding(mesh, zero_spec(x.shape, n_dev)), tree)


def shard_pi05_agent(agent, mesh, zero_actor_state=False):
    agent = jax.device_put(agent, NamedSharding(mesh, P()))
    if zero_actor_state:
        agent = agent.replace(
            actor_opt_state=jax.device_put(agent.actor_opt_state, zero_shardings(agent.actor_opt_state, mesh)),
            init_actor_params=jax.device_put(agent.init_actor_params, zero_shardings(agent.init_actor_params, mesh)))
    return agent.replace(mesh=mesh, zero_actor_state=zero_actor_state)
