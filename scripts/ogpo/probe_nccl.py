"""Collective-only probe: all-reduce over every visible GPU, prints per-step time. Hangs here = NCCL, not the model."""
import faulthandler
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

faulthandler.dump_traceback_later(60, exit=True, file=sys.stderr)
devs = jax.devices()
print(f"[probe] {len(devs)} devices: {[d.device_kind for d in devs][:1]}", flush=True)
mesh = Mesh(devs, axis_names=('data',))
x = jax.device_put(np.ones((len(devs) * 1024, 1024), np.float32), NamedSharding(mesh, P('data')))
f = jax.jit(lambda a: jnp.sum(a))  # sharded input -> cross-device all-reduce
for n in range(3):
    t0 = time.time()
    print(f"[probe] allreduce {n}: {float(f(x)):.0f} in {time.time() - t0:.2f}s", flush=True)
# 256 MiB all-reduce, the size class of gradient reductions
big = jax.device_put(np.ones((len(devs) * 8192, 8192), np.float32), NamedSharding(mesh, P('data')))
g = jax.jit(lambda a: jnp.sum(a, axis=0), out_shardings=NamedSharding(mesh, P()))
t0 = time.time()
jax.block_until_ready(g(big))
print(f"[probe] big reduce: {time.time() - t0:.2f}s", flush=True)
print("[probe] OK", flush=True)
