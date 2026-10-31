#!/usr/bin/env python3
"""Profile BC pretraining to identify performance bottlenecks.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/profile_bc.py

Measures:
    1. Dataset loading time
    2. sample_sequence() time (with/without images)
    3. Agent creation time
    4. bc_update() JIT compilation time (first call)
    5. bc_update() steady-state time per step
    6. Encoder-only forward pass time
    7. Image augmentation time
    8. block_until_ready() overhead
    9. Full BC loop comparison (with vs without block_until_ready every step)
"""

import os
import sys
import time
import functools

# Must set before JAX import
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ogpo.utils.datasets import Dataset
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.agents.ogpo import OGPOAgent
from ogpo.utils.main_helper import process_train_dataset


# ===========================================================================
# Config
# ===========================================================================
BATCH_SIZE = 256
HORIZON_LENGTH = 4
WARMUP_STEPS = 3          # JIT warmup
MEASURE_STEPS = 50        # Steps to average over
DISCOUNT = 0.98
SEED = 1


def timer(label):
    """Context manager that prints elapsed time."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            print(f"  {label}: {self.elapsed:.4f}s")
    return Timer()


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load_dataset():
    """Load the SIMPLER dataset."""
    print_header("1. Dataset Loading")

    from envs import simplerenv_utils
    with timer("Dataset load from disk"):
        env = simplerenv_utils.make_env("put_eggplant_in_basket_ogpo", seed=0, launch_server=False)
        dataset = simplerenv_utils.get_dataset(env, "put_eggplant_in_basket_ogpo")

    # Process dataset (MC returns, etc.)
    dataset_rng = np.random.default_rng(SEED + 20000)
    with timer("process_train_dataset"):
        dataset = process_train_dataset(
            dataset,
            "put_eggplant_in_basket_ogpo",
            dataset_proportion=1.0,
            sparse=False,
            numpy_rng=dataset_rng,
            compute_mc_returns=True,
            discount=DISCOUNT,
        )
    dataset.p_aug = 0.0  # disable augmentation for baseline

    N = len(dataset['observations'])
    print(f"  Dataset size: {N} transitions")
    if 'images' in dataset._dict:
        imgs = dataset['images']
        print(f"  Images: shape={imgs.shape}, dtype={imgs.dtype}")
        mem_mb = imgs.nbytes / 1024 / 1024
        print(f"  Image memory: {mem_mb:.1f} MB")
    else:
        print(f"  No images in dataset (state-only)")

    return dataset, env


def profile_sampling(dataset):
    """Profile sample_sequence with and without images."""
    print_header("2. Dataset Sampling (sample_sequence)")

    times_with_img = []
    for i in range(WARMUP_STEPS + MEASURE_STEPS):
        t0 = time.perf_counter()
        batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
        t1 = time.perf_counter()
        if i >= WARMUP_STEPS:
            times_with_img.append(t1 - t0)

    avg = np.mean(times_with_img)
    std = np.std(times_with_img)
    print(f"  sample_sequence (with images): {avg*1000:.2f} +/- {std*1000:.2f} ms")
    print(f"    Batch keys: {list(batch.keys())}")
    if 'images' in batch:
        print(f"    images shape: {batch['images'].shape}, dtype: {batch['images'].dtype}")

    # Profile augmentation separately (set p_aug on the dataset directly)
    if 'images' in dataset._dict:
        old_p_aug = dataset.p_aug
        dataset.p_aug = 1.0  # force augmentation every time

        # Warmup augmentation JIT
        _ = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)

        times_aug = []
        for i in range(WARMUP_STEPS + MEASURE_STEPS):
            t0 = time.perf_counter()
            batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
            t1 = time.perf_counter()
            if i >= WARMUP_STEPS:
                times_aug.append(t1 - t0)

        avg_aug = np.mean(times_aug)
        std_aug = np.std(times_aug)
        print(f"  sample_sequence (with augmentation): {avg_aug*1000:.2f} +/- {std_aug*1000:.2f} ms")
        print(f"    Augmentation overhead: {(avg_aug - avg)*1000:.2f} ms per batch")

        dataset.p_aug = old_p_aug  # restore

    return batch


def create_agent(dataset):
    """Create agent and profile creation time."""
    print_header("3. Agent Creation")

    # Load config like the shell script does — values must be parsed (not strings)
    cli_overrides = {
        "agent.horizon_length": HORIZON_LENGTH,
        "agent.discount": DISCOUNT,
        "agent.batch_size": BATCH_SIZE,
        "agent.actor_obs": "image",
        "agent.critic_obs": "image",
        "agent.vision_backbone": "impala_small",
        "agent.use_denoiser": False,
        "agent.error_correct_ode_to_sde": False,  # renamed; legacy FPO+denoiser path off
        "agent.flow_steps": 10,
        "agent.best_of_n": 8,
        "agent.num_qs": 4,
        "agent.ppo_batch_size": 256,
        "agent.actor_backbone": "mlp",
        "agent.critic_backbone": "mlp",
    }
    config = load_config("ogpo", cli_overrides)
    agent_config = config_to_agent_dict(config)

    # Debug: check types of critical config values
    for k in ['horizon_length', 'discount', 'batch_size', 'flow_steps', 'best_of_n', 'num_qs',
              'action_chunking', 'use_denoiser']:
        v = agent_config.get(k)
        print(f"    {k} = {v!r} (type: {type(v).__name__})")

    # Q bounds
    agent_config["q_min"] = -1.0
    agent_config["q_max"] = 1.0 / (1.0 - DISCOUNT)

    example_batch = dataset.sample(())
    ex_observations = example_batch["observations"]
    ex_actions = example_batch["actions"]
    ex_images = example_batch.get("images")

    with timer("OGPOAgent.create()"):
        agent = OGPOAgent.create(
            SEED, ex_observations, ex_actions, agent_config,
            adv_clip_min=None, ex_images=ex_images,
        )
    jax.block_until_ready(agent)

    # Count parameters
    actor_params = jax.tree_util.tree_leaves(agent.actor_network.params)
    critic_params = jax.tree_util.tree_leaves(agent.critic_network.params)
    n_actor = sum(p.size for p in actor_params)
    n_critic = sum(p.size for p in critic_params)
    print(f"  Actor params: {n_actor:,}")
    print(f"  Critic params: {n_critic:,}")

    return agent


def profile_bc_update(agent, dataset):
    """Profile bc_update step by step."""
    print_header("4. bc_update() Timing")

    batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)

    # --- JIT compilation (first call) ---
    with timer("bc_update (JIT compile, 1st call)"):
        agent, info = agent.bc_update(batch)
        jax.block_until_ready(agent)

    # --- Warmup ---
    for _ in range(WARMUP_STEPS):
        batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
        agent, info = agent.bc_update(batch)
        jax.block_until_ready(agent)

    # --- Steady state: total time (sample + update + sync) ---
    times_total = []
    times_update = []
    times_sample = []
    times_sync = []

    for _ in range(MEASURE_STEPS):
        # Sample
        t0 = time.perf_counter()
        batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
        t1 = time.perf_counter()

        # Update (async — JAX returns immediately)
        agent, info = agent.bc_update(batch)
        t2 = time.perf_counter()

        # Sync
        jax.block_until_ready(agent)
        t3 = time.perf_counter()

        times_sample.append(t1 - t0)
        times_update.append(t2 - t1)
        times_sync.append(t3 - t2)
        times_total.append(t3 - t0)

    print(f"\n  Steady-state breakdown ({MEASURE_STEPS} steps):")
    print(f"    sample_sequence:     {np.mean(times_sample)*1000:7.2f} +/- {np.std(times_sample)*1000:.2f} ms")
    print(f"    bc_update (launch):  {np.mean(times_update)*1000:7.2f} +/- {np.std(times_update)*1000:.2f} ms")
    print(f"    block_until_ready:   {np.mean(times_sync)*1000:7.2f} +/- {np.std(times_sync)*1000:.2f} ms")
    print(f"    TOTAL per step:      {np.mean(times_total)*1000:7.2f} +/- {np.std(times_total)*1000:.2f} ms")
    print(f"    Throughput:          {1.0/np.mean(times_total):.1f} steps/sec")

    # --- Without block_until_ready every step ---
    print(f"\n  Without block_until_ready every step ({MEASURE_STEPS} steps):")
    t_start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
        agent, info = agent.bc_update(batch)
    jax.block_until_ready(agent)  # sync once at end
    t_end = time.perf_counter()
    avg_async = (t_end - t_start) / MEASURE_STEPS
    print(f"    Avg per step:        {avg_async*1000:7.2f} ms")
    print(f"    Throughput:          {1.0/avg_async:.1f} steps/sec")
    print(f"    Speedup:             {np.mean(times_total)/avg_async:.2f}x")

    # Print loss info
    print(f"\n  BC loss info keys: {list(info.keys())}")
    for k in ['actor/bc_flow_loss', 'bc_flow_loss', 'actor_loss']:
        if k in info:
            print(f"    {k}: {float(info[k]):.6f}")

    return agent


def profile_encoder_only(agent, dataset):
    """Profile just the image encoder forward pass."""
    print_header("5. Encoder Forward Pass (ImpalaEncoder)")

    batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
    images = batch.get('images')
    observations = batch['observations']

    if images is None:
        print("  No images in batch, skipping encoder profiling")
        return

    images_jax = jnp.array(images)
    obs_jax = jnp.array(observations)

    # Use actor encode method
    @jax.jit
    def encode_fn(agent, images, obs):
        return agent._encode_for_actor(obs, images=images)

    # JIT compile
    with timer("Encoder JIT compile"):
        encoded = encode_fn(agent, images_jax, obs_jax)
        jax.block_until_ready(encoded)

    print(f"  Encoded shape: {encoded.shape}")

    # Steady state
    times = []
    for _ in range(MEASURE_STEPS):
        t0 = time.perf_counter()
        encoded = encode_fn(agent, images_jax, obs_jax)
        jax.block_until_ready(encoded)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = np.mean(times)
    print(f"  Encoder forward: {avg*1000:.2f} +/- {np.std(times)*1000:.2f} ms ({BATCH_SIZE} images)")
    print(f"  Per image: {avg*1000/BATCH_SIZE:.4f} ms")


def profile_augmentation(dataset):
    """Profile image augmentation separately."""
    print_header("6. Image Augmentation")

    if 'images' not in dataset._dict:
        print("  No images, skipping")
        return

    from ogpo.utils.augmentations import batched_random_crop, color_transform

    # Get a batch of images
    batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
    images = batch['images']  # (B, H, W, C) uint8
    print(f"  Image batch: shape={images.shape}, dtype={images.dtype}")

    # Profile random crop
    rng = jax.random.PRNGKey(0)
    padding = 3
    crop_froms = np.random.randint(0, 2 * padding + 1, (BATCH_SIZE, 2))
    crop_froms = np.concatenate([crop_froms, np.zeros((BATCH_SIZE, 1), dtype=np.int64)], axis=1)

    # Warmup
    _ = batched_random_crop(images, crop_froms, padding)
    jax.block_until_ready(_)

    times_crop = []
    for _ in range(MEASURE_STEPS):
        crop_froms = np.random.randint(0, 2 * padding + 1, (BATCH_SIZE, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((BATCH_SIZE, 1), dtype=np.int64)], axis=1)
        t0 = time.perf_counter()
        cropped = batched_random_crop(images, crop_froms, padding)
        jax.block_until_ready(cropped)
        t1 = time.perf_counter()
        times_crop.append(t1 - t0)

    print(f"  Random crop: {np.mean(times_crop)*1000:.2f} +/- {np.std(times_crop)*1000:.2f} ms")

    # Profile color transform
    images_float = images / 255.0
    rng, aug_rng = jax.random.split(rng)
    _ = color_transform(aug_rng, images_float)
    jax.block_until_ready(_)

    times_color = []
    for i in range(MEASURE_STEPS):
        rng, aug_rng = jax.random.split(rng)
        t0 = time.perf_counter()
        augmented = color_transform(aug_rng, images_float)
        jax.block_until_ready(augmented)
        t1 = time.perf_counter()
        times_color.append(t1 - t0)

    print(f"  Color transform: {np.mean(times_color)*1000:.2f} +/- {np.std(times_color)*1000:.2f} ms")

    # Profile full augmentation (crop + color)
    times_full = []
    for i in range(MEASURE_STEPS):
        crop_froms = np.random.randint(0, 2 * padding + 1, (BATCH_SIZE, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((BATCH_SIZE, 1), dtype=np.int64)], axis=1)
        rng, aug_rng = jax.random.split(rng)
        t0 = time.perf_counter()
        cropped = np.array(batched_random_crop(images, crop_froms, padding))
        augmented = (color_transform(aug_rng, cropped / 255.0) * 255.0).astype(jnp.uint8)
        jax.block_until_ready(augmented)
        t1 = time.perf_counter()
        times_full.append(t1 - t0)

    print(f"  Full augmentation: {np.mean(times_full)*1000:.2f} +/- {np.std(times_full)*1000:.2f} ms")


def profile_data_transfer(dataset):
    """Profile CPU→GPU data transfer for images."""
    print_header("7. CPU→GPU Data Transfer")

    batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)

    if 'images' not in batch:
        print("  No images, skipping")
        return

    images_np = batch['images']  # (B, H, W, C) uint8
    obs_np = batch['observations']  # (B, obs_dim)
    actions_np = batch['actions']  # (B, T, act_dim)

    # Image transfer
    times_img = []
    for _ in range(MEASURE_STEPS):
        t0 = time.perf_counter()
        img_jax = jax.device_put(images_np)
        jax.block_until_ready(img_jax)
        t1 = time.perf_counter()
        times_img.append(t1 - t0)

    mem_mb = images_np.nbytes / 1024 / 1024
    print(f"  Image transfer ({images_np.shape}, {images_np.dtype}): "
          f"{np.mean(times_img)*1000:.2f} ms ({mem_mb:.1f} MB)")

    # Full batch transfer
    times_full = []
    for _ in range(MEASURE_STEPS):
        t0 = time.perf_counter()
        batch_jax = jax.device_put(batch)
        jax.block_until_ready(batch_jax)
        t1 = time.perf_counter()
        times_full.append(t1 - t0)

    total_mb = sum(v.nbytes for v in batch.values() if hasattr(v, 'nbytes')) / 1024 / 1024
    print(f"  Full batch transfer: {np.mean(times_full)*1000:.2f} ms ({total_mb:.1f} MB)")


def profile_sync_frequency(agent, dataset):
    """Compare different block_until_ready frequencies."""
    print_header("8. Sync Frequency Comparison")

    N = 100  # total steps

    # Warmup
    for _ in range(WARMUP_STEPS):
        batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
        agent, _ = agent.bc_update(batch)
    jax.block_until_ready(agent)

    for sync_every in [1, 5, 10, 25, 50, 100]:
        t_start = time.perf_counter()
        for i in range(1, N + 1):
            batch = dataset.sample_sequence(BATCH_SIZE, sequence_length=HORIZON_LENGTH, discount=DISCOUNT)
            agent, _ = agent.bc_update(batch)
            if i % sync_every == 0:
                jax.block_until_ready(agent)
        jax.block_until_ready(agent)
        t_end = time.perf_counter()
        avg = (t_end - t_start) / N
        print(f"  sync_every={sync_every:3d}: {avg*1000:7.2f} ms/step  ({1/avg:.1f} steps/sec)")


def main():
    print("=" * 60)
    print("  OGPO BC Pretraining Profiler")
    print("=" * 60)
    print(f"  Device: {jax.devices()}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Horizon length: {HORIZON_LENGTH}")
    print(f"  Measure steps: {MEASURE_STEPS}")

    # 1. Load dataset
    dataset, env = load_dataset()

    # 2. Profile sampling
    profile_sampling(dataset)

    # 3. Create agent
    agent = create_agent(dataset)

    # 4. Profile bc_update
    agent = profile_bc_update(agent, dataset)

    # 5. Profile encoder
    profile_encoder_only(agent, dataset)

    # 6. Profile augmentation
    profile_augmentation(dataset)

    # 7. Profile data transfer
    profile_data_transfer(dataset)

    # 8. Profile sync frequency
    profile_sync_frequency(agent, dataset)

    print_header("SUMMARY")
    print("  Done! Review timings above to identify bottlenecks.")
    print("  Key things to look for:")
    print("    - If block_until_ready dominates → GPU compute is the bottleneck")
    print("    - If sample_sequence dominates → CPU data loading is the bottleneck")
    print("    - If augmentation is expensive → consider disabling or GPU-side aug")
    print("    - If sync_every speedup is large → async pipelining would help")


if __name__ == "__main__":
    main()
