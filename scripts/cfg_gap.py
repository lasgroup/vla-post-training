"""Compare the conditional and unconditional velocity fields of a CFG checkpoint.

Guidance extrapolates along `v_cond - v_uncond`. If the prompt adds little on top
of the images -- plausible on LIBERO, where each task owns a scene and the cameras
are fixed -- that difference is ~0 and no cfg_scale can do anything but amplify
sampling noise.

Inference only: loads the `params` item (which is the EMA weights, see
checkpoints._split_params) and reads real observations straight out of a replay
shard, which already stores them in Observation format. No train_state, no
optimizer moments, no env.

`cfg_scale=0.0` turns on the CFG path (`use_cfg = cfg_scale != 1.0`) and returns
`v_uncond + 0*(v_cond - v_uncond)`, i.e. the pure unconditional velocity. Running
it against `cfg_scale=1.0` from the same noise gives both branches; at the first
denoising step they share x_t exactly, so that step is a true same-input contrast
of the two velocity fields.

    python scripts/cfg_gap.py --checkpoint <ckpt>/5000 --shard <...>.h5
"""

import argparse
import pathlib

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model

import src.training.config as _config

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def load_rows(shard, idx):
    """Observation-format batch for the given shard row indices.

    h5py fancy indexing wants strictly increasing indices, so read the unique
    rows and re-expand to the requested (possibly repeating) order in numpy.
    """
    uniq = sorted(set(idx))
    pos = np.array([uniq.index(i) for i in idx])
    with h5py.File(shard, "r") as f:
        obs = f["observations"]
        take = lambda d: np.asarray(d[uniq])[pos]
        return {
            "image": {k: take(obs[f"image/{k}"]) for k in IMAGE_KEYS},
            "image_mask": {k: take(obs[f"image_mask/{k}"]) for k in IMAGE_KEYS},
            "state": take(obs["state"]),
            "tokenized_prompt": take(obs["tokenized_prompt"]),
            "tokenized_prompt_mask": take(obs["tokenized_prompt_mask"]),
        }


def distinct_prompt_rows(shard, limit):
    """One row index per distinct prompt, so a rotation gives a real task swap."""
    with h5py.File(shard, "r") as f:
        prompts = np.asarray(f["observations/tokenized_prompt"][:limit])
        masks = np.asarray(f["observations/tokenized_prompt_mask"][:limit])
    seen, rows = {}, []
    for i, (p, m) in enumerate(zip(prompts, masks)):
        key = p[m].tobytes()
        if key not in seen:
            seen[key] = i
            rows.append(i)
    return rows


def velocities(model, obs, noise, rng, cfg_scale, num_steps):
    """Per-step (x_t, v_t) along the sampling trajectory at this guidance scale."""
    _, info = model.sample_actions(
        rng, obs, noise=noise, num_steps=num_steps, cfg_scale=cfg_scale, return_info_dict=True
    )
    dt = -1.0 / num_steps
    v = (np.asarray(info["x_next"]) - np.asarray(info["x"])) / dt
    return np.asarray(info["x"]), v


def rel(a, b):
    """Per-sample ||a - b|| / ||a||, flattened over the action chunk."""
    num = np.linalg.norm((a - b).reshape(a.shape[0], -1), axis=1)
    den = np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)
    return num / np.maximum(den, 1e-8)


def cosine(a, b):
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return np.sum(a * b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="checkpoint step dir (contains params/)")
    p.add_argument("--shard", required=True, help="replay shard .h5 to draw observations from")
    p.add_argument("--config", default="pi05_libero_online_filtered_sft")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scan-rows", type=int, default=4000, help="rows to scan for distinct prompts")
    args = p.parse_args()

    rows = distinct_prompt_rows(args.shard, args.scan_rows)
    print(f"{len(rows)} distinct prompts in the first {args.scan_rows} rows: {rows}")
    if len(rows) < 2:
        raise SystemExit("need at least 2 distinct prompts for the swap test")
    rows = (rows * ((args.batch // len(rows)) + 1))[: args.batch]

    obs_dict = load_rows(args.shard, rows)
    to_jax = lambda d: jax.tree.map(jnp.asarray, d)
    obs = _model.Observation.from_dict(to_jax(obs_dict))

    # Rotate prompts by one so every sample gets another task's instruction.
    swapped_dict = dict(obs_dict)
    roll = lambda x: np.roll(x, 1, axis=0)
    swapped_dict["tokenized_prompt"] = roll(obs_dict["tokenized_prompt"])
    swapped_dict["tokenized_prompt_mask"] = roll(obs_dict["tokenized_prompt_mask"])
    obs_swap = _model.Observation.from_dict(to_jax(swapped_dict))

    params = _model.restore_params(pathlib.Path(args.checkpoint) / "params")
    model = _config.get_config(args.config).model.load(params)
    model.eval()

    b = args.batch
    rng = jax.random.key(args.seed)
    rng, noise_rng, alt_rng = jax.random.split(rng, 3)
    shape = (b, model.action_horizon, model.action_dim)
    noise = jax.random.normal(noise_rng, shape)
    noise_alt = jax.random.normal(alt_rng, shape)
    k = jax.random.split(rng, 4)

    x_c, v_cond = velocities(model, obs, noise, k[0], 1.0, args.num_steps)
    _, v_uncond = velocities(model, obs, noise, k[1], 0.0, args.num_steps)
    _, v_swap = velocities(model, obs_swap, noise, k[2], 1.0, args.num_steps)
    _, v_alt = velocities(model, obs, noise_alt, k[3], 1.0, args.num_steps)

    # Step 0 is the honest comparison: every run starts from the same x_t = noise,
    # so the velocities are evaluated at identical inputs. Later steps drift apart.
    print(f"\n=== step 0 (t=1, identical x_t across runs), batch {b} ===")
    print(f"  ||v_cond - v_uncond|| / ||v_cond||   {rel(v_cond[0], v_uncond[0]).mean():.4f}")
    print(f"  ||v_cond - v_swap||   / ||v_cond||   {rel(v_cond[0], v_swap[0]).mean():.4f}")
    print(f"  cos(v_cond, v_uncond)                {cosine(v_cond[0], v_uncond[0]).mean():.4f}")
    print(f"  cos(v_cond, v_swap)                  {cosine(v_cond[0], v_swap[0]).mean():.4f}")

    print("\n=== per denoising step: relative gap vs v_cond ===")
    print(f"{'step':>4} {'t':>6} {'uncond':>8} {'swap':>8} {'noise*':>8}")
    for s in range(args.num_steps):
        t = 1.0 + s * (-1.0 / args.num_steps)
        print(
            f"{s:>4} {t:>6.2f} {rel(v_cond[s], v_uncond[s]).mean():>8.4f} "
            f"{rel(v_cond[s], v_swap[s]).mean():>8.4f} {rel(v_cond[s], v_alt[s]).mean():>8.4f}"
        )
    print("* noise = same prompt, different sampling noise -- the reference scale.")
    print("  Only step 0 shares x_t across runs; later rows also include trajectory drift.")

    u, sw, n = (rel(v_cond[0], v)[None].mean() for v in (v_uncond[0], v_swap[0], v_alt[0]))
    print(f"\nuncond/noise at step 0: {u / max(n, 1e-8):.3f}   swap/noise: {sw / max(n, 1e-8):.3f}")
    if u < n:
        print(
            "Conditioning moves the velocity less than resampling noise does: guidance "
            "has almost nothing to amplify, which is consistent with the flat sweep."
        )


if __name__ == "__main__":
    main()
