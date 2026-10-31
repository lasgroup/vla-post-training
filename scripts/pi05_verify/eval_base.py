"""Success rate of the unchanged pi05_libero checkpoint in OGPO's libero_pi05 env.

ODE sampling (10 Euler steps), 10-step chunks, first `replan` actions executed.
Episode i resets with seed=i, so the init states are fixed across runs.
"""

import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from envs.libero_pi05_utils import make_env
from ogpo.networks.pi05 import transforms as T
from ogpo.networks.pi05.pi05 import Pi05, load_pi05_params, merge_params, split_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--env", default="libero_pi05-libero_90-59")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--replan", type=int, default=5)
    p.add_argument("--precision", choices=["bf16", "vla"], default="vla")
    args = p.parse_args()
    ckpt = pathlib.Path(args.ckpt)

    env = make_env(args.env, seed=0)
    norm_stats = T.load_norm_stats(ckpt / "assets")
    tokens, token_mask = T.PromptTokenizer(args.tokenizer).tokenize(env.prompt)
    tokens, token_mask = tokens[None], token_mask[None]

    model = Pi05()
    params = load_pi05_params(ckpt / "params")
    if args.precision == "bf16":
        params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params)
    else:
        params = merge_params(*split_params(params))

    @jax.jit
    def act(images, image_masks, noise):
        return model.apply({"params": params}, images, image_masks, tokens, token_mask, noise, method=model.sample_ode)

    rng = jax.random.PRNGKey(0)
    successes, lengths = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        t0, done, info = time.time(), False, {}
        while not done:
            images, image_masks = T.to_model_images(
                T.resize_images(obs["image"][None]), T.resize_images(obs["wrist_image"][None])
            )
            rng, key = jax.random.split(rng)
            x0 = act(images, image_masks, jax.random.normal(key, (1, 10, 32)))
            chunk = T.actions_to_env(np.asarray(x0[0]), norm_stats)
            for a in chunk[: args.replan]:
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
                if done:
                    break
        successes.append(info["success"])
        lengths.append(env.t)
        print(f"ep {ep:3d}  success={info['success']}  len={env.t}  ({time.time() - t0:.0f}s)", flush=True)

    sr = float(np.mean(successes))
    succ_len = np.mean([l for s, l in zip(successes, lengths) if s]) if any(successes) else float("nan")
    print(f"== {args.env} precision={args.precision}: SR {sr:.3f} ({sum(successes)}/{len(successes)}), mean success length {succ_len:.1f}")


if __name__ == "__main__":
    main()
