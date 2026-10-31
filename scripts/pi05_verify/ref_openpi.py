"""Step 1 (babel venv, openpi): reference pi05_libero outputs on fixed inputs.

Mirrors vla-post-training's rollout path: resize_with_pad(224) -> uint8 ->
policy input transforms -> model.sample_actions(noise, noise_level=0) -> output transforms.
"""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi_client import image_tools

PROMPT = "pick up the tomato sauce and put it in the tray"  # libero_90 task 59


def smooth_images(rng, n, res=256):
    low = rng.integers(0, 256, size=(n, 8, 8, 3)).astype(np.float32)
    up = np.asarray(jax.image.resize(low, (n, res, res, 3), "bilinear"))
    return np.clip(up + rng.normal(0, 8, up.shape), 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    base = smooth_images(rng, args.n)
    wrist = smooth_images(rng, args.n)
    state = rng.normal(0, 0.3, (args.n, 8)).astype(np.float32)
    noise = rng.normal(size=(args.n, 10, 32)).astype(np.float32)

    policy = policy_config.create_trained_policy(_config.get_config("pi05_libero"), args.ckpt)
    to224 = lambda x: image_tools.convert_to_uint8(image_tools.resize_with_pad(x, 224, 224))

    out = {k: [] for k in ["img_base", "img_wrist", "state_model", "tokens", "token_mask",
                           "actions_model", "actions_env", "prefix_mean", "prefix_valid_mean"]}
    for i in range(args.n):
        obs = {
            "observation/image": to224(base[i]),
            "observation/wrist_image": to224(wrist[i]),
            "observation/state": state[i],
            "prompt": PROMPT,
        }
        inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        observation = _model.Observation.from_dict(batched)
        x0 = policy._sample_actions(jax.random.key(0), observation, noise=jnp.asarray(noise[i:i + 1]), num_steps=10)
        prefix, _ = policy._get_prefix_rep(observation)
        env = policy._output_transform({"state": inputs["state"], "actions": np.asarray(x0[0])})["actions"]

        valid = np.concatenate([np.ones(256 * 2, bool), np.zeros(256, bool), inputs["tokenized_prompt_mask"]])
        out["img_base"].append(inputs["image"]["base_0_rgb"])
        out["img_wrist"].append(inputs["image"]["left_wrist_0_rgb"])
        out["state_model"].append(inputs["state"])
        out["tokens"].append(inputs["tokenized_prompt"])
        out["token_mask"].append(inputs["tokenized_prompt_mask"])
        out["actions_model"].append(np.asarray(x0[0], np.float32))
        out["actions_env"].append(np.asarray(env, np.float32))
        out["prefix_mean"].append(np.asarray(prefix[0], np.float32).mean(0))
        out["prefix_valid_mean"].append(np.asarray(prefix[0], np.float32)[valid].mean(0))
        print(f"[ref] sample {i}: env action[0] = {np.round(env[0], 4)}")

    tokenizer_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    np.savez(
        args.out,
        raw_base=base, raw_wrist=wrist, raw_state=state, noise=noise, prompt=PROMPT,
        tokenizer_path=str(tokenizer_path),
        **{k: np.stack(v) for k, v in out.items()},
    )
    print(f"[ref] wrote {pathlib.Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
