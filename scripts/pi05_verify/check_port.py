"""Step 2 (OGPO venv): compare ogpo/networks/pi05 against the openpi reference npz."""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from ogpo.networks.pi05 import transforms as T
from ogpo.networks.pi05.pi05 import Pi05, load_pi05_params, merge_params, split_params


def report(name, ours, ref, tol):
    err = float(np.abs(np.asarray(ours, np.float64) - np.asarray(ref, np.float64)).max())
    scale = float(np.abs(np.asarray(ref, np.float64)).mean()) + 1e-12
    ok = err <= tol
    print(f"  {'OK ' if ok else 'BAD'} {name:<34} max|diff|={err:.3e}  mean|ref|={scale:.3e}  tol={tol:g}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="pi05_libero checkpoint dir (with params/ and assets/)")
    p.add_argument("--ref", required=True)
    args = p.parse_args()
    ref = dict(np.load(args.ref))
    ckpt = pathlib.Path(args.ckpt)
    ok = True

    print("== input transforms")
    base = T.resize_images(ref["raw_base"])
    wrist = T.resize_images(ref["raw_wrist"])
    ok &= report("base image (uint8)", base, ref["img_base"], 0)
    ok &= report("wrist image (uint8)", wrist, ref["img_wrist"], 0)
    norm_stats = T.load_norm_stats(ckpt / "assets")
    ok &= report("state (normalised, padded)", T.state_to_model(ref["raw_state"], norm_stats), ref["state_model"], 1e-5)
    tokens, token_mask = T.PromptTokenizer(str(ref["tokenizer_path"])).tokenize(str(ref["prompt"]))
    ok &= report("tokens", tokens[None].repeat(len(base), 0), ref["tokens"], 0)
    ok &= report("token mask", token_mask[None].repeat(len(base), 0), ref["token_mask"], 0)

    images, image_masks = T.to_model_images(base, wrist)
    n = len(base)
    tokens = np.repeat(tokens[None], n, 0)
    token_mask = np.repeat(token_mask[None], n, 0)
    noise = jnp.asarray(ref["noise"])

    model = Pi05()
    params = load_pi05_params(ckpt / "params")
    sample = jax.jit(lambda prm: model.apply({"params": prm}, images, image_masks, tokens, token_mask, noise, method=model.sample_ode))
    prefix = jax.jit(lambda prm: model.apply({"params": prm}, images, image_masks, tokens, token_mask, method=model.prefix))

    valid = np.concatenate([np.ones(256 * 2, bool), np.zeros(256, bool), token_mask[0]])
    for tag, prm, tol in [
        ("all-bf16 (== openpi inference)", jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params), 5e-2),
        ("frozen bf16 / trainable fp32 (VLA)", merge_params(*split_params(params)), 1.5e-1),
    ]:
        print(f"== model: {tag}")
        x0 = np.asarray(sample(prm), np.float32)
        prefix_out = np.asarray(prefix(prm)[0], np.float32)
        ok &= report("actions (normalised, 7 dims)", x0[..., :7], ref["actions_model"][..., :7], tol)
        ok &= report("actions (env, unnormalised)", T.actions_to_env(x0, norm_stats), ref["actions_env"], tol)
        # Gemma hidden states have O(30) outlier channels, so compare relative L2 error.
        for name, ours, r in [("prefix mean (all tokens)", prefix_out.mean(1), ref["prefix_mean"]),
                              ("prefix mean (valid tokens)", prefix_out[:, valid].mean(1), ref["prefix_valid_mean"])]:
            rel = float(np.linalg.norm(ours - r) / np.linalg.norm(r))
            print(f"  {'OK ' if rel <= 2e-2 else 'BAD'} {name:<34} ||diff||/||ref||={rel:.3e}  tol=0.02")
            ok &= rel <= 2e-2

    print("PASS" if ok else "FAIL (see BAD lines)")


if __name__ == "__main__":
    main()
