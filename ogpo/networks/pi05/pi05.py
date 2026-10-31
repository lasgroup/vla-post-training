"""pi0.5 (PaliGemma + Gemma-300M action expert) as a flax.linen module.

Linen port of openpi/models/pi0.py (pi05=True branch). Numerics follow openpi:
flow time runs t=1 (noise) -> t=0 (data), velocity target is `noise - actions`.

Inputs are already preprocessed:
  images:       dict name -> (B, 224, 224, 3) float in [-1, 1], ordered as IMAGE_KEYS
  image_masks:  dict name -> (B,) bool
  tokens:       (B, L) int32 tokenized prompt, token_mask: (B, L) bool
"""

import re

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from ogpo.networks.encoders.paligemma import SigLIPViT
from ogpo.networks.pi05 import gemma as _gemma

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# openpi Pi0Config.get_freeze_filter / VLA _make_ogpo_freeze_filter:
# freeze PaliGemma (llm expert 0 + SigLIP), train the action expert and heads.
_TRAINABLE_LLM = re.compile(r".*llm.*_1.*")


def make_attn_mask(input_mask, mask_ar):
    """big_vision prefix-LM mask: token i attends to j iff cumsum(ar)[j] <= cumsum(ar)[i]."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def posemb_sincos(pos, embedding_dim, min_period, max_period):
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi05(nn.Module):
    action_dim: int = 32
    action_horizon: int = 10
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    def setup(self):
        pg_config = _gemma.get_config(self.paligemma_variant)
        ae_config = _gemma.get_config(self.action_expert_variant)
        self.llm = _gemma.Module(
            configs=[pg_config, ae_config],
            embed_dtype=self.dtype,
            adarms=True,
        )
        self.img = SigLIPViT(num_classes=pg_config.width, dtype_mm=self.dtype)
        self.action_in_proj = nn.Dense(ae_config.width)
        self.time_mlp_in = nn.Dense(ae_config.width)
        self.time_mlp_out = nn.Dense(ae_config.width)
        self.action_out_proj = nn.Dense(self.action_dim)
        self._ae_width = ae_config.width

    def embed_prefix(self, images, image_masks, tokens, token_mask):
        unknown = sorted(set(images) - set(IMAGE_KEYS))
        if unknown or IMAGE_KEYS[0] not in images:
            raise ValueError(f"embed_prefix: images keys {sorted(images)}; expected a subset of {IMAGE_KEYS} with {IMAGE_KEYS[0]}")
        input_mask, ar_mask, out = [], [], []
        for name in IMAGE_KEYS:
            if name not in images:  # camera slot left out (OGPOPi05Agent._prefix(kv_only=True) drops the always-masked one)
                continue
            image_tokens = self.img(images[name], train=False)
            out.append(image_tokens)
            input_mask.append(einops.repeat(image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]
        out.append(self.llm.embed(tokens))
        input_mask.append(token_mask)
        ar_mask += [False] * tokens.shape[1]
        return (
            jnp.concatenate(out, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.array(ar_mask),
        )

    def embed_suffix(self, noisy_actions, timestep):
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self._ae_width, min_period=4e-3, max_period=4.0)
        time_emb = nn.swish(self.time_mlp_in(time_emb))
        time_emb = nn.swish(self.time_mlp_out(time_emb))
        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        return action_tokens, input_mask, ar_mask, time_emb

    def prefix(self, images, image_masks, tokens, token_mask):
        """Frozen VLM pass. Returns (prefix_out (B,S,2048), kv_cache, prefix_mask)."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(images, image_masks, tokens, token_mask)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        return prefix_out, kv_cache, prefix_mask

    def velocity(self, kv_cache, prefix_mask, x_t, time):
        """Action-expert velocity on a cached prefix (openpi `compute_v_t`).

        x_t: (B, H, action_dim), time: (B,) in openpi convention (1 = noise).
        """
        batch_size = x_t.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(x_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None and suffix_out.shape[0] == batch_size
        return self.action_out_proj(suffix_out[:, -self.action_horizon:])

    def sample_ode(self, images, image_masks, tokens, token_mask, noise, num_steps=10):
        """Deterministic Euler sampling, identical to openpi `sample_actions(noise_level=0)`."""
        _, kv_cache, prefix_mask = self.prefix(images, image_masks, tokens, token_mask)
        dt = -1.0 / num_steps
        batch_size = noise.shape[0]

        def step(carry, _):
            x_t, time = carry
            v_t = self.velocity(kv_cache, prefix_mask, x_t, jnp.broadcast_to(time, batch_size))
            return (x_t + dt * v_t, time + dt), None

        (x_0, _), _ = jax.lax.scan(step, (noise, jnp.ones((batch_size,), noise.dtype)), None, length=num_steps)
        return x_0

    def __call__(self, images, image_masks, tokens, token_mask, x_t, time):
        """Joint prefix+suffix pass (openpi `compute_loss` forward). Also initialises all params."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(images, image_masks, tokens, token_mask)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon:])


# ---------------------------------------------------------------------------
# Params: checkpoint loading and frozen / trainable split
# ---------------------------------------------------------------------------

def load_pi05_params(checkpoint_path):
    """Restore a pi05 orbax checkpoint (e.g. pi05_libero/params) onto host.

    Returns a float32 param tree keyed like `Pi05`: {llm, img, action_in_proj,
    time_mlp_in, time_mlp_out, action_out_proj}.
    """
    import orbax.checkpoint as ocp

    ckptr = ocp.PyTreeCheckpointer()
    metadata = ckptr.metadata(checkpoint_path)
    for attr in ("item_metadata", "metadata", "tree"):
        inner = getattr(metadata, attr, None)
        if inner is not None:
            metadata = inner
            break

    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])

    def _is_array_leaf(x):
        return hasattr(x, "shape") and hasattr(x, "dtype")

    target = jax.tree.map(
        lambda l: jax.ShapeDtypeStruct(l.shape, l.dtype, sharding=cpu_sharding) if _is_array_leaf(l) else l,
        metadata,
        is_leaf=_is_array_leaf,
    )
    restore_args = ocp.checkpoint_utils.construct_restore_args(target)
    params = ckptr.restore(checkpoint_path, restore_args=restore_args)["params"]

    flat = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat):  # NNX checkpoints
        flat = {kp[:-1]: v for kp, v in flat.items()}
    # PaliGemma/{llm,img} -> {llm,img}
    flat = {(kp[1:] if kp[0] == "PaliGemma" else kp): np.asarray(v, dtype=np.float32) for kp, v in flat.items()}
    return traverse_util.unflatten_dict(flat)


def _is_trainable(path):
    name = "/".join(path)
    if name.startswith("img/"):
        return False
    if name.startswith("llm/"):
        return bool(_TRAINABLE_LLM.match(name))
    return True


def split_params(params, frozen_dtype=jnp.bfloat16):
    """-> (frozen, trainable). Frozen leaves are cast to bf16 as in VLA's init_train_state."""
    flat = traverse_util.flatten_dict(params)
    trainable = {kp: v for kp, v in flat.items() if _is_trainable(kp)}
    frozen = {kp: jnp.asarray(v, frozen_dtype) for kp, v in flat.items() if not _is_trainable(kp)}
    return traverse_util.unflatten_dict(frozen), traverse_util.unflatten_dict(trainable)


def merge_params(frozen, trainable):
    return traverse_util.unflatten_dict({
        **traverse_util.flatten_dict(frozen),
        **traverse_util.flatten_dict(trainable),
    })
