"""Classifier-free guidance sampling for pi0/pi05.

Mirrors the deterministic (Euler) branch of Pi0.sample_actions, adding an
unconditional batch that reuses the same prefix with the prompt tokens masked
out of attention, so the image tower runs once and the guided velocity costs a
single batch-doubled forward per denoising step.
"""

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at


def sample_actions_cfg(
    model,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    num_steps: int = 10,
    cfg_scale: float = 1.0,
    return_prefix_rep: bool = False,
):
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    use_cfg = cfg_scale != 1.0 and observation.tokenized_prompt is not None
    if use_cfg:
        num_lang_tokens = observation.tokenized_prompt.shape[1]
        uncond_prefix_mask = prefix_mask.at[:, -num_lang_tokens:].set(False)
        prefix_tokens = jnp.concatenate([prefix_tokens, prefix_tokens], axis=0)
        prefix_mask = jnp.concatenate([prefix_mask, uncond_prefix_mask], axis=0)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_rep, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    def compute_v_t(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        if use_cfg:
            suffix_tokens = jnp.concatenate([suffix_tokens, suffix_tokens], axis=0)
            suffix_mask = jnp.concatenate([suffix_mask, suffix_mask], axis=0)
            if adarms_cond is not None:
                adarms_cond = jnp.concatenate([adarms_cond, adarms_cond], axis=0)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        if use_cfg:
            v_cond, v_uncond = jnp.split(v_t, 2, axis=0)
            v_t = v_uncond + cfg_scale * (v_cond - v_uncond)
        return v_t

    def step(carry, _):
        x_t, time = carry
        x_next = x_t + dt * compute_v_t(x_t, time)
        return (x_next, time + dt), None

    initial_time = jnp.ones((batch_size,), dtype=noise.dtype)
    (x_0, _), _ = jax.lax.scan(step, (noise, initial_time), xs=None, length=num_steps)
    if return_prefix_rep:
        return x_0, prefix_rep[:batch_size]
    return x_0
