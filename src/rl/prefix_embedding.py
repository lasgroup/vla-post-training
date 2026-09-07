from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


PREFIX_EMBEDDING_NAME = "prefix_embedding"

# collect.prefix_pooling values. The critic input is the pooled PaliGemma prefix:
#   "mean"            one mean over all prefix tokens (image + language)  -> (E,)
#   "image_text_mean" image tokens and language tokens mean-pooled separately,
#                     concatenated                                        -> (2E,)
PREFIX_POOLING_MODES = ("mean", "image_text_mean")


def pooled_prefix_dim(embed_dim: int, pooling: str) -> int:
    if pooling == "image_text_mean":
        return 2 * embed_dim
    if pooling == "mean":
        return embed_dim
    raise ValueError(f"Unknown prefix_pooling {pooling!r}; expected one of {PREFIX_POOLING_MODES}")


def pool_prefix_rep(prefix_rep, pooling: str, num_text_tokens: int):
    """Pool a prefix representation (..., S, E) over its token axis.

    The language tokens are the LAST ``num_text_tokens`` positions of the
    prefix (Pi0.embed_prefix appends them after the image tokens).
    """
    xp = jnp if isinstance(prefix_rep, jax.Array) else np
    if pooling == "mean":
        return xp.mean(prefix_rep, axis=-2)
    if pooling == "image_text_mean":
        assert 0 < num_text_tokens < prefix_rep.shape[-2], (
            f"num_text_tokens={num_text_tokens} must split prefix_seq={prefix_rep.shape[-2]}"
        )
        image = prefix_rep[..., :-num_text_tokens, :]
        text = prefix_rep[..., -num_text_tokens:, :]
        return xp.concatenate([xp.mean(image, axis=-2), xp.mean(text, axis=-2)], axis=-1)
    raise ValueError(f"Unknown prefix_pooling {pooling!r}; expected one of {PREFIX_POOLING_MODES}")
