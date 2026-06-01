from typing import Callable, Optional, Sequence

import flax.nnx as nnx
import jax.numpy as jnp

from src.rl.networks.constants import default_init


class _TransformerBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_mult: int,
        activations: Callable[[jnp.ndarray], jnp.ndarray],
        dropout_rate: Optional[float],
        init_scale: float,
        *,
        rngs: nnx.Rngs,
    ):
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
        self._activations = activations
        self._dropout_rate = dropout_rate

        self.attn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            use_bias=False,
            dropout_rate=dropout_rate or 0.0,
            broadcast_dropout=False,
            decode=False,
            kernel_init=default_init(init_scale),
            rngs=rngs,
        )

        self.ffn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        ffn_dim = d_model * ffn_mult
        self.gate_proj = nnx.Linear(
            d_model,
            ffn_dim,
            use_bias=False,
            kernel_init=default_init(init_scale),
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            d_model,
            ffn_dim,
            use_bias=False,
            kernel_init=default_init(init_scale),
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            ffn_dim,
            d_model,
            use_bias=False,
            kernel_init=default_init(init_scale),
            rngs=rngs,
        )

        if dropout_rate is not None:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jnp.ndarray,
        token_mask: jnp.ndarray | None = None,
        training: bool = False,
    ) -> jnp.ndarray:
        h = self.attn_norm(x)
        attn_mask = None
        if token_mask is not None:
            attn_mask = token_mask[:, None, None, :]
        h = self.self_attn(h, mask=attn_mask, deterministic=not training)
        if self.dropout is not None:
            h = self.dropout(h, deterministic=not training)
        x = x + h

        h = self.ffn_norm(x)
        h = self.down_proj(self._activations(self.gate_proj(h)) * self.up_proj(h))
        if self.dropout is not None:
            h = self.dropout(h, deterministic=not training)
        x = x + h
        return x


class Transformer(nnx.Module):
    def __init__(
        self,
        input: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        hidden_dims: Sequence[int],
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nnx.silu,
        activate_final: bool = False,
        dropout_rate: Optional[float] = None,
        init_scale: Optional[float] = 1.0,
        use_layer_norm: bool = False,
        *,
        num_heads: int = 4,
        ffn_mult: int = 4,
        rngs: nnx.Rngs,
    ):
        dummy_tokens, _, _ = input
        d_input = dummy_tokens.shape[-1]
        max_seq_len = int(dummy_tokens.shape[1])
        d_model = hidden_dims[0]

        self._activations = activations
        self._activate_final = activate_final
        self._num_heads = num_heads

        self.input_proj = nnx.Linear(
            d_input,
            d_model,
            use_bias=False,
            kernel_init=default_init(init_scale),
            rngs=rngs,
        )
        self.cls_token = nnx.Param(jnp.zeros((d_model,), dtype=jnp.float32))
        self.position_embedding = nnx.Embed(
            max_seq_len + 1,
            d_model,
            embedding_init=default_init(init_scale),
            rngs=rngs,
        )
        self.type_embedding = nnx.Embed(
            3,
            d_model,
            embedding_init=default_init(init_scale),
            rngs=rngs,
        )

        self.blocks = []
        self.inter_projs: list[Optional[nnx.Linear]] = []
        for i, d_model in enumerate(hidden_dims):
            self.blocks.append(
                _TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    activations=activations,
                    dropout_rate=dropout_rate,
                    init_scale=init_scale,
                    rngs=rngs,
                )
            )
            if i + 1 < len(hidden_dims) and hidden_dims[i + 1] != d_model:
                self.inter_projs.append(
                    nnx.Linear(
                        d_model,
                        hidden_dims[i + 1],
                        use_bias=False,
                        kernel_init=default_init(init_scale),
                        rngs=rngs,
                    )
                )
            else:
                self.inter_projs.append(None)

        self.final_norm = (
            nnx.RMSNorm(hidden_dims[-1], rngs=rngs) if use_layer_norm else None
        )

    def __call__(
        self,
        x: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        training: bool = False,
    ) -> jnp.ndarray:
        x, token_mask, token_type_ids = x
        assert x.ndim == 3, f"Transformer expects (B, T, D), got shape {x.shape}"
        assert x.shape[1] > 10, (
            f"Transformer expects all PaliGemma prefix embeddings + 1 state vector, got shape {x.shape}"
        )

        x = self.input_proj(x)
        batch_size = x.shape[0]
        cls = jnp.broadcast_to(
            self.cls_token.value[None, None, :],
            (batch_size, 1, x.shape[-1]),
        )
        x = jnp.concatenate([cls, x], axis=1)
        cls_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)
        token_mask = jnp.concatenate([cls_mask, token_mask.astype(jnp.bool_)], axis=1)
        cls_type = jnp.full((batch_size, 1), 2, dtype=jnp.int32)
        token_type_ids = jnp.concatenate(
            [cls_type, token_type_ids.astype(jnp.int32)], axis=1
        )

        position_ids = jnp.arange(x.shape[1], dtype=jnp.int32)
        x = (
            x
            + self.position_embedding(position_ids)[None, :, :]
            + self.type_embedding(token_type_ids)
        )
        for block, inter in zip(self.blocks, self.inter_projs):
            x = block(x, token_mask=token_mask, training=training)
            if inter is not None:
                x = inter(x)

        x = x[:, 0]

        if self.final_norm is not None:
            x = self.final_norm(x)
        if self._activate_final:
            x = self._activations(x)
        return x
