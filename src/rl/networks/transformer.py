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

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        h = self.attn_norm(x)
        h = self.self_attn(h, deterministic=not training)
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
        input: jnp.ndarray | int,
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
        d_input = input if isinstance(input, int) else input.shape[-1]

        self._activations = activations
        self._activate_final = activate_final

        self.input_proj = nnx.Linear(
            d_input,
            hidden_dims[0],
            use_bias=False,
            kernel_init=default_init(init_scale),
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

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        assert x.ndim == 3, f"Transformer expects (B, T, D), got shape {x.shape}"
        assert x.shape[1] > 10, (
            f"Transformer expects all PaliGemma prefix embeddings + 1 state vector, got shape {x.shape}"
        )

        x = self.input_proj(x)
        for block, inter in zip(self.blocks, self.inter_projs):
            x = block(x, training=training)
            if inter is not None:
                x = inter(x)

        x = jnp.mean(x, axis=-2)

        if self.final_norm is not None:
            x = self.final_norm(x)
        if self._activate_final:
            x = self._activations(x)
        return x
