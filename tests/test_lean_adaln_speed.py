"""
Lean AdaLN Transformer: PyTorch vs JAX Speed & Accuracy Comparison

Compares:
1. PyTorch lean AdaLN (adapted from user's reference GPT_AdaLN)
2. JAX/Flax lean AdaLN (exact port)
3. Current OGPO ActorVectorFieldTF (bloated baseline)

Tests:
- Numerical accuracy: PyTorch ↔ JAX weight transfer + forward pass match
- Speed: PyTorch vs JAX forward/backward
- Speed: Lean JAX vs current OGPO TF
- Param count comparison
"""

import math
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import jax
import jax.numpy as jnp
import flax.linen as fnn
from einops import rearrange

# ============================================================================
# Config
# ============================================================================
B = 256
OBS_DIM = 23
ACTION_DIM = 7        # per-step action dim
CHUNK_SIZE = 4         # action_chunk_size (T)
TOTAL_ACT_DIM = ACTION_DIM * CHUNK_SIZE  # 28
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 3
FF_DIM = 1024          # 4 * HIDDEN_DIM
FLOW_STEPS = 10
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 64

print(f"JAX devices: {jax.devices()}")
print(f"PyTorch CUDA: {torch.cuda.is_available()}")
print(f"Config: B={B}, obs={OBS_DIM}, act={ACTION_DIM}, T={CHUNK_SIZE}, "
      f"D={HIDDEN_DIM}, L={NUM_LAYERS}, H={NUM_HEADS}, ff={FF_DIM}, S={FLOW_STEPS}")


# ============================================================================
# SECTION 1: PyTorch Lean AdaLN
# ============================================================================

def wt_init_(layer, activation="relu"):
    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation))
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
    return layer


def zero_init_(layer):
    nn.init.constant_(layer.weight, 0)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
    return layer


def build_sinusoidal_table(max_len, d_model):
    """Build sinusoidal position embedding table: (max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MultiHeadAttention_Torch(nn.Module):
    """Self-attention with separate W_Q/W_K/W_V/W_O and causal masking."""

    def __init__(self, model_dim, num_heads, max_seq_len):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads

        self.W_Q = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_K = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_V = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")
        self.W_O = wt_init_(nn.Linear(model_dim, model_dim), activation="linear")

        self.register_buffer('causal_mask',
                             torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x):
        bs, seq_len, _ = x.shape
        Q = self.W_Q(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-1, -2)) * scale
        attn = attn.masked_fill(
            self.causal_mask[:seq_len, :seq_len] == 0, float('-inf')
        )
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.model_dim)
        return self.W_O(out)


class FF_MLP_Torch(nn.Module):
    def __init__(self, model_dim, ff_dim):
        super().__init__()
        self.fc1 = wt_init_(nn.Linear(model_dim, ff_dim))
        self.fc2 = wt_init_(nn.Linear(ff_dim, model_dim))

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class AdaLNLayer_Torch(nn.Module):
    """adaLN-Zero block: zero-initialized modulation."""

    def __init__(self, model_dim, num_heads, max_seq_len, ff_dim):
        super().__init__()
        self.attn = MultiHeadAttention_Torch(model_dim, num_heads, max_seq_len)
        self.mlp = FF_MLP_Torch(model_dim, ff_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            zero_init_(nn.Linear(model_dim, 6 * model_dim, bias=True)),
        )
        self.layer_norm1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, cond):
        # cond: (B, T, D) — per-token conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        norm_x = self.layer_norm1(x)
        mod_x = norm_x * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(mod_x)

        norm_x = self.layer_norm2(x)
        mod_x = norm_x * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(mod_x)
        return x


class FinalLayer_Torch(nn.Module):
    """adaLN final layer with zero-initialized output."""

    def __init__(self, model_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            zero_init_(nn.Linear(model_dim, 2 * model_dim, bias=True)),
        )
        self.linear = zero_init_(nn.Linear(model_dim, output_dim, bias=True))

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.linear(x)


class LeanAdaLNActor_Torch(nn.Module):
    """Lean AdaLN transformer actor for flow-matching OGPO.

    Adapted from GPT_AdaLN reference: single-agent, action chunking,
    sinusoidal pos/time embeddings (0 learnable params), single linear state encoding.
    """

    def __init__(self, obs_dim, action_dim, chunk_size, hidden_dim, num_heads,
                 num_layers, ff_dim, flow_steps):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.flow_steps = flow_steps
        self.hidden_dim = hidden_dim

        # Single linear state encoding (not 2-layer MLP!)
        self.state_enc = wt_init_(nn.Linear(obs_dim, hidden_dim), activation="linear")
        # Action token projection + GELU
        self.action_proj = wt_init_(nn.Linear(action_dim, hidden_dim), activation="linear")

        # Sinusoidal buffers: 0 learnable params
        self.register_buffer('pos_table', build_sinusoidal_table(chunk_size, hidden_dim))
        self.register_buffer('time_table', build_sinusoidal_table(flow_steps, hidden_dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            AdaLNLayer_Torch(hidden_dim, num_heads, chunk_size, ff_dim)
            for _ in range(num_layers)
        ])
        self.final_layer = FinalLayer_Torch(hidden_dim, action_dim)

    def forward(self, observations, actions, times):
        """
        Args:
            observations: (B, obs_dim)
            actions: (B, T * action_dim) — flat actions
            times: (B, 1) — continuous time in [0, 1), converted to int index
        Returns:
            velocity: (B, T * action_dim) — flat velocity field
        """
        B = observations.shape[0]
        T = self.chunk_size

        # Reshape flat actions → (B, T, action_dim)
        act = actions.reshape(B, T, self.action_dim)
        x = F.gelu(self.action_proj(act))  # (B, T, D)

        # State encoding: single linear
        state_enc = self.state_enc(observations)  # (B, D)

        # Time → integer index → sinusoidal lookup
        t_idx = (times.squeeze(-1) * self.flow_steps).long().clamp(0, self.flow_steps - 1)
        time_enc = self.time_table[t_idx]  # (B, D)

        # Position: sinusoidal lookup for each token position
        pos_enc = self.pos_table[:T]  # (T, D)

        # Per-token conditioning: state + time + position
        cond = state_enc[:, None, :] + time_enc[:, None, :] + pos_enc[None, :, :]  # (B, T, D)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, cond)

        # Final layer
        out = self.final_layer(x, cond)  # (B, T, action_dim)
        return out.reshape(B, T * self.action_dim)


# ============================================================================
# SECTION 2: JAX Lean AdaLN
# ============================================================================

def jax_default_init(scale=1.0):
    return fnn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def jax_zero_init():
    return fnn.initializers.constant(0.0)


def build_sinusoidal_table_jax(max_len, d_model):
    """Build sinusoidal position embedding table: (max_len, d_model)."""
    position = jnp.arange(max_len)[:, None].astype(jnp.float32)
    div_term = jnp.exp(
        jnp.arange(0, d_model, 2).astype(jnp.float32) * (-math.log(10000.0) / d_model)
    )
    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


class MultiHeadAttention_JAX(fnn.Module):
    """Self-attention with separate W_Q/W_K/W_V/W_O and causal masking."""
    hidden_dim: int
    num_heads: int
    max_seq_len: int

    @fnn.compact
    def __call__(self, x):
        head_dim = self.hidden_dim // self.num_heads
        bs, seq_len, _ = x.shape

        Q = fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(), name='W_Q')(x)
        K = fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(), name='W_K')(x)
        V = fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(), name='W_V')(x)

        Q = Q.reshape(bs, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(bs, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(bs, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        scale = jax.lax.rsqrt(jnp.float32(head_dim))
        attn = jnp.einsum('bhqd,bhkd->bhqk', Q, K) * scale

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attn = jnp.where(causal_mask[None, None], attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum('bhqk,bhkd->bhqd', attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(bs, seq_len, self.hidden_dim)
        return fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(), name='W_O')(out)


class FF_MLP_JAX(fnn.Module):
    hidden_dim: int
    ff_dim: int

    @fnn.compact
    def __call__(self, x):
        x = fnn.Dense(self.ff_dim, kernel_init=jax_default_init(), name='fc1')(x)
        x = fnn.gelu(x)
        x = fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(), name='fc2')(x)
        return x


class AdaLNLayer_JAX(fnn.Module):
    """adaLN-Zero block with zero-initialized modulation."""
    hidden_dim: int
    num_heads: int
    max_seq_len: int
    ff_dim: int

    @fnn.compact
    def __call__(self, x, cond):
        # adaLN-Zero: SiLU + zero-init linear → 6 modulation params
        cond_processed = fnn.silu(cond)
        adaln_params = fnn.Dense(
            6 * self.hidden_dim,
            kernel_init=jax_zero_init(),
            bias_init=jax_zero_init(),
            name='adaLN_linear',
        )(cond_processed)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.split(adaln_params, 6, axis=-1)
        )

        # Attention branch
        norm_x = fnn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6,
                               name='layer_norm1')(x)
        mod_x = norm_x * (1 + scale_msa) + shift_msa
        attn_out = MultiHeadAttention_JAX(
            self.hidden_dim, self.num_heads, self.max_seq_len, name='attn'
        )(mod_x)
        x = x + gate_msa * attn_out

        # FFN branch
        norm_x = fnn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6,
                               name='layer_norm2')(x)
        mod_x = norm_x * (1 + scale_mlp) + shift_mlp
        ff_out = FF_MLP_JAX(self.hidden_dim, self.ff_dim, name='mlp')(mod_x)
        x = x + gate_mlp * ff_out
        return x


class FinalLayer_JAX(fnn.Module):
    """adaLN final layer with zero-initialized output."""
    hidden_dim: int
    output_dim: int

    @fnn.compact
    def __call__(self, x, cond):
        cond_processed = fnn.silu(cond)
        adaln_params = fnn.Dense(
            2 * self.hidden_dim,
            kernel_init=jax_zero_init(),
            bias_init=jax_zero_init(),
            name='adaLN_linear',
        )(cond_processed)
        shift, scale = jnp.split(adaln_params, 2, axis=-1)

        x = fnn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6, name='norm')(x)
        x = x * (1 + scale) + shift
        x = fnn.Dense(
            self.output_dim,
            kernel_init=jax_zero_init(),
            bias_init=jax_zero_init(),
            name='linear',
        )(x)
        return x


class MultiHeadAttention_FusedQKV_JAX(fnn.Module):
    """Self-attention with FUSED QKV Dense and causal masking (like current OGPO TF)."""
    hidden_dim: int
    num_heads: int
    max_seq_len: int

    @fnn.compact
    def __call__(self, x):
        head_dim = self.hidden_dim // self.num_heads
        bs, seq_len, _ = x.shape

        # Single fused Dense for Q, K, V
        qkv = fnn.Dense(3 * self.hidden_dim, kernel_init=jax_default_init(),
                         name='qkv')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(bs, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bs, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bs, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        scale = jax.lax.rsqrt(jnp.float32(head_dim))
        attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attn = jnp.where(causal_mask[None, None], attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(bs, seq_len, self.hidden_dim)
        return fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(), name='out_proj')(out)


class AdaLNLayer_FusedQKV_JAX(fnn.Module):
    """adaLN-Zero block with FUSED QKV attention."""
    hidden_dim: int
    num_heads: int
    max_seq_len: int
    ff_dim: int

    @fnn.compact
    def __call__(self, x, cond):
        cond_processed = fnn.silu(cond)
        adaln_params = fnn.Dense(
            6 * self.hidden_dim,
            kernel_init=jax_zero_init(),
            bias_init=jax_zero_init(),
            name='adaLN_linear',
        )(cond_processed)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.split(adaln_params, 6, axis=-1)
        )

        # Attention branch (fused QKV)
        norm_x = fnn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6,
                               name='layer_norm1')(x)
        mod_x = norm_x * (1 + scale_msa) + shift_msa
        attn_out = MultiHeadAttention_FusedQKV_JAX(
            self.hidden_dim, self.num_heads, self.max_seq_len, name='attn'
        )(mod_x)
        x = x + gate_msa * attn_out

        # FFN branch
        norm_x = fnn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6,
                               name='layer_norm2')(x)
        mod_x = norm_x * (1 + scale_mlp) + shift_mlp
        ff_out = FF_MLP_JAX(self.hidden_dim, self.ff_dim, name='mlp')(mod_x)
        x = x + gate_mlp * ff_out
        return x


class LeanFusedAdaLNActor_JAX(fnn.Module):
    """Lean AdaLN transformer actor with FUSED QKV (hybrid: lean embedding + fused attention)."""
    obs_dim: int
    action_dim: int
    chunk_size: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    flow_steps: int

    @fnn.compact
    def __call__(self, observations, actions, times):
        B = observations.shape[0]
        T = self.chunk_size

        act = actions.reshape(B, T, self.action_dim)
        x = fnn.gelu(fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(),
                                name='action_proj')(act))

        state_enc = fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(),
                              name='state_enc')(observations)

        pos_table = build_sinusoidal_table_jax(T, self.hidden_dim)
        time_table = build_sinusoidal_table_jax(self.flow_steps, self.hidden_dim)

        t_idx = (times.squeeze(-1) * self.flow_steps).astype(jnp.int32)
        t_idx = jnp.clip(t_idx, 0, self.flow_steps - 1)
        time_enc = time_table[t_idx]

        pos_enc = pos_table[:T]
        cond = state_enc[:, None, :] + time_enc[:, None, :] + pos_enc[None, :, :]

        for i in range(self.num_layers):
            x = AdaLNLayer_FusedQKV_JAX(
                self.hidden_dim, self.num_heads, T, self.ff_dim,
                name=f'layer_{i}'
            )(x, cond)

        out = FinalLayer_JAX(self.hidden_dim, self.action_dim,
                             name='final_layer')(x, cond)
        return out.reshape(B, T * self.action_dim)


class LeanAdaLNActor_JAX(fnn.Module):
    """Lean AdaLN transformer actor for flow-matching OGPO (JAX/Flax port)."""
    obs_dim: int
    action_dim: int
    chunk_size: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    flow_steps: int

    @fnn.compact
    def __call__(self, observations, actions, times):
        """
        Args:
            observations: (B, obs_dim)
            actions: (B, T * action_dim) — flat actions
            times: (B, 1) — continuous time in [0, 1)
        Returns:
            velocity: (B, T * action_dim)
        """
        B = observations.shape[0]
        T = self.chunk_size

        # Reshape flat actions → (B, T, action_dim)
        act = actions.reshape(B, T, self.action_dim)
        x = fnn.gelu(fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(),
                                name='action_proj')(act))  # (B, T, D)

        # State encoding: single linear
        state_enc = fnn.Dense(self.hidden_dim, kernel_init=jax_default_init(),
                              name='state_enc')(observations)  # (B, D)

        # Time → integer index → sinusoidal lookup
        pos_table = build_sinusoidal_table_jax(T, self.hidden_dim)
        time_table = build_sinusoidal_table_jax(self.flow_steps, self.hidden_dim)

        t_idx = (times.squeeze(-1) * self.flow_steps).astype(jnp.int32)
        t_idx = jnp.clip(t_idx, 0, self.flow_steps - 1)
        time_enc = time_table[t_idx]  # (B, D)

        # Position: sinusoidal for each token position
        pos_enc = pos_table[:T]  # (T, D)

        # Per-token conditioning
        cond = state_enc[:, None, :] + time_enc[:, None, :] + pos_enc[None, :, :]  # (B, T, D)

        # Transformer layers
        for i in range(self.num_layers):
            x = AdaLNLayer_JAX(
                self.hidden_dim, self.num_heads, T, self.ff_dim,
                name=f'layer_{i}'
            )(x, cond)

        # Final layer
        out = FinalLayer_JAX(self.hidden_dim, self.action_dim,
                             name='final_layer')(x, cond)  # (B, T, action_dim)
        return out.reshape(B, T * self.action_dim)


# ============================================================================
# SECTION 3: Weight Transfer (PyTorch → JAX)
# ============================================================================

def transfer_weights_torch_to_jax(torch_model, jax_params):
    """Transfer weights from PyTorch lean model to JAX lean model.

    PyTorch nn.Linear stores weight as (out, in), Flax Dense stores kernel as (in, out).
    """
    new_params = jax.tree.map(lambda x: x, jax_params)  # deep copy structure

    def t2j(tensor):
        return jnp.array(tensor.detach().cpu().numpy())

    sd = torch_model.state_dict()
    params = dict(new_params['params'])

    # state_enc
    params['state_enc'] = {
        'kernel': t2j(sd['state_enc.weight'].T),
        'bias': t2j(sd['state_enc.bias']),
    }

    # action_proj
    params['action_proj'] = {
        'kernel': t2j(sd['action_proj.weight'].T),
        'bias': t2j(sd['action_proj.bias']),
    }

    # Transformer layers
    for i in range(torch_model.layers.__len__()):
        prefix = f'layers.{i}'
        layer_key = f'layer_{i}'

        layer_params = {}

        # adaLN linear (Sequential[1] = the Linear)
        layer_params['adaLN_linear'] = {
            'kernel': t2j(sd[f'{prefix}.adaLN_modulation.1.weight'].T),
            'bias': t2j(sd[f'{prefix}.adaLN_modulation.1.bias']),
        }

        # Attention W_Q, W_K, W_V, W_O
        layer_params['attn'] = {}
        for w_name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            layer_params['attn'][w_name] = {
                'kernel': t2j(sd[f'{prefix}.attn.{w_name}.weight'].T),
                'bias': t2j(sd[f'{prefix}.attn.{w_name}.bias']),
            }

        # FF MLP
        layer_params['mlp'] = {
            'fc1': {
                'kernel': t2j(sd[f'{prefix}.mlp.fc1.weight'].T),
                'bias': t2j(sd[f'{prefix}.mlp.fc1.bias']),
            },
            'fc2': {
                'kernel': t2j(sd[f'{prefix}.mlp.fc2.weight'].T),
                'bias': t2j(sd[f'{prefix}.mlp.fc2.bias']),
            },
        }

        params[layer_key] = layer_params

    # Final layer
    params['final_layer'] = {
        'adaLN_linear': {
            'kernel': t2j(sd['final_layer.adaLN_modulation.1.weight'].T),
            'bias': t2j(sd['final_layer.adaLN_modulation.1.bias']),
        },
        'linear': {
            'kernel': t2j(sd['final_layer.linear.weight'].T),
            'bias': t2j(sd['final_layer.linear.bias']),
        },
    }

    return {'params': params}


# ============================================================================
# SECTION 4: Tests
# ============================================================================

def count_params_torch(model):
    return sum(p.numel() for p in model.parameters())


def count_params_jax(params):
    return sum(x.size for x in jax.tree.leaves(params))


def randomize_adaln_weights(model):
    """Re-initialize adaLN modulation layers with non-zero weights for meaningful testing.
    At zero-init, the entire network outputs zeros (identity residual + zero final layer).
    """
    for name, param in model.named_parameters():
        if 'adaLN_modulation' in name or 'final_layer' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.uniform_(param, -0.1, 0.1)


def benchmark(fn, warmup=10, iters=100, name=""):
    """Benchmark a function, return mean and std in ms."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        # Sync
        if isinstance(result, torch.Tensor):
            torch.cuda.synchronize()
        else:
            jax.block_until_ready(result)
        times.append((time.perf_counter() - t0) * 1000)

    mean = np.mean(times)
    std = np.std(times)
    if name:
        print(f"  {name}: {mean:.2f} ± {std:.2f} ms")
    return mean, std


# ---------- Test 1: Numerical Accuracy ----------
def test_numerical_accuracy():
    print("\n" + "=" * 70)
    print("TEST 1: Numerical Accuracy (PyTorch ↔ JAX weight transfer)")
    print("=" * 70)

    # Create PyTorch model on CPU with non-zero adaLN weights for meaningful test
    torch_model = LeanAdaLNActor_Torch(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    )
    randomize_adaln_weights(torch_model)
    torch_model.eval()

    # Create JAX model and init with dummy input
    jax_model = LeanAdaLNActor_JAX(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    )
    rng = jax.random.PRNGKey(42)
    dummy_obs = jnp.ones((1, OBS_DIM))
    dummy_act = jnp.ones((1, TOTAL_ACT_DIM))
    dummy_t = jnp.ones((1, 1)) * 0.5
    jax_params = jax_model.init(rng, dummy_obs, dummy_act, dummy_t)

    # Transfer weights
    jax_params = transfer_weights_torch_to_jax(torch_model, jax_params)

    # Create identical inputs
    np.random.seed(123)
    obs_np = np.random.randn(B, OBS_DIM).astype(np.float32)
    act_np = np.random.randn(B, TOTAL_ACT_DIM).astype(np.float32)
    # Use discrete time values that map cleanly to indices
    t_np = np.full((B, 1), 0.3, dtype=np.float32)  # maps to index 3

    obs_torch = torch.from_numpy(obs_np)
    act_torch = torch.from_numpy(act_np)
    t_torch = torch.from_numpy(t_np)

    obs_jax = jnp.array(obs_np)
    act_jax = jnp.array(act_np)
    t_jax = jnp.array(t_np)

    # Forward pass
    with torch.no_grad():
        out_torch = torch_model(obs_torch, act_torch, t_torch).numpy()

    out_jax = np.array(jax_model.apply(jax_params, obs_jax, act_jax, t_jax))

    max_diff = np.max(np.abs(out_torch - out_jax))
    mean_diff = np.mean(np.abs(out_torch - out_jax))

    print(f"\n  Forward pass comparison:")
    print(f"    Max abs diff:  {max_diff:.2e}")
    print(f"    Mean abs diff: {mean_diff:.2e}")
    print(f"    PyTorch output range: [{out_torch.min():.4f}, {out_torch.max():.4f}]")
    print(f"    JAX output range:     [{out_jax.min():.4f}, {out_jax.max():.4f}]")

    if max_diff < 1e-4:
        print(f"    ✓ Numerical match confirmed (tol=1e-4)")
    else:
        print(f"    ✗ MISMATCH! max diff = {max_diff:.2e}")

    return max_diff


# ---------- Test 2: Speed — PyTorch vs JAX ----------
def test_speed_torch_vs_jax():
    print("\n" + "=" * 70)
    print("TEST 2: Speed — PyTorch vs JAX (forward + backward)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PyTorch model
    torch_model = LeanAdaLNActor_Torch(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    ).to(device)

    # Try torch.compile (may fail without Python.h)
    try:
        # Quick smoke test to catch compilation errors early
        _test_input = torch.randn(1, OBS_DIM, device=device)
        _test_act = torch.randn(1, TOTAL_ACT_DIM, device=device)
        _test_t = torch.full((1, 1), 0.3, device=device)
        torch_compiled = torch.compile(torch_model)
        _ = torch_compiled(_test_input, _test_act, _test_t)
        use_compiled = True
    except Exception as e:
        print(f"  torch.compile failed ({type(e).__name__}), falling back to eager mode")
        torch_compiled = torch_model
        use_compiled = False

    # JAX model
    jax_model = LeanAdaLNActor_JAX(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    )
    rng = jax.random.PRNGKey(42)
    dummy_obs = jnp.ones((B, OBS_DIM))
    dummy_act = jnp.ones((B, TOTAL_ACT_DIM))
    dummy_t = jnp.ones((B, 1)) * 0.5
    jax_params = jax_model.init(rng, dummy_obs, dummy_act, dummy_t)

    # Inputs
    obs_torch = torch.randn(B, OBS_DIM, device=device)
    act_torch = torch.randn(B, TOTAL_ACT_DIM, device=device)
    t_torch = torch.full((B, 1), 0.3, device=device)

    obs_jax = jnp.array(obs_torch.cpu().numpy())
    act_jax = jnp.array(act_torch.cpu().numpy())
    t_jax = jnp.array(t_torch.cpu().numpy())

    # JIT compile JAX
    jax_forward = jax.jit(lambda p, o, a, t: jax_model.apply(p, o, a, t))
    # Warmup JIT
    _ = jax_forward(jax_params, obs_jax, act_jax, t_jax)

    # --- Forward pass ---
    print(f"\n  Forward pass (B={B}):")
    print(f"  {'(torch.compile)' if use_compiled else '(eager)'}")

    benchmark(lambda: torch_compiled(obs_torch, act_torch, t_torch),
              name="PyTorch forward")
    benchmark(lambda: jax_forward(jax_params, obs_jax, act_jax, t_jax),
              name="JAX forward")

    # --- Forward + Backward ---
    print(f"\n  Forward + Backward (grad of MSE loss, B={B}):")

    target_torch = torch.randn(B, TOTAL_ACT_DIM, device=device)
    target_jax = jnp.array(target_torch.cpu().numpy())

    def torch_fwd_bwd():
        torch_model.zero_grad()
        out = torch_compiled(obs_torch, act_torch, t_torch)
        loss = F.mse_loss(out, target_torch)
        loss.backward()
        return loss

    @jax.jit
    def jax_fwd_bwd(params):
        def loss_fn(p):
            out = jax_model.apply(p, obs_jax, act_jax, t_jax)
            return jnp.mean((out - target_jax) ** 2)
        return jax.grad(loss_fn)(params)

    # Warmup
    _ = jax_fwd_bwd(jax_params)

    benchmark(torch_fwd_bwd, name="PyTorch fwd+bwd")
    benchmark(lambda: jax_fwd_bwd(jax_params), name="JAX fwd+bwd")

    # --- 10 sequential forward calls (simulating flow_steps) ---
    print(f"\n  10 sequential forward calls (simulating flow_steps, B={B}):")

    @jax.jit
    def jax_10_calls(params, obs, act, t):
        result = jnp.zeros((B, TOTAL_ACT_DIM))
        for i in range(10):
            ti = jnp.full((B, 1), i / 10.0)
            result = result + jax_model.apply(params, obs, act, ti)
        return result

    def torch_10_calls():
        result = torch.zeros(B, TOTAL_ACT_DIM, device=device)
        for i in range(10):
            ti = torch.full((B, 1), i / 10.0, device=device)
            result = result + torch_compiled(obs_torch, act_torch, ti)
        return result

    # Warmup
    _ = jax_10_calls(jax_params, obs_jax, act_jax, t_jax)

    benchmark(torch_10_calls, name="PyTorch 10 calls")
    benchmark(lambda: jax_10_calls(jax_params, obs_jax, act_jax, t_jax),
              name="JAX 10 calls (JIT-fused)")


# ---------- Test 3: Lean JAX vs Current OGPO TF ----------
def test_lean_vs_bloated():
    print("\n" + "=" * 70)
    print("TEST 3: Lean JAX vs Current OGPO ActorVectorFieldTF")
    print("=" * 70)

    # Import current OGPO TF
    sys.path.insert(0, '.')
    from ogpo.networks.actors import ActorVectorFieldTF

    # Lean model (separate Q/K/V/O)
    lean_model = LeanAdaLNActor_JAX(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    )

    # Lean + fused QKV (hybrid)
    hybrid_model = LeanFusedAdaLNActor_JAX(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    )

    # Current OGPO TF (adaLN)
    bloated_model = ActorVectorFieldTF(
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        action_chunk_size=CHUNK_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=0.0,
        use_denoiser=True,
        conditioning_type='adaln',
    )

    # Cross-attention variant (same preprocessing, cross-attn conditioning)
    crossattn_model = ActorVectorFieldTF(
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        action_chunk_size=CHUNK_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=0.0,
        use_denoiser=True,
        conditioning_type='cross_attn',
    )

    # Init all
    rng = jax.random.PRNGKey(42)
    obs = jnp.ones((B, OBS_DIM))
    act = jnp.ones((B, TOTAL_ACT_DIM))
    t = jnp.ones((B, 1)) * 0.5

    lean_params = lean_model.init(rng, obs, act, t)
    hybrid_params = hybrid_model.init(rng, obs, act, t)
    bloated_params = bloated_model.init(rng, obs, act, t, deterministic=True)
    crossattn_params = crossattn_model.init(rng, obs, act, t, deterministic=True)

    lean_count = count_params_jax(lean_params)
    hybrid_count = count_params_jax(hybrid_params)
    bloated_count = count_params_jax(bloated_params)
    crossattn_count = count_params_jax(crossattn_params)

    print(f"\n  Parameter counts:")
    print(f"    Lean JAX (sep Q/K/V/O):  {lean_count:,}")
    print(f"    Lean+FusedQKV (hybrid):  {hybrid_count:,}")
    print(f"    AdaLN OGPO TF:           {bloated_count:,}")
    print(f"    CrossAttn OGPO TF:       {crossattn_count:,}")

    # JIT compile
    lean_fwd = jax.jit(lambda p, o, a, t: lean_model.apply(p, o, a, t))
    hybrid_fwd = jax.jit(lambda p, o, a, t: hybrid_model.apply(p, o, a, t))
    bloated_fwd = jax.jit(lambda p, o, a, t: bloated_model.apply(p, o, a, t, deterministic=True))
    crossattn_fwd = jax.jit(lambda p, o, a, t: crossattn_model.apply(p, o, a, t, deterministic=True))

    # Warmup
    _ = lean_fwd(lean_params, obs, act, t)
    _ = hybrid_fwd(hybrid_params, obs, act, t)
    _ = bloated_fwd(bloated_params, obs, act, t)
    _ = crossattn_fwd(crossattn_params, obs, act, t)

    # --- Single forward pass ---
    print(f"\n  Single forward pass (B={B}):")
    lean_time, _ = benchmark(lambda: lean_fwd(lean_params, obs, act, t), name="Lean JAX (sep QKV)")
    hybrid_time, _ = benchmark(lambda: hybrid_fwd(hybrid_params, obs, act, t), name="Lean+FusedQKV")
    bloated_time, _ = benchmark(lambda: bloated_fwd(bloated_params, obs, act, t), name="AdaLN OGPO TF")
    crossattn_time, _ = benchmark(lambda: crossattn_fwd(crossattn_params, obs, act, t), name="CrossAttn OGPO TF")
    print(f"  CrossAttn vs AdaLN:  {bloated_time/crossattn_time:.2f}x")

    # --- Forward + backward ---
    print(f"\n  Forward + Backward (B={B}):")

    target = jnp.ones((B, TOTAL_ACT_DIM))

    @jax.jit
    def lean_grad(params):
        def loss_fn(p):
            out = lean_model.apply(p, obs, act, t)
            return jnp.mean((out - target) ** 2)
        return jax.grad(loss_fn)(params)

    @jax.jit
    def hybrid_grad(params):
        def loss_fn(p):
            out = hybrid_model.apply(p, obs, act, t)
            return jnp.mean((out - target) ** 2)
        return jax.grad(loss_fn)(params)

    @jax.jit
    def bloated_grad(params):
        def loss_fn(p):
            out = bloated_model.apply(p, obs, act, t, deterministic=True)
            return jnp.mean((out - target) ** 2)
        return jax.grad(loss_fn)(params)

    @jax.jit
    def crossattn_grad(params):
        def loss_fn(p):
            out = crossattn_model.apply(p, obs, act, t, deterministic=True)
            return jnp.mean((out - target) ** 2)
        return jax.grad(loss_fn)(params)

    _ = lean_grad(lean_params)
    _ = hybrid_grad(hybrid_params)
    _ = bloated_grad(bloated_params)
    _ = crossattn_grad(crossattn_params)

    lean_time, _ = benchmark(lambda: lean_grad(lean_params), name="Lean fwd+bwd")
    hybrid_time, _ = benchmark(lambda: hybrid_grad(hybrid_params), name="Hybrid fwd+bwd")
    bloated_time, _ = benchmark(lambda: bloated_grad(bloated_params), name="AdaLN fwd+bwd")
    crossattn_time, _ = benchmark(lambda: crossattn_grad(crossattn_params), name="CrossAttn fwd+bwd")
    print(f"  CrossAttn vs AdaLN:  {bloated_time/crossattn_time:.2f}x")


# ---------- Test 4: Param Count Summary ----------
def test_param_count():
    print("\n" + "=" * 70)
    print("TEST 4: Parameter Count Summary")
    print("=" * 70)

    # PyTorch lean
    torch_model = LeanAdaLNActor_Torch(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    )

    # JAX lean (separate Q/K/V/O)
    jax_model = LeanAdaLNActor_JAX(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    )

    # JAX lean + fused QKV (hybrid)
    hybrid_model = LeanFusedAdaLNActor_JAX(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, flow_steps=FLOW_STEPS,
    )

    rng = jax.random.PRNGKey(42)
    obs = jnp.ones((1, OBS_DIM))
    act = jnp.ones((1, TOTAL_ACT_DIM))
    t = jnp.ones((1, 1)) * 0.5
    jax_params = jax_model.init(rng, obs, act, t)
    hybrid_params = hybrid_model.init(rng, obs, act, t)

    # Current OGPO TF (adaLN)
    sys.path.insert(0, '.')
    from ogpo.networks.actors import ActorVectorFieldTF
    adaln_model = ActorVectorFieldTF(
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        action_chunk_size=CHUNK_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=0.0,
        use_denoiser=True,
        conditioning_type='adaln',
    )
    adaln_params = adaln_model.init(rng, obs, act, t, deterministic=True)

    # Cross-attention variant
    crossattn_model = ActorVectorFieldTF(
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        action_chunk_size=CHUNK_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=0.0,
        use_denoiser=True,
        conditioning_type='cross_attn',
    )
    crossattn_params = crossattn_model.init(rng, obs, act, t, deterministic=True)

    torch_count = count_params_torch(torch_model)
    jax_count = count_params_jax(jax_params)
    hybrid_count = count_params_jax(hybrid_params)
    adaln_count = count_params_jax(adaln_params)
    crossattn_count = count_params_jax(crossattn_params)

    print(f"\n  {'Model':<30} {'Params':>12}")
    print(f"  {'-'*30} {'-'*12}")
    print(f"  {'PyTorch lean (sep QKV)':<30} {torch_count:>12,}")
    print(f"  {'JAX lean (sep QKV)':<30} {jax_count:>12,}")
    print(f"  {'JAX lean+fusedQKV (hybrid)':<30} {hybrid_count:>12,}")
    print(f"  {'OGPO TF AdaLN':<30} {adaln_count:>12,}")
    print(f"  {'OGPO TF CrossAttn':<30} {crossattn_count:>12,}")
    print(f"\n  AdaLN vs CrossAttn: {adaln_count - crossattn_count:,} params ({100*(adaln_count - crossattn_count)/adaln_count:.1f}%)")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    test_param_count()
    test_numerical_accuracy()
    test_speed_torch_vs_jax()
    test_lean_vs_bloated()
