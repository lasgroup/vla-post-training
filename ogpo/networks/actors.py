from typing import Any, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.modules import (
    MLP,
    MLPCond,
    MLPWithFiLM,
    SimBaMLP,
    TanhMultivariateNormalDiag,
    AdaLNLayer,
    AdaLNFinalLayer,
    CrossAttnLayer,
    CrossAttnFinalLayer,
    SinusoidalTimeEmbedding,
    TwoTierObsEncoder,
)


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class TanhNormalDist:
    """Wrapper around distrax.Transformed(Normal, Tanh) with mode() support."""

    def __init__(self, base_dist, base_mean):
        self._dist = distrax.Transformed(base_dist, distrax.Tanh())
        self._base_mean = base_mean

    def sample(self, seed):
        return self._dist.sample(seed=seed)

    def sample_and_log_prob(self, seed):
        samples, log_prob = self._dist.sample_and_log_prob(seed=seed)
        return samples, log_prob.sum(axis=-1)

    def log_prob(self, value):
        return self._dist.log_prob(value).sum(axis=-1)

    def mode(self):
        return jnp.tanh(self._base_mean)


class Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = True
    const_std: bool = False
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, temperature=1.0,):
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TanhMultivariateNormalDiag(distribution, low=self.low, high=self.high)

        return distribution


class ActorVectorField(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_film: bool = False
    use_denoiser: bool = False
    time_embedding: Optional[nn.Module] = None
    obs_two_tier: bool = False
    two_tier_img_dim: int = 0
    two_tier_proprio_dim: int = 0
    two_tier_fused_dim: int = 0  # if 0, defaults to hidden_dims[0]

    def setup(self) -> None:
        if self.obs_two_tier:
            fused = self.two_tier_fused_dim if self.two_tier_fused_dim > 0 else self.hidden_dims[0]
            self.two_tier_encoder = TwoTierObsEncoder(
                img_feat_dim=self.two_tier_img_dim,
                proprio_dim=self.two_tier_proprio_dim,
                fused_dim=fused,
            )

        if self.use_film:
            self.mlp = MLPWithFiLM(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )
        else:
            self.mlp = MLP(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )

        if self.use_denoiser:
            if self.use_film:
                self.denoiser_mlp = MLPWithFiLM(
                    (*self.hidden_dims, self.action_dim),
                    activate_final=False,
                    layer_norm=self.layer_norm
                )
            else:
                self.denoiser_mlp = MLP(
                    (*self.hidden_dims, self.action_dim),
                    activate_final=False,
                    layer_norm=self.layer_norm
                )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    @nn.compact
    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False, return_denoiser=False, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        if self.obs_two_tier:
            observations = self.two_tier_encoder(observations)

        # Embed timestep (sinusoidal or raw scalar)
        if times is not None and self.time_embedding is not None:
            times = self.time_embedding(times)

        if self.use_film:
            cond = observations
            if times is not None:
                cond = jnp.concatenate([cond, times], axis=-1)
            if dt is not None:
                cond = jnp.concatenate([cond, dt], axis=-1)
            # Velocity output
            v = self.mlp(actions, cond)
        else:
            if times is None:
                inputs = jnp.concatenate([observations, actions], axis=-1)
            else:
                inputs = jnp.concatenate([observations, actions, times], axis=-1)
            if dt is not None:
                inputs = jnp.concatenate([inputs, dt], axis=-1)
            # Velocity output
            v = self.mlp(inputs)

        if return_denoiser:
            if not self.use_denoiser:
                raise ValueError("Actor was initialized with use_denoiser=False")

            if self.use_film:
                z, _ = self.denoiser_mlp(actions, cond)  # MLPWithFiLM returns tuple
            else:
                z = self.denoiser_mlp(inputs)
            return v, z

        return v


class ActorVectorFieldTF(nn.Module):
    hidden_dim: int
    action_dim: int
    action_chunk_size: int = 4  # T
    layer_norm: bool = False
    encoder: nn.Module = None
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.0
    use_denoiser: bool = False
    conditioning_type: str = 'adaln'  # 'adaln' or 'cross_attn'
    time_embedding: Optional[nn.Module] = None

    def setup(self) -> None:
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.conditioning_type in ('adaln', 'cross_attn'), \
            f"conditioning_type must be 'adaln' or 'cross_attn', got '{self.conditioning_type}'"

        self.action_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())
        self.cond_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())

        if self.conditioning_type == 'adaln':
            self.transformer_layers = [
                AdaLNLayer(self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                           self.dropout_rate)
                for _ in range(self.num_layers)
            ]
            self.final_layer = AdaLNFinalLayer(self.hidden_dim, self.action_dim)
            if self.use_denoiser:
                self.denoiser_final_layer = AdaLNFinalLayer(self.hidden_dim, self.action_dim)
        else:  # cross_attn
            self.transformer_layers = [
                CrossAttnLayer(self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                               self.dropout_rate)
                for _ in range(self.num_layers)
            ]
            self.final_layer = CrossAttnFinalLayer(self.hidden_dim, self.action_dim)
            if self.use_denoiser:
                self.denoiser_final_layer = CrossAttnFinalLayer(self.hidden_dim, self.action_dim)

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    @nn.compact
    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False,
                 return_denoiser=False, deterministic=True, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Reshape flat actions -> (B, T, a_dim); handle unbatched (2D) input
        if actions.ndim == 2:
            actions = actions.reshape(-1, self.action_chunk_size, self.action_dim)
        else:
            actions = actions.reshape(self.action_chunk_size, self.action_dim)

        x = self.action_proj(actions)  # (B, T, hidden_dim)
        cond = self.cond_proj(observations)  # (B, hidden_dim)

        if times is not None:
            t_feat = self.time_embedding(times) if self.time_embedding is not None else times
            time_emb = nn.Dense(self.hidden_dim, kernel_init=default_init())(t_feat)
            cond = cond + time_emb

        if dt is not None:
            dt_emb = nn.Dense(self.hidden_dim, kernel_init=default_init())(dt)
            cond = cond + dt_emb

        if self.conditioning_type == 'adaln':
            # AdaLN: cond is (B, D), passed directly to each layer
            for layer in self.transformer_layers:
                x = layer(x, cond, deterministic=deterministic)
            v = self.final_layer(x, cond)  # (B, T, action_dim)
            if self.use_denoiser:
                z = self.denoiser_final_layer(x, cond)
        else:
            # Cross-attn: form context tokens [state_token, time_token] -> (B, N, D)
            # cond already has state + time summed into (B, D); expand to (B, 1, D)
            context = cond[:, None, :]  # (B, 1, D)
            for layer in self.transformer_layers:
                x = layer(x, context, deterministic=deterministic)
            v = self.final_layer(x)  # (B, T, action_dim)
            if self.use_denoiser:
                z = self.denoiser_final_layer(x)

        if self.use_denoiser:
            z = z.reshape(-1, self.action_chunk_size * self.action_dim)
            if return_denoiser:
                v = v.reshape(-1, self.action_chunk_size * self.action_dim)
                return v, z

        v = v.reshape(-1, self.action_chunk_size * self.action_dim)  # (B, T*a_dim)
        return v


class ActorVectorFieldSimBa(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    encoder: Optional[nn.Module] = None
    rs_norm_momentum: float = 0.99
    rs_norm_epsilon: float = 1e-8
    time_embedding: Optional[nn.Module] = None

    def setup(self):
        self.simba_mlp = SimBaMLP(
            hidden_dims=(*self.hidden_dims, self.action_dim),
            rs_norm_momentum=self.rs_norm_momentum,
            rs_norm_epsilon=self.rs_norm_epsilon
        )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Embed timestep (sinusoidal or raw scalar)
        if times is not None and self.time_embedding is not None:
            times = self.time_embedding(times)

        # Concatenate all inputs
        inputs = [observations, actions]
        if times is not None:
            inputs.append(times)
        if dt is not None:
            inputs.append(dt)

        x = jnp.concatenate(inputs, axis=-1)

        # Pass through SimBa MLP
        return self.simba_mlp(x)


class EditPolicy(nn.Module):
    hidden_dims: Tuple[int, ...]
    action_dim: int
    encoder: nn.Module = None
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    edit_bound: float = 0.1
    layer_norm: bool = True

    def setup(self):
        if self.encoder:
            self.encoder_module = self.encoder

        mlp_dims = self.hidden_dims

        self.net = nn.Sequential([
            nn.Dense(mlp_dims[0]),
            nn.LayerNorm() if self.layer_norm else lambda x: x,
            nn.relu,
            *[
                nn.Sequential([
                    nn.Dense(dim),
                    nn.LayerNorm() if self.layer_norm else lambda x: x,
                    nn.relu,
                ]) for dim in mlp_dims[1:]
            ]
        ])

        self.mean_layer = nn.Dense(self.action_dim)
        self.log_std_layer = nn.Dense(self.action_dim)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, base_actions: jnp.ndarray, is_encoded: bool = False) -> distrax.Distribution:
        if self.encoder and not is_encoded:
            observations = self.encoder_module(observations)

        if observations.ndim > 2:
            observations = observations.reshape(observations.shape[0], -1)

        x = jnp.concatenate([observations, base_actions], axis=-1)
        x = self.net(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)
        base_dist = distrax.Normal(loc=mean, scale=std)
        return TanhNormalDist(base_dist, mean)


class EditActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    encoder: nn.Module = None
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    edit_action_scale: float = 0.5  # Maximum edit magnitude
    layer_norm: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    def setup(self):
        # MLP backbone
        self.edit_net = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm
        )

        # Output layers for mean and std
        self.mean_net = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_fc_init_scale)
        )

        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            self.log_stds = self.param(
                'log_stds',
                nn.initializers.zeros,
                (self.action_dim,)
            )

    def __call__(self, observations, base_actions, is_encoded=False, temperature=1.0):
        if self.encoder is not None and not is_encoded:
            observations = self.encoder(observations)

        # Concatenate observations and base actions
        inputs = jnp.concatenate([observations, base_actions], axis=-1)
        outputs = self.edit_net(inputs)

        # Compute mean and std
        means = self.mean_net(outputs)

        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # Create distribution - note we use TanhMultivariateNormalDiag to bound edits
        distribution = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * temperature
        )

        # Apply tanh squashing to bound edits to [-edit_action_scale, +edit_action_scale]
        distribution = TanhMultivariateNormalDiag(
            distribution,
            low=-self.edit_action_scale,
            high=self.edit_action_scale
        )

        return distribution


__all__ = [
    'Actor',
    'ActorVectorField',
    'ActorVectorFieldTF',
    'ActorVectorFieldSimBa',
    'EditPolicy',
    'EditActor',
]
