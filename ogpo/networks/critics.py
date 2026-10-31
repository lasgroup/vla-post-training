from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.modules import (
    BroNet,
    MLP,
    MLPCond,
    MLPWithFiLM,
    SimBaMLP,
    AdaLNLayer,
    CrossAttnLayer,
    TwoTierObsEncoder,
)


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True, 'dropout': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    action_repeat: bool = False
    use_film: bool = False
    # Two-tier obs encoder (frozen-encoder image runs): see TwoTierObsEncoder.
    obs_two_tier: bool = False
    two_tier_img_dim: int = 0
    two_tier_proprio_dim: int = 0
    two_tier_fused_dim: int = 0  # if 0, defaults to hidden_dims[0]
    head_arch: str = 'mlp'
    bronet_hidden_dim: int = 1024
    bronet_depth: int = 2

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        if self.obs_two_tier:
            fused = self.two_tier_fused_dim if self.two_tier_fused_dim > 0 else self.hidden_dims[0]
            self.two_tier_encoder = TwoTierObsEncoder(
                img_feat_dim=self.two_tier_img_dim,
                proprio_dim=self.two_tier_proprio_dim,
                fused_dim=fused,
            )

        if self.head_arch == 'bronet':
            if self.use_film or self.action_repeat or self.critic_loss_type != 'mse':
                raise ValueError(f"head_arch='bronet' supports critic_loss_type='mse' without use_film / action_repeat; "
                                 f"got critic_loss_type={self.critic_loss_type!r}, use_film={self.use_film}, "
                                 f"action_repeat={self.action_repeat}")
            bronet_class = ensemblize(BroNet, self.num_ensembles) if self.num_ensembles > 1 else BroNet
            self.value_net = bronet_class(hidden_dim=self.bronet_hidden_dim, depth=self.bronet_depth,
                                          out_dim=num_output)
            return
        if self.head_arch != 'mlp':
            raise ValueError(f"head_arch={self.head_arch!r}; expected 'mlp' or 'bronet'")

        if self.use_film:
            mlp_class = MLPWithFiLM
        else:
            mlp_class = MLPCond if self.action_repeat else MLP

        if self.num_ensembles > 1:
            if self.use_film:
                # For FiLM, ensemblize with two inputs (x, cond)
                mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=(None, None), out_axes=0)
            else:
                mlp_class = ensemblize(mlp_class, self.num_ensembles)

        value_net = mlp_class(
            (*self.hidden_dims, num_output),
            activate_final=False,
            layer_norm=self.layer_norm
        )
        self.value_net = value_net

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, images=None):
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        # Two-tier obs encoder for frozen image features (L2-norm + dual-tower fusion).
        if self.obs_two_tier:
            observations = self.two_tier_encoder(observations)

        if self.use_film:
            # FiLM: observations condition the network, actions are the input
            assert actions is not None, "FiLM critic requires actions"
            if self.critic_loss_type == 'hlgauss':
                q_logits = self.value_net(actions, observations)
                v = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )
                if return_logits:
                    return v, q_logits
            else:
                v = self.value_net(actions, observations).squeeze(-1)
        else:
            # Standard MLP path
            inputs = [observations]
            if actions is not None and not self.action_repeat:
                inputs.append(actions)
            inputs = jnp.concatenate(inputs, axis=-1)

            if self.critic_loss_type == 'hlgauss':
                if self.action_repeat:
                    q_logits = self.value_net(inputs, actions)
                else:
                    q_logits = self.value_net(inputs)
                v = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )
                if return_logits:
                    return v, q_logits
            elif self.critic_loss_type == 'mse':
                if self.action_repeat:
                    v = self.value_net(inputs, actions).squeeze(-1)
                else:
                    v = self.value_net(inputs).squeeze(-1)
            else:
                raise ValueError(f"Unknown critic loss type: {self.critic_loss_type}")

        return v


class ValueTF(nn.Module):
    hidden_dim: int
    action_dim: int
    action_chunk_size: int = 1
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.0
    conditioning_type: str = 'adaln'  # 'adaln' or 'cross_attn'

    def setup(self):
        assert self.hidden_dim % self.num_heads == 0

        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        class TransformerCritic(nn.Module):
            hidden_dim: int
            num_layers: int
            num_heads: int
            dropout_rate: float
            num_output: int
            conditioning_type: str = 'adaln'

            @nn.compact
            def __call__(self, observations, actions, deterministic=True):
                x = nn.Dense(self.hidden_dim, kernel_init=default_init())(actions)
                cond = nn.Dense(self.hidden_dim, kernel_init=default_init())(observations)

                if self.conditioning_type == 'adaln':
                    for _ in range(self.num_layers):
                        x = AdaLNLayer(
                            self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                            self.dropout_rate
                        )(x, cond, deterministic=deterministic)
                else:  # cross_attn
                    # Critic only has state conditioning (no time), wrap as (B, 1, D)
                    context = cond[:, None, :]
                    for _ in range(self.num_layers):
                        x = CrossAttnLayer(
                            self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                            self.dropout_rate
                        )(x, context, deterministic=deterministic)

                # Pool over sequence: (B, T, hidden_dim) -> (B, hidden_dim)
                x = jnp.mean(x, axis=1)
                return nn.Dense(self.num_output, kernel_init=default_init())(x)

        self.value_net = ensemblize(TransformerCritic, self.num_ensembles,
            in_axes=(None, None),
            out_axes=0,
        )(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers, num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, num_output=num_output,
            conditioning_type=self.conditioning_type,
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, deterministic=True, images=None):
        """Forward pass through the transformer critic."""
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        # Reshape flat actions (B, T*a_dim) -> (B, T, a_dim)
        if actions.ndim == 2:
            actions = actions.reshape(-1, self.action_chunk_size, self.action_dim)

        output = self.value_net(observations, actions, deterministic=deterministic)

        if self.critic_loss_type == 'hlgauss':
            q_logits = output
            v = jnp.sum(jax.nn.softmax(q_logits, axis=-1) * jnp.linspace(self.q_min, self.q_max, self.num_bins), axis=-1)
            if return_logits:
                return v, q_logits
        else:
            v = output.squeeze(-1)
        return v


class ValueSimBa(nn.Module):
    hidden_dims: Sequence[int]
    num_ensembles: int = 2
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    rs_norm_momentum: float = 0.99
    rs_norm_epsilon: float = 1e-8

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        # Create single or ensemble of SimBa MLPs
        mlp_class = SimBaMLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        # Create the value network(s)
        self.value_net = mlp_class(
            hidden_dims=(*self.hidden_dims, num_output),
            rs_norm_momentum=self.rs_norm_momentum,
            rs_norm_epsilon=self.rs_norm_epsilon
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False,
                 is_encoded=False, images=None):
        # Encode observations if needed
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Prepare inputs
        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        x = jnp.concatenate(inputs, axis=-1)

        q_values = self.value_net(x)

        # Handle distributional critic if needed
        if self.critic_loss_type == 'hlgauss':
            if self.num_ensembles > 1:
                # Process each ensemble member
                v_list = []
                for q_logits in q_values:
                    v = jnp.sum(
                        jax.nn.softmax(q_logits, axis=-1) *
                        jnp.linspace(self.q_min, self.q_max, self.num_bins),
                        axis=-1
                    )
                    v_list.append(v)
                v = jnp.stack(v_list, axis=0)
            else:
                v = jnp.sum(
                    jax.nn.softmax(q_values, axis=-1) *
                    jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1
                )

            if return_logits:
                return v, q_values
        else:
            v = q_values.squeeze(-1)

        return v


class ValueMIP(nn.Module):
    hidden_dims: Sequence[int]
    mip_q_noise_scale: float = 1.0
    mip_t_star: float = 0.9
    layer_norm: bool = True
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        # Single MLP for both steps (like MIP actor)
        # Input: [obs, action, scalar_value, t] -> Q-value
        self.mlp = MLP(
            (*self.hidden_dims, num_output),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(
        self,
        observations,
        actions=None,
        scalar_input=None,
        time=None,
        return_logits=False,
        is_encoded=False,
        images=None,
        rng=None,
    ):
        # Encode observations
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        # Handle both 2D (batch, dim) and 3D (batch, chunk, dim) inputs for action chunking
        obs_shape = observations.shape
        batch_size = obs_shape[0]

        # Sample scalar noise if not provided: Uniform[-scale, scale]
        if scalar_input is None:
            if rng is None:
                raise ValueError("Must provide either scalar_input or rng")
            scalar_input = jax.random.uniform(
                rng, (batch_size, 1),
                minval=-self.mip_q_noise_scale,
                maxval=self.mip_q_noise_scale
            )

        # Default time to 0 if not provided
        if time is None:
            if len(obs_shape) == 3:
                chunk_size = obs_shape[1]
                time = jnp.zeros((batch_size, chunk_size, 1))
            else:
                time = jnp.zeros((batch_size, 1))

        # Broadcast scalar_input if needed for action chunking
        if len(obs_shape) == 3:
            chunk_size = obs_shape[1]
            # Broadcast: (batch, 1) -> (batch, chunk, 1)
            if scalar_input.shape[1] == 1:
                scalar_input = jnp.tile(scalar_input[:, None, :], (1, chunk_size, 1))

        # Forward pass: [obs, action, scalar, time] -> Q
        mlp_input = jnp.concatenate([observations, actions, scalar_input, time], axis=-1)
        q_output = self.mlp(mlp_input)

        # Process output based on loss type
        if self.critic_loss_type == 'hlgauss':
            q_logits = q_output
            q_values = jnp.sum(
                jax.nn.softmax(q_logits, axis=-1)
                * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                axis=-1,
            )
            if return_logits:
                return q_values, q_logits
        else:
            q_values = q_output.squeeze(-1)

        return q_values


class ValueMIPEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    num_ensembles: int = 2
    mip_q_noise_scale: float = 1.0
    mip_t_star: float = 0.9
    layer_norm: bool = True
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        # Create MLP class
        mlp_class = MLP

        # Ensemblize if num_ensembles > 1
        # Use in_axes=0 to map over the first dimension of concatenated input
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0, out_axes=0)

        # Create the network (will be vmapped if ensemblized)
        self.mlp = mlp_class(
            (*self.hidden_dims, num_output),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(
        self,
        observations,
        actions=None,
        scalar_input=None,
        time=None,
        return_logits=False,
        is_encoded=False,
        images=None,
        rng=None,
    ):
        # Encode observations
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        # Handle both 2D (batch, dim) and 3D (batch, chunk, dim) inputs for action chunking
        obs_shape = observations.shape
        batch_size = obs_shape[0]

        # Sample scalar noise if not provided: Uniform[-scale, scale]
        if scalar_input is None:
            if rng is None:
                raise ValueError("Must provide either scalar_input or rng")
            scalar_input = jax.random.uniform(
                rng, (batch_size, 1),
                minval=-self.mip_q_noise_scale,
                maxval=self.mip_q_noise_scale
            )

        # Default time to 0 if not provided
        if time is None:
            if len(obs_shape) == 3:
                chunk_size = obs_shape[1]
                time = jnp.zeros((batch_size, chunk_size, 1))
            else:
                time = jnp.zeros((batch_size, 1))

        # Broadcast scalar_input if needed for action chunking
        if len(obs_shape) == 3:
            chunk_size = obs_shape[1]
            # Broadcast: (batch, 1) -> (batch, chunk, 1)
            if scalar_input.ndim == 2 and scalar_input.shape[1] == 1:
                scalar_input = jnp.tile(scalar_input[:, None, :], (1, chunk_size, 1))

        # Prepare inputs for vmapped MLP (expects in_axes=0)
        # Need to reshape all inputs to [num_ensembles, B, D] (or [num_ensembles, B, chunk, D])

        # Handle scalar_input: either [B, 1], [B, num_ensembles, 1], or [B, chunk, 1]
        if scalar_input.ndim == 3:
            # Check if it's from action chunking [B, chunk, 1] or from ensemble [B, num_ensembles, 1]
            if len(obs_shape) == 3:
                # Action chunking case: [B, chunk, 1] -> [num_ensembles, B, chunk, 1] (broadcast)
                scalar_input = jnp.tile(scalar_input[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            elif scalar_input.shape[1] == self.num_ensembles:
                # Step 2 case: [B, num_ensembles, 1] -> [num_ensembles, B, 1]
                scalar_input = jnp.transpose(scalar_input, (1, 0, 2))
            else:
                raise ValueError(f"Unexpected scalar_input shape: {scalar_input.shape}")
        elif scalar_input.ndim == 2:
            # Step 1 case: [B, 1] -> [num_ensembles, B, 1] (broadcast)
            scalar_input = jnp.tile(scalar_input[None, :, :], (self.num_ensembles, 1, 1))
        else:
            raise ValueError(f"scalar_input must be 2D or 3D, got shape: {scalar_input.shape}")

        # Broadcast other inputs to [num_ensembles, B, ...]
        # Handle both 2D and 3D (action chunking) cases
        if observations.ndim == 2:
            observations = jnp.tile(observations[None, :, :], (self.num_ensembles, 1, 1))
            actions = jnp.tile(actions[None, :, :], (self.num_ensembles, 1, 1))
            time = jnp.tile(time[None, :, :], (self.num_ensembles, 1, 1))
        else:  # 3D case for action chunking
            observations = jnp.tile(observations[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            actions = jnp.tile(actions[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            time = jnp.tile(time[None, :, :, :], (self.num_ensembles, 1, 1, 1))

        # Forward pass: [obs, action, scalar, time] -> Q
        mlp_input = jnp.concatenate([observations, actions, scalar_input, time], axis=-1)
        q_output = self.mlp(mlp_input)  # [num_ensembles, B, num_output]

        # Process output based on loss type
        if self.critic_loss_type == 'hlgauss':
            q_logits = q_output  # [num_ensembles, B, num_bins]
            q_values = jnp.sum(
                jax.nn.softmax(q_logits, axis=-1)
                * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                axis=-1,
            )  # [num_ensembles, B]
            if return_logits:
                return q_values, q_logits
        else:
            q_values = q_output.squeeze(-1)  # [num_ensembles, B]

        return q_values


# Export list
__all__ = [
    'default_init',
    'ensemblize',
    'Value',
    'ValueTF',
    'ValueSimBa',
    'ValueMIP',
    'ValueMIPEnsemble',
]
