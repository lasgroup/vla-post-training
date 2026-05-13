"""SimbaV2-based critic networks for AWR using Hyper* building blocks.

Architecture per SimbaV2:
    input → HyperEmbedder → N × HyperLERPBlock → HyperCategoricalValue

Output: each critic returns (expected_values, log_probs) where
    expected_values: (num_qs/vs, B)          — scalar Q/V per ensemble member
    log_probs:       (num_qs/vs, B, num_bins) — categorical distribution per member
"""
import jax.numpy as jnp
import flax.nnx as nnx
from flax.core.frozen_dict import FrozenDict

from src.rl.networks.rl_networks import ObsType, ActionType
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.rl.networks.simba_hyper import HyperEmbedder, HyperLERPBlock, HyperCategoricalValue


def _prepare_obs_vector(observation: ObsType) -> jnp.ndarray:
    """Concatenate prefix_embedding (if present) and state into a flat vector."""
    if isinstance(observation, (dict, FrozenDict)):
        parts = []
        if PREFIX_EMBEDDING_NAME in observation:
            parts.append(jnp.asarray(observation[PREFIX_EMBEDDING_NAME], dtype=jnp.float32))
        parts.append(jnp.asarray(observation["state"], dtype=jnp.float32))
        return jnp.concatenate(parts, axis=-1)
    return jnp.asarray(observation, dtype=jnp.float32)


class _SimbaV2HyperSingleCritic(nnx.Module):
    """Single SimbaV2 critic: HyperEmbedder → N × HyperLERPBlock → HyperCategoricalValue."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        expansion: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        self.embedder = HyperEmbedder(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            rngs=rngs,
        )
        self.blocks = [
            HyperLERPBlock(
                hidden_dim=hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                expansion=expansion,
                rngs=rngs,
            )
            for _ in range(num_blocks)
        ]
        self.predictor = HyperCategoricalValue(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        expected, info = self.predictor(x)
        return expected, info["log_prob"]


class SimbaV2StateActionCritic(nnx.Module):
    """SimbaV2 Q-critic ensemble.

    Returns:
        expected_values: (num_qs, B)          — expected Q per ensemble member
        log_probs:       (num_qs, B, num_bins) — categorical distribution per member
    """

    def __init__(
        self,
        observation: ObsType,
        action: ActionType,
        hidden_dim: int,
        num_blocks: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        scaler_init: float = 0.0884,   # sqrt(2 / hidden_dim) for hidden_dim=256
        scaler_scale: float = 0.0884,  # sqrt(2 / hidden_dim) for hidden_dim=256
        alpha_init: float = 0.3333,    # 1 / (num_blocks + 1) for num_blocks=2
        alpha_scale: float = 0.0625,   # 1 / sqrt(hidden_dim) for hidden_dim=256
        c_shift: float = 3.0,
        num_qs: int = 2,
        expansion: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        obs_vec = _prepare_obs_vector(observation)
        act_vec = jnp.asarray(action, dtype=jnp.float32).reshape(action.shape[0], -1)
        input_dim = obs_vec.shape[-1] + act_vec.shape[-1]

        @nnx.split_rngs(splits=num_qs)
        @nnx.vmap(out_axes=0, in_axes=0)
        def create_ensemble(rgs: nnx.Rngs) -> _SimbaV2HyperSingleCritic:
            return _SimbaV2HyperSingleCritic(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                c_shift=c_shift,
                expansion=expansion,
                rngs=rgs,
            )

        self.ensemble = create_ensemble(rngs)

    def __call__(
        self, observation: ObsType, action: ActionType, training: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        obs_vec = _prepare_obs_vector(observation)
        act_vec = jnp.asarray(action, dtype=jnp.float32).reshape(action.shape[0], -1)
        x = jnp.concatenate([obs_vec, act_vec], axis=-1)

        @nnx.vmap(in_axes=(0, None), out_axes=0)
        def call_member(model: _SimbaV2HyperSingleCritic, inp: jnp.ndarray):
            return model(inp)

        return call_member(self.ensemble, x)  # ((num_qs, B), (num_qs, B, num_bins))


class SimbaV2StateValue(nnx.Module):
    """SimbaV2 V-critic ensemble.

    Returns:
        expected_values: (num_vs, B)          — expected V per ensemble member
        log_probs:       (num_vs, B, num_bins) — categorical distribution per member
    """

    def __init__(
        self,
        observation: ObsType,
        hidden_dim: int,
        num_blocks: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        scaler_init: float = 0.0884,   # sqrt(2 / hidden_dim) for hidden_dim=256
        scaler_scale: float = 0.0884,  # sqrt(2 / hidden_dim) for hidden_dim=256
        alpha_init: float = 0.3333,    # 1 / (num_blocks + 1) for num_blocks=2
        alpha_scale: float = 0.0625,   # 1 / sqrt(hidden_dim) for hidden_dim=256
        c_shift: float = 3.0,
        num_vs: int = 2,
        expansion: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        obs_vec = _prepare_obs_vector(observation)
        input_dim = obs_vec.shape[-1]

        @nnx.split_rngs(splits=num_vs)
        @nnx.vmap(out_axes=0, in_axes=0)
        def create_ensemble(rgs: nnx.Rngs) -> _SimbaV2HyperSingleCritic:
            return _SimbaV2HyperSingleCritic(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                c_shift=c_shift,
                expansion=expansion,
                rngs=rgs,
            )

        self.ensemble = create_ensemble(rngs)

    def __call__(self, observation: ObsType, training: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        obs_vec = _prepare_obs_vector(observation)

        @nnx.vmap(in_axes=(0, None), out_axes=0)
        def call_member(model: _SimbaV2HyperSingleCritic, inp: jnp.ndarray):
            return model(inp)

        return call_member(self.ensemble, obs_vec)  # ((num_vs, B), (num_vs, B, num_bins))
