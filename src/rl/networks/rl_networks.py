from typing import Callable, Union
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import flax.nnx as nnx

from src.rl.networks.encoders.encoders import BaseEncoder
from src.rl.networks.decoders.values.state_action_value import (
    StateActionValueDecoder,
    StateActionEnsembleDecoder,
)
from src.rl.networks.decoders.values.state_value import (
    StateValueDecoder,
    StateValueEnsembleDecoder,
)

ObsType = Union[dict, FrozenDict, jnp.ndarray]
# Key of the cached VLA prefix embedding in critic observations and replay-buffer entries.
PREFIX_EMBEDDING_NAME = "prefix_embedding"
EmbeddingType = jnp.ndarray
ActionType = jnp.ndarray

StateActionValueDecoderType = Union[StateActionValueDecoder, StateActionEnsembleDecoder]
StateValueDecoderType = Union[StateValueDecoder, StateValueEnsembleDecoder]

EncoderDef = Callable[[ObsType, nnx.Rngs], BaseEncoder]
StateActionDecoderDef = Callable[
    [EmbeddingType, ActionType, nnx.Rngs], StateActionValueDecoderType
]
StateValueDecoderDef = Callable[[EmbeddingType, nnx.Rngs], StateValueDecoderType]


class StateActionCritic(nnx.Module):
    def __init__(
        self,
        observation: ObsType,
        action: ActionType,
        encoder_def: EncoderDef,
        decoder_def: StateActionDecoderDef,
        rngs: nnx.Rngs,
    ):
        self.encoder = encoder_def(observation, rngs)
        dummy_embedding = self.encoder(observation)
        self.state_action_decoder = decoder_def(
            dummy_embedding,
            action,
            rngs,
        )

    def __call__(
        self, observation: ObsType, action: ActionType, training: bool = False
    ) -> jnp.ndarray:
        embedding = self.encoder(observation, training=training)
        q = self.state_action_decoder(
            observations=embedding, actions=action, training=training
        )
        return q


class StateValue(nnx.Module):
    def __init__(
        self,
        observation: ObsType,
        encoder_def: EncoderDef,
        decoder_def: StateValueDecoderDef,
        rngs: nnx.Rngs,
    ):
        self.encoder = encoder_def(observation, rngs)
        dummy_embedding = self.encoder(observation)
        self.state_value_decoder = decoder_def(
            dummy_embedding,
            rngs,
        )

    def __call__(self, observation: ObsType, training: bool = False) -> jnp.ndarray:
        embedding = self.encoder(observation, training=training)
        v = self.state_value_decoder(observations=embedding, training=training)
        return v
