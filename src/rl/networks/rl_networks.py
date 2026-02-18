from typing import Callable, Dict, Union
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import flax.nnx as nn

from src.rl.networks.encoders.encoders import BaseEncoder
from src.rl.networks.decoders.values.state_action_value import (
    StateActionValueDecoder,
    StateActionEnsembleDecoder,
)
from src.rl.networks.decoders.values.state_value import (
    StateValueDecoder,
    StateValueEnsembleDecoder,
)
from src.rl.networks.decoders.policies.normal_policy import NormalPolicyDecoder
from src.rl.networks.decoders.policies.learned_std_normal_policy import (
    LearnedStdNormalPolicyDecoder,
    LearnedStdTanhNormalPolicyDecoder,
)
from tensorflow_probability.substrates import jax as tfp

# TFP aliases
tfd = tfp.distributions

ObsType = Union[Dict, FrozenDict, jnp.ndarray]
EmbeddingType = jnp.ndarray
ActionType = jnp.ndarray

StateActionValueDecoderType = Union[StateActionValueDecoder, StateActionEnsembleDecoder]
StateValueDecoderType = Union[StateValueDecoder, StateValueEnsembleDecoder]
PolicyDecoderDef = Union[
    NormalPolicyDecoder,
    LearnedStdNormalPolicyDecoder,
    LearnedStdTanhNormalPolicyDecoder,
]

EncoderDef = Callable[[ObsType, nn.Rngs], BaseEncoder]
StateActionDecoderDef = Callable[
    [EmbeddingType, ActionType, nn.Rngs], StateActionValueDecoderType
]
StateValueDecoderDef = Callable[[EmbeddingType, nn.Rngs], StateValueDecoderType]
PolicyDecoderDef = Callable[[EmbeddingType, ActionType, nn.Rngs], PolicyDecoderDef]


class StateActionCritic(nn.Module):
    def __init__(
        self,
        observation: ObsType,
        action: ActionType,
        encoder_def: EncoderDef,
        decoder_def: StateActionDecoderDef,
        rngs: nn.Rngs,
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


class StateValue(nn.Module):
    def __init__(
        self,
        observation: ObsType,
        encoder_def: EncoderDef,
        decoder_def: StateValueDecoderDef,
        rngs: nn.Rngs,
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


class Policy(nn.Module):
    def __init__(
        self,
        observation: ObsType,
        action: ActionType,
        encoder_def: EncoderDef,
        decoder_def: PolicyDecoderDef,
        rngs: nn.Rngs,
    ):
        self.encoder = encoder_def(observation, rngs)
        dummy_embedding = self.encoder(observation)
        self.policy_decoder = decoder_def(dummy_embedding, action, rngs)

    def __call__(
        self, observation: ObsType, training: bool = False
    ) -> tfd.Distribution:
        embedding = self.encoder(observation, training=training)
        policy_dist = self.policy_decoder(observations=embedding, training=training)
        return policy_dist
