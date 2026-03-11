import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb
from typing import Any

import src.training.config as _config
from src.rl.networks.mlp import MLP
from src.rl.networks.encoders.encoders import BaseEncoder, ImageEncoder, MLPEncoder
from src.rl.networks.encoders.cnn_encoder import CNNEncoder
from src.rl.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from src.rl.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from src.rl.networks.encoders.resnet_encoderv2 import ResNetv2_18, ResNetv2_34, ResNetv2_Small
from src.rl.dsrl.update_critic import StateActionCriticDef
from src.rl.dsrl.update_actor import PolicyDef
from src.rl.networks.rl_networks import Policy
from src.rl.networks.decoders.values.state_action_value import StateActionEnsembleDecoder
from src.rl.networks.decoders.policies.learned_std_normal_policy import LearnedStdNormalPolicyDecoder, LearnedStdTanhNormalPolicyDecoder
from src.rl.networks.rl_networks import ObsType, ActionType, StateActionCritic

def _get_rl_attr(config: _config.OnlineTrainConfig, name: str, default: Any) -> Any:
    rl = getattr(config, "rl", None)
    if rl is None:
        return default
    return getattr(rl, name, default)

def _build_actor_critic_defs(
    config: _config.OnlineTrainConfig,
    action_low: jax.Array,
    action_high: jax.Array,
    *,
    policy_distribution: str,
) -> tuple[StateActionCriticDef, PolicyDef]:
    critic_encoder_hidden_dims = tuple(_get_rl_attr(config, "critic_encoder_hidden_dims", ()))
    critic_decoder_hidden_dims = tuple(_get_rl_attr(config, "critic_decoder_hidden_dims", (256, 256)))
    policy_decoder_hidden_dims = tuple(_get_rl_attr(config, "policy_decoder_hidden_dims", (256, 256)))
    critic_num_qs = int(_get_rl_attr(config, "critic_num_qs", 2))
    encoder_type = str(_get_rl_attr(config, "encoder_type", "resnet_34_v1")).lower()
    encoder_norm = str(_get_rl_attr(config, "encoder_norm", "group")).lower()
    use_spatial_softmax = bool(_get_rl_attr(config, "use_spatial_softmax", True))
    softmax_temperature = float(_get_rl_attr(config, "softmax_temperature", 1.0))
    image_latent_dim = int(_get_rl_attr(config, "image_latent_dim", 50))
    use_image_bottleneck = bool(_get_rl_attr(config, "use_image_bottleneck", True))
    use_state_branch = bool(_get_rl_attr(config, "use_state_branch", True))

    def _build_image_backbone(
        observation: ObsType,
        image_keys: tuple[str, ...],
        rngs: nnx.Rngs,
    ):
        if encoder_type == "small":
            return CNNEncoder(
                input_example=observation,
                features=(32, 32, 32, 32),
                strides=(2, 1, 1, 1),
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "impala":
            return ImpalaEncoder(
                input_example=observation,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "impala_small":
            return SmallerImpalaEncoder(
                input_example=observation,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_small":
            return ResNetSmall(
                input_example=observation,
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_18_v1":
            return ResNet18(
                input_example=observation,
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_34_v1":
            return ResNet34(
                input_example=observation,
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_small_v2":
            v2_norm = "groupnorm" if encoder_norm == "group" else "batch"
            return ResNetv2_Small(
                input_example=observation,
                norm=v2_norm,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_18_v2":
            v2_norm = "groupnorm" if encoder_norm == "group" else "batch"
            return ResNetv2_18(
                input_example=observation,
                norm=v2_norm,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_34_v2":
            v2_norm = "groupnorm" if encoder_norm == "group" else "batch"
            return ResNetv2_34(
                input_example=observation,
                norm=v2_norm,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        raise ValueError(
            f"Unsupported rl.encoder_type={encoder_type!r}. "
            "Expected one of: "
            "'small', 'impala', 'impala_small', "
            "'resnet_small', 'resnet_18_v1', 'resnet_34_v1', "
            "'resnet_small_v2', 'resnet_18_v2', 'resnet_34_v2'."
        )

    def encoder_def(observation: ObsType, rngs: nnx.Rngs):
        if not isinstance(observation, dict):
            return BaseEncoder(
                dummy_obs=observation,
                mlp_encoder_def=None,
                image_encoder_def=None,
                rngs=rngs,
            )

        state_vector_keys = ["state"]
        # if PREFIX_EMBEDDING_NAME in observation:
        #     state_vector_keys = [PREFIX_EMBEDDING_NAME, "state"]

        use_pixel_encoder = True # TODO: Add from config
        if use_pixel_encoder:
            image_keys = tuple(
                key for key in ("image", "wrist_image", "pixels") if key in observation
            )
            use_pixel_encoder = len(image_keys) > 0
        else:
            image_keys = ()

        mlp_encoder_def = None
        if use_state_branch or not use_pixel_encoder:
            network_def = lambda o, rg: MLP(
                input=o,
                hidden_dims=critic_encoder_hidden_dims,
                activate_final=True,
                rngs=rg,
            )
            mlp_encoder_def = lambda obs, rg: MLPEncoder(
                dummy_obs=obs,
                encoder_def=network_def,
                state_vector_keys=state_vector_keys,
                rngs=rg,
            )

        image_encoder_def = None
        if use_pixel_encoder:
            image_backbone_def = lambda obs, rg: _build_image_backbone(
                observation=obs,
                image_keys=image_keys,
                rngs=rg,
            )
            image_encoder_def = lambda obs, rg: ImageEncoder(
                dummy_obs=obs,
                encoder_def=image_backbone_def,
                latent_dim=image_latent_dim,
                use_bottleneck=use_image_bottleneck,
                rngs=rg,
            )

        return BaseEncoder(
            dummy_obs=observation,
            mlp_encoder_def=mlp_encoder_def,
            image_encoder_def=image_encoder_def,
            rngs=rngs,
        )

    def state_action_decoder_def(
        embedding: jax.Array, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionEnsembleDecoder:
        return StateActionEnsembleDecoder(
            observation=embedding,
            action=action,
            hidden_dims=critic_decoder_hidden_dims,
            num_qs=critic_num_qs,
            rngs=rngs,
        )

    def policy_decoder_def(
        embedding: jax.Array, action: jax.Array, rngs: nnx.Rngs
    ):
        if policy_distribution == "normal":
            return LearnedStdNormalPolicyDecoder(
                observation=embedding,
                action=action,
                hidden_dims=policy_decoder_hidden_dims,
                rngs=rngs,
            )
        if policy_distribution == "tanh_normal":
            return LearnedStdTanhNormalPolicyDecoder(
                observation=embedding,
                action=action,
                hidden_dims=policy_decoder_hidden_dims,
                low=action_low,
                high=action_high,
                rngs=rngs,
            )
        raise ValueError(
            f"Unsupported policy distribution {policy_distribution!r}. "
            "Expected 'normal' or 'tanh_normal'."
        )

    def state_action_critic_def(
        observation: ObsType, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionCritic:
        return StateActionCritic(
            observation=observation,
            action=action,
            encoder_def=encoder_def,
            decoder_def=state_action_decoder_def,
            rngs=rngs,
        )

    def policy_def(
        observation: ObsType, action: ActionType, rngs: nnx.Rngs
    ) -> Policy:
        return Policy(
            observation=observation, 
            action=action, 
            encoder_def=encoder_def,
            decoder_def=policy_decoder_def, 
            rngs=rngs)

    return state_action_critic_def, policy_def
