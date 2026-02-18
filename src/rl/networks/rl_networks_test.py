import jax
import jax.numpy as jnp
import flax.nnx as nn

# Import the networks to test
from src.rl.networks.rl_networks import StateActionCritic, StateValue, Policy
from src.rl.networks.encoders.encoders import BaseEncoder, MLPEncoder
from src.rl.networks.mlp import MLP
from src.rl.networks.decoders.values.state_action_value import StateActionValueDecoder
from src.rl.networks.decoders.values.state_value import StateValueDecoder
from src.rl.networks.decoders.policies.normal_policy import NormalPolicyDecoder


# =========================================
# HELPERS & MOCKS
# =========================================

def create_dummy_obs(batch_size=4):
    return {
        'state': jnp.zeros((batch_size, 10), dtype=jnp.float32),
        'other': jnp.zeros((batch_size, 5), dtype=jnp.float32)
    }


rngs = nn.Rngs(42)


# Simple Encoder Factory for testing (uses MLPEncoder logic)
def simple_encoder_factory(obs_dict, rng):
    # Inner MLP factory
    def mlp_def(flat_input, rg):
        return MLP(input=flat_input, hidden_dims=[16], rngs=rg)

    # Outer MLPEncoder wrapper
    def mlp_encoder_def(o, rg):
        return MLPEncoder(dummy_obs=o, encoder_def=mlp_def, state_vector_keys=['state', 'other'], rngs=rg)

    return BaseEncoder(dummy_obs=obs_dict,
                       mlp_encoder_def=mlp_encoder_def,
                       image_encoder_def=None,
                       rngs=rng)  # No images for this fast test


# =========================================
# 1. STATE ACTION CRITIC TEST (Q-Function)
# =========================================

def test_state_action_critic_integration():
    print("\n[Test] StateActionCritic Integration")
    batch_size = 4
    action_dim = 3
    obs = create_dummy_obs(batch_size)
    actions = jnp.zeros((batch_size, action_dim))

    # Define Decoder Factory
    # The integration class passes (embedding, action) to this factory
    def decoder_factory(embedding, action_input, rg):
        return StateActionValueDecoder(
            observation=embedding,
            action=action_input,
            hidden_dims=[32, 32],
            rngs=rg
        )

    # Instantiate the full Q-network
    q_net = StateActionCritic(
        observation=obs,
        action=actions,
        encoder_def=simple_encoder_factory,
        decoder_def=decoder_factory,
        rngs=rngs
    )
    # Run Forward
    q_values = q_net(obs, actions, training=False)

    print(f"  -> Q-values shape: {q_values.shape}")

    # Check shape: Should be (Batch,) because StateActionValueDecoder squeezes the last dim
    assert q_values.shape == (batch_size,)
    print("  -> Success: Q-network end-to-end.")


# =========================================
# 2. STATE VALUE TEST (V-Function)
# =========================================

def test_state_value_integration():
    print("\n[Test] StateValue Integration")
    batch_size = 4
    obs = create_dummy_obs(batch_size)

    # Define Decoder Factory
    def decoder_factory(embedding, rg):
        return StateValueDecoder(
            observation=embedding,
            hidden_dims=[32],
            rngs=rg
        )

    # Instantiate V-network
    v_net = StateValue(
        observation=obs,
        encoder_def=simple_encoder_factory,
        decoder_def=decoder_factory,
        rngs=rngs
    )

    # Run Forward
    v_values = v_net(obs, training=True)

    print(f"  -> V-values shape: {v_values.shape}")

    # Check shape: (Batch,)
    assert v_values.shape == (batch_size,)
    print("  -> Success: V-network end-to-end.")


# =========================================
# 3. POLICY TEST (Actor)
# =========================================

def test_policy_integration():
    print("\n[Test] Policy Integration")
    batch_size = 4
    action_dim = 3
    obs = create_dummy_obs(batch_size)
    dummy_action = jnp.zeros((batch_size, action_dim))

    # Define Policy Decoder Factory
    def policy_decoder_factory(embedding, action_input, rg):
        return NormalPolicyDecoder(
            observation=embedding,
            action=action_input,
            hidden_dims=[32],
            std=0.1,
            rngs=rg
        )

    # Instantiate Policy
    policy_net = Policy(
        observation=obs,
        action=dummy_action,
        encoder_def=simple_encoder_factory,
        decoder_def=policy_decoder_factory,
        rngs=rngs
    )

    # Run Forward
    dist = policy_net(obs, training=False)

    # Check Output is a Distribution
    assert hasattr(dist, 'sample')
    assert hasattr(dist, 'log_prob')

    # Check Sampling
    samples = dist.sample(seed=jax.random.key(0))
    print(f"  -> Sampled Action shape: {samples.shape}")

    assert samples.shape == (batch_size, action_dim)
    print("  -> Success: Policy network end-to-end.")


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":
    try:
        test_state_action_critic_integration()
        test_state_value_integration()
        test_policy_integration()
        print("\nAll RL Network Integration Tests Passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback

        traceback.print_exc()