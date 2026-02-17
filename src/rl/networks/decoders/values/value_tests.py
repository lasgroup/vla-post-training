import sys
import pytest
import jax.numpy as jnp
import flax.nnx as nn
from state_action_value import StateActionValue, StateActionEnsemble
from state_value import StateValue, StateValueEnsemble

# ==========================================
# 3. TEST SUITE
# ==========================================

rngs = nn.Rngs(42)


def test_state_action_value_shape():
    """Test single Q-function output shape."""
    dummy_obs = jnp.zeros(10)
    dummy_act = jnp.zeros(2)

    net = StateActionValue(
        observation=dummy_obs,
        action=dummy_act,
        hidden_dims=[32, 32], rngs=rngs)

    # Batch size 4, State dim 10, Action dim 2
    obs = jnp.zeros((4, 10))
    act = jnp.zeros((4, 2))

    out = net(obs, act)

    # MLP returns (Batch, 1), StateActionValue squeezes to (Batch,)
    assert out.shape == (4,)


def test_state_action_ensemble_shape():
    """Test Q-Ensemble output shape."""
    num_qs = 5
    dummy_obs = jnp.zeros(10)
    dummy_act = jnp.zeros(2)
    net = StateActionEnsemble(
        observation=dummy_obs,
        action=dummy_act,
        hidden_dims=[32, 32], num_qs=num_qs, rngs=rngs)
    obs = jnp.ones((1, 10))
    act = jnp.ones((1, 2))

    out = net(obs, act)
    # Vmap out_axes=0 -> (Num_Qs, Batch)
    assert out.shape == (num_qs, 1)
    assert out.std() > 1e-4


def test_state_value_shape():
    """Test single V-function output shape."""
    dummy_obs = jnp.zeros(10)
    net = StateValue(observation=dummy_obs, hidden_dims=[32], rngs=rngs)

    obs = jnp.zeros((3, 10))
    out = net(obs)

    # MLP returns (Batch, 1), StateValue squeezes to (Batch,)
    assert out.shape == (3,)


def test_state_value_ensemble_shape():
    """Test V-Ensemble output shape."""
    num_vs = 3
    dummy_obs = jnp.zeros(10)
    net = StateValueEnsemble(
        observation=dummy_obs,
        hidden_dims=[32], num_vs=num_vs, rngs=rngs)

    obs = jnp.ones((5, 10))
    out = net(obs)

    # Vmap out_axes=0 -> (Num_Vs, Batch)
    assert out.shape == (num_vs, 5)
    assert out.std() > 1e-4


def test_jit_compatibility():
    """Ensure modules work under JIT."""
    dummy_obs = jnp.zeros(10)
    dummy_act = jnp.zeros(2)
    net = StateActionEnsemble(
        observation=dummy_obs,
        action=dummy_act,
        hidden_dims=[16],
        num_qs=2, rngs=rngs)

    @nn.jit
    def forward(model, s, a):
        return model(s, a)

    obs = jnp.zeros((2, 10))
    act = jnp.zeros((2, 2))

    out = forward(net, obs, act)
    assert out.shape == (2, 2)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))