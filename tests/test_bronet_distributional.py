"""Validates that the BroNet critic works both with and without the distributional
(categorical / C51) critic, matching the MLP decoder's output convention.

Run on a compute node:
    .venv/bin/python tests/test_bronet_distributional.py

What it checks:
  1. Scalar mode (num_bins=1): BroNet returns (ensemble, B) — backward compatible
     with the existing AWR scalar path.
  2. Distributional mode (num_bins=K): BroNet returns (ensemble, B, K) logits,
     identical in shape to StateActionEnsembleDecoder / StateValueEnsembleDecoder.
  3. The (ensemble, B, K) logits flow correctly through the C51 helpers used in
     src/rl/best_of_n/update_critic.py (reduce_ensemble_probs, categorical_project,
     make_value_distribution) and produce valid probability distributions.
  4. BroNet and the MLP decoder produce the same output shapes for the same config,
     so update_critic.py's architecture-agnostic loss works for both.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

from src.rl.networks.bronet_critic import BroNetStateActionCritic, BroNetStateValue
from src.rl.networks.decoders.values.state_action_value import StateActionEnsembleDecoder
from src.rl.networks.decoders.values.state_value import StateValueEnsembleDecoder
from src.rl.value_distribution import (
    make_value_distribution,
    make_bin_centers,
    reduce_ensemble_probs,
    categorical_project,
)

B, S, A = 4, 7, 3
NUM_QS, NUM_VS = 2, 2
K = 51
LOWER, UPPER = -5.0, 0.0


def _obs_act():
    return {"state": jnp.ones((B, S))}, jnp.ones((B, A))


def test_scalar_mode_shapes():
    """num_bins=1: scalar ensemble output, unchanged from the original BroNet."""
    obs, act = _obs_act()
    q = BroNetStateActionCritic(observation=obs, action=act, hidden_dim=32, depth=2,
                                num_qs=NUM_QS, rngs=nnx.Rngs(1))
    v = BroNetStateValue(observation=obs, hidden_dim=32, depth=2,
                         num_vs=NUM_VS, rngs=nnx.Rngs(2))
    qo, vo = q(obs, act), v(obs)
    assert qo.shape == (NUM_QS, B), qo.shape
    assert vo.shape == (NUM_VS, B), vo.shape
    print("[ok] scalar mode  : Q", qo.shape, "V", vo.shape)


def test_distributional_mode_shapes():
    """num_bins=K: categorical logits, same shape as the MLP decoder."""
    obs, act = _obs_act()
    q = BroNetStateActionCritic(observation=obs, action=act, hidden_dim=32, depth=2,
                                num_qs=NUM_QS, num_bins=K, rngs=nnx.Rngs(1))
    v = BroNetStateValue(observation=obs, hidden_dim=32, depth=2,
                         num_vs=NUM_VS, num_bins=K, rngs=nnx.Rngs(2))
    qo, vo = q(obs, act), v(obs)
    assert qo.shape == (NUM_QS, B, K), qo.shape
    assert vo.shape == (NUM_VS, B, K), vo.shape
    print("[ok] distrib mode : Q", qo.shape, "V", vo.shape)


def test_bronet_matches_mlp_decoder_shapes():
    """BroNet and the MLP decoder must agree on output shape so update_critic.py
    can stay architecture-agnostic."""
    obs, act = _obs_act()
    # Encoders feed embeddings to the MLP decoders; here we just feed the raw
    # state vector as the embedding to compare the *decoder* head shapes.
    emb = obs["state"]
    for num_bins in (1, K):
        bro_q = BroNetStateActionCritic(observation=obs, action=act, hidden_dim=32,
                                        depth=2, num_qs=NUM_QS, num_bins=num_bins,
                                        rngs=nnx.Rngs(1))(obs, act)
        mlp_q = StateActionEnsembleDecoder(observation=emb, action=act,
                                           hidden_dims=(32, 32), num_qs=NUM_QS,
                                           num_bins=num_bins, rngs=nnx.Rngs(3))(emb, act)
        assert bro_q.shape == mlp_q.shape, (num_bins, bro_q.shape, mlp_q.shape)

        bro_v = BroNetStateValue(observation=obs, hidden_dim=32, depth=2,
                                 num_vs=NUM_VS, num_bins=num_bins,
                                 rngs=nnx.Rngs(2))(obs)
        mlp_v = StateValueEnsembleDecoder(observation=emb, hidden_dims=(32, 32),
                                          num_vs=NUM_VS, num_bins=num_bins,
                                          rngs=nnx.Rngs(4))(emb)
        assert bro_v.shape == mlp_v.shape, (num_bins, bro_v.shape, mlp_v.shape)
        print(f"[ok] bronet==mlp  : num_bins={num_bins} Q{bro_q.shape} V{bro_v.shape}")


def test_c51_flow_through_helpers():
    """Exercise the exact C51 path update_critic.py runs on BroNet logits."""
    obs, act = _obs_act()
    q = BroNetStateActionCritic(observation=obs, action=act, hidden_dim=32, depth=2,
                                num_qs=NUM_QS, num_bins=K, rngs=nnx.Rngs(1))
    v = BroNetStateValue(observation=obs, hidden_dim=32, depth=2,
                         num_vs=NUM_VS, num_bins=K, rngs=nnx.Rngs(2))
    q_logits, v_logits = q(obs, act), v(obs)

    centers = make_bin_centers(LOWER, UPPER, K)
    reward = jnp.array([-1.0, 0.0, -1.0, 0.0])
    discount = jnp.array([0.99, 0.0, 0.99, 0.0])  # discount=0 => terminal

    # Q-step target: reduce V ensemble -> project under Bellman op -> CE vs Q.
    next_probs = reduce_ensemble_probs(v_logits, "mean", centers)
    target_probs = categorical_project(next_probs, reward, discount, centers)
    assert next_probs.shape == (B, K)
    assert target_probs.shape == (B, K)
    assert np.allclose(np.array(next_probs).sum(-1), 1.0, atol=1e-5)
    assert np.allclose(np.array(target_probs).sum(-1), 1.0, atol=1e-4)

    q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
    td_loss = -jnp.mean(jnp.sum(target_probs[jnp.newaxis] * q_logprobs, axis=-1))
    assert jnp.isfinite(td_loss)

    # "min" reduction must also stay a valid distribution.
    min_probs = reduce_ensemble_probs(v_logits, "min", centers)
    assert np.allclose(np.array(min_probs).sum(-1), 1.0, atol=1e-5)

    # Inference path: expected value per ensemble member.
    q_dist = make_value_distribution(q_logits, K, LOWER, UPPER, "one_hot")
    assert q_dist.mean().shape == (NUM_QS, B)
    print("[ok] c51 flow     : td_loss", float(td_loss), "E[Q]", q_dist.mean().shape)


def test_terminal_projection_is_reward_only():
    """When discount=0 the projected distribution should concentrate on reward."""
    centers = make_bin_centers(LOWER, UPPER, K)
    next_probs = jnp.ones((1, K)) / K  # arbitrary next distribution
    reward = jnp.array([-2.0])
    discount = jnp.array([0.0])
    proj = categorical_project(next_probs, reward, discount, centers)
    # Expected value of projected dist should equal the reward (Tz = r).
    ev = float(jnp.sum(proj[0] * centers))
    assert abs(ev - (-2.0)) < 0.1, ev
    print("[ok] terminal proj: E[Tz]=", ev, "(expected -2.0)")


if __name__ == "__main__":
    test_scalar_mode_shapes()
    test_distributional_mode_shapes()
    test_bronet_matches_mlp_decoder_shapes()
    test_c51_flow_through_helpers()
    test_terminal_projection_is_reward_only()
    print("\nALL BRONET x DISTRIBUTIONAL CHECKS PASSED")
