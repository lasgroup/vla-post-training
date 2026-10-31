"""Tests verifying that Q-value-based advantages produce real gradient updates
to the actor network during online RL.

OGPO uses REINFORCE/PPO-style policy gradients: advantages (from Q-values) are
scalar weights that multiply d(log pi)/d(theta). The Q-values determine *which
direction* to push the policy, but the gradient signal comes from the policy's
own log-probability. These tests verify that:

1. Actor weights actually change after an update step.
2. Different Q-values produce different actor updates (the critic matters).
3. The frozen-encoder path still produces valid actor gradients.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Force CPU to avoid GPU OOM from concurrent training jobs
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import copy
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from ogpo.agents.ogpo import OGPOAgent
from ogpo.utils.config_loader import AttrDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Minimal OGPO config for a tiny low-dim agent."""
    cfg = dict(
        agent_name='ogpo',
        # dims (set at runtime)
        ob_dims=None,
        action_dim=None,
        q_min=-10.0,
        q_max=10.0,
        act_min=-1.0,
        act_max=1.0,
        proprio_dim=None,
        full_act_dim=None,
        # architecture
        actor_hidden_dims=[64, 64],
        value_hidden_dims=[64, 64],
        layer_norm=True,
        actor_layer_norm=False,
        num_qs=2,
        encoder=None,
        actor_obs='state',
        critic_obs='state',
        _original_critic_obs='state',
        actor_backbone='mlp',
        critic_backbone='mlp',
        useSimBa=False,
        useFiLM=False,
        useTF=False,
        # flow matching
        flow_steps=4,
        ft_flow_steps=0,
        policy_type='flow',
        mip_t_star=0.999,
        horizon_length=2,
        action_chunking=True,
        clip_intermediate_actions=True,
        denoised_clip_value=1.0,
        use_shortcut=False,
        shortcut_inference_steps=4,
        # denoiser / noise
        use_denoiser=False,
        use_constant_noise=True,
        constant_noise_std=0.01,
        min_noise_std=0.01,
        max_noise_std=0.01,
        # learning
        lr=3e-4,
        actor_lr=3e-4,
        critic_lr=3e-4,
        ppo_lr=3e-4,
        clip_grad_norm=1000.0,
        actor_scheduler='constant',
        critic_scheduler='constant',
        use_constant_scheduler_for_bc=True,
        actor_warmup_steps=0,
        actor_decay_steps=1,
        actor_end_value=3e-4,
        critic_warmup_steps=0,
        critic_decay_steps=1,
        critic_end_value=3e-4,
        actor_weight_decay=0.0,
        critic_weight_decay=0.0,
        use_muon=False,
        # target
        tau=0.05,
        actor_tau=0.05,
        # PPO / GRPO
        batch_size=8,
        ppo_batch_size=8,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        group_num_samples=4,
        use_bc_regularization=False,
        bc_coeff=0.0,
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        normalize_group=False,
        clip_min_epsilon_multiplier=1.0,
        clip_max_epsilon_multiplier=1.0,
        adv_clip_min=None,
        adv_strategy='vanilla',
        awr_beta=0.05,
        # misc
        discount=0.99,
        best_of_n=1,
        subsample_bon=False,
        randn_clip_value=3.0,
        q_agg='min',
        calql_q_subsample=2,
        cql_alpha=0.0,
        q_variance_reduction=False,
        q_vr_num_samples=4,
        q_vr_reduction='mean',
        mc_regression=False,
        mc_regression_coeff=1.0,
        use_success_buffer_q=False,
        error_correct_ode_to_sde=False,  # renamed; kept off (legacy FPO+denoiser path)
        use_fpo=False,
        fpo_n_mc=4,
        fpo_per_sample=True,
        fpo_use_aspo=False,
        fpo_zero_sampling=False,
        fpo_cfm_loss_clamp=100.0,
        fpo_cfm_diff_clamp=5.0,
        # critic loss
        critic_loss_type='mse',
        num_bins=256,
        action_repeat=False,
        # vision (not used in lowdim tests)
        vit_encoder='minvit',
        vit_state_proj_dim=0,
        vit_img_proj_dim=0,
        freeze_vision_encoder=False,
    )
    cfg.update(overrides)
    return AttrDict(cfg)


def _create_agent(obs_dim=10, act_dim=4, seed=42, **config_overrides):
    """Create a tiny OGPO agent for testing."""
    config = _make_config(**config_overrides)
    ex_obs = np.random.randn(obs_dim).astype(np.float32)
    ex_act = np.random.randn(act_dim).astype(np.float32)
    agent = OGPOAgent.create(seed, ex_obs, ex_act, config, adv_clip_min=None)
    return agent


def _make_batch(agent, obs_dim=10, act_dim=4, batch_size=8, horizon=2,
                full_state_dim=None):
    """Create a fake training batch matching what batch_update expects.

    Returns (q_batch, pi_batch, success_batch, success_flag) where
    q_batch and pi_batch have shape [1, B, ...] (UTD=1).
    """
    B = batch_size
    T = horizon
    utd = 1

    def randn(*shape):
        return np.random.randn(*shape).astype(np.float32)

    obs = randn(utd, B, obs_dim)
    acts = randn(utd, B, T, act_dim)
    next_obs = randn(utd, B, obs_dim)
    rewards = np.zeros((utd, B, T), dtype=np.float32)
    masks = np.ones((utd, B, T), dtype=np.float32)
    terminals = np.zeros((utd, B, T), dtype=np.float32)
    valid = np.ones((utd, B, T), dtype=np.float32)

    batch = dict(
        observations=obs,
        next_observations=next_obs,
        actions=acts,
        rewards=rewards,
        masks=masks,
        terminals=terminals,
        valid=valid,
    )
    if full_state_dim is not None:
        batch['full_states'] = randn(utd, B, full_state_dim)
        batch['next_full_states'] = randn(utd, B, full_state_dim)

    # success_batch has same structure
    success_batch = copy.deepcopy(batch)
    success_flag = jnp.array(False)

    return batch, batch, success_batch, success_flag


def _get_actor_params(agent):
    """Extract a flat vector of all actor params for comparison."""
    leaves = jax.tree_util.tree_leaves(agent.actor_network.params['modules_actor'])
    return jnp.concatenate([l.ravel() for l in leaves])


def _get_critic_params(agent):
    """Extract a flat vector of all critic params for comparison."""
    leaves = jax.tree_util.tree_leaves(agent.critic_network.params['modules_critic'])
    return jnp.concatenate([l.ravel() for l in leaves])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActorWeightsChange:
    """Verify actor parameters actually change after a batch_update."""

    def test_actor_params_change_after_update(self):
        """After one batch_update, actor params must differ from before."""
        agent = _create_agent()
        params_before = _get_actor_params(agent)

        q_batch, pi_batch, success_batch, sflag = _make_batch(agent)
        agent_after, info = agent.batch_update(q_batch, pi_batch, success_batch, sflag)

        params_after = _get_actor_params(agent_after)
        diff = jnp.abs(params_after - params_before).max()
        assert diff > 1e-8, (
            f"Actor params did not change after batch_update (max diff={diff:.2e}). "
            "Gradient flow from Q-values to actor may be broken."
        )

    def test_critic_params_change_after_update(self):
        """Sanity check: critic should also update."""
        agent = _create_agent()
        params_before = _get_critic_params(agent)

        q_batch, pi_batch, success_batch, sflag = _make_batch(agent)
        agent_after, _ = agent.batch_update(q_batch, pi_batch, success_batch, sflag)

        params_after = _get_critic_params(agent_after)
        diff = jnp.abs(params_after - params_before).max()
        assert diff > 1e-8, (
            f"Critic params did not change (max diff={diff:.2e})."
        )


class TestQValuesInfluenceActor:
    """Different critic Q-values should produce different actor updates.

    Since advantages = f(Q), and loss = -E[adv * ratio], different Q-values
    should yield different gradient directions, hence different final params.
    """

    def test_different_critics_produce_different_actor_updates(self):
        """Two agents with identical actors but different critics should
        diverge after one update step."""
        agent_a = _create_agent(seed=42)
        agent_b = _create_agent(seed=42)

        # Verify actors start identical
        pa = _get_actor_params(agent_a)
        pb = _get_actor_params(agent_b)
        assert jnp.allclose(pa, pb), "Actors should start identical"

        # Perturb agent_b's critic so Q-values differ
        critic_params_b = agent_b.critic_network.params
        perturbed = jax.tree_util.tree_map(
            lambda x: x + 5.0 * jax.random.normal(jax.random.PRNGKey(99), x.shape),
            critic_params_b
        )
        agent_b = agent_b.replace(
            critic_network=agent_b.critic_network.replace(params=perturbed)
        )

        # Same batch for both
        np.random.seed(123)
        q_batch, pi_batch, success_batch, sflag = _make_batch(agent_a)

        agent_a2, _ = agent_a.batch_update(q_batch, pi_batch, success_batch, sflag)
        agent_b2, _ = agent_b.batch_update(q_batch, pi_batch, success_batch, sflag)

        pa2 = _get_actor_params(agent_a2)
        pb2 = _get_actor_params(agent_b2)
        diff = jnp.abs(pa2 - pb2).max()
        assert diff > 1e-6, (
            f"Actors updated identically despite different critics (max diff={diff:.2e}). "
            "Q-values may not be influencing the actor update."
        )


class TestActorLossGradient:
    """Directly test that jax.grad of actor_total_loss produces nonzero grads."""

    def test_actor_loss_has_nonzero_grads(self):
        """Compute jax.grad of the actor loss w.r.t. actor params and check nonzero."""
        agent = _create_agent()

        # Make a single-step batch (no UTD reshape)
        B = 8
        obs_dim, act_dim, T = 10, 4, 2
        batch = dict(
            observations=np.random.randn(B, obs_dim).astype(np.float32),
            next_observations=np.random.randn(B, obs_dim).astype(np.float32),
            actions=np.random.randn(B, T, act_dim).astype(np.float32),
            rewards=np.zeros((B, T), dtype=np.float32),
            masks=np.ones((B, T), dtype=np.float32),
            terminals=np.zeros((B, T), dtype=np.float32),
            valid=np.ones((B, T), dtype=np.float32),
        )
        batch_success = copy.deepcopy(batch)
        success_flag = jnp.array(False)
        rng = jax.random.PRNGKey(0)

        def loss_fn(params):
            return agent.actor_total_loss(batch, batch_success, params, success_flag, rng)

        grads, info = jax.grad(loss_fn, has_aux=True)(agent.actor_network.params)

        # Check that at least some gradients are nonzero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        max_grad = max(jnp.abs(g).max() for g in grad_leaves)
        assert max_grad > 1e-10, (
            f"All actor gradients are ~zero (max={max_grad:.2e}). "
            "The advantage-weighted policy gradient may not be flowing."
        )


class TestFrozenEncoderGradientFlow:
    """When _encoder_frozen=True, observations are pre-encoded embeddings.
    The actor should still receive valid gradients through log pi(a|s)."""

    def test_frozen_encoder_actor_updates(self):
        """Simulate frozen encoder: obs are 512D embeddings, actor_obs/critic_obs='state'."""
        embed_dim = 64  # small for testing
        act_dim = 4
        agent = _create_agent(
            obs_dim=embed_dim,
            act_dim=act_dim,
            _encoder_frozen=True,
            _original_critic_obs='state',
        )
        params_before = _get_actor_params(agent)

        # Batch with pre-encoded obs (like frozen encoder produces)
        q_batch, pi_batch, success_batch, sflag = _make_batch(
            agent, obs_dim=embed_dim, act_dim=act_dim
        )
        agent_after, _ = agent.batch_update(q_batch, pi_batch, success_batch, sflag)

        params_after = _get_actor_params(agent_after)
        diff = jnp.abs(params_after - params_before).max()
        assert diff > 1e-8, (
            f"Actor params did not change with frozen encoder (max diff={diff:.2e}). "
            "Frozen encoder may be blocking gradient flow."
        )

    def test_frozen_encoder_mixed_obs_critic_uses_full_states(self):
        """With _original_critic_obs='state', critic should use full_states from batch."""
        embed_dim = 64
        full_state_dim = 20
        act_dim = 4
        agent = _create_agent(
            obs_dim=embed_dim,
            act_dim=act_dim,
            _encoder_frozen=True,
            _original_critic_obs='state',
            ex_full_states_dim=full_state_dim,
        )

        batch = dict(
            observations=np.random.randn(8, embed_dim).astype(np.float32),
            full_states=np.random.randn(8, full_state_dim).astype(np.float32),
        )
        critic_obs = agent._get_critic_obs(batch)
        assert critic_obs.shape[-1] == full_state_dim, (
            f"Expected critic to use full_states ({full_state_dim}D) but got {critic_obs.shape[-1]}D"
        )

    def test_frozen_encoder_both_image_critic_uses_observations(self):
        """With _original_critic_obs='image', critic should use observations (embeddings)."""
        embed_dim = 64
        full_state_dim = 20
        act_dim = 4
        agent = _create_agent(
            obs_dim=embed_dim,
            act_dim=act_dim,
            _encoder_frozen=True,
            _original_critic_obs='image',
        )

        batch = dict(
            observations=np.random.randn(8, embed_dim).astype(np.float32),
            full_states=np.random.randn(8, full_state_dim).astype(np.float32),
        )
        critic_obs = agent._get_critic_obs(batch)
        assert critic_obs.shape[-1] == embed_dim, (
            f"Expected critic to use observations ({embed_dim}D) but got {critic_obs.shape[-1]}D"
        )


class TestMultipleUpdatesConverge:
    """Run multiple update steps and verify the actor loss changes
    (indicating the Q-based advantages are steering the optimization)."""

    def test_loss_changes_over_updates(self):
        """Actor loss should not be static across multiple updates."""
        agent = _create_agent()
        np.random.seed(0)

        losses = []
        for _ in range(3):
            q_batch, pi_batch, success_batch, sflag = _make_batch(agent)
            agent, info = agent.batch_update(q_batch, pi_batch, success_batch, sflag)
            # actor_total_loss is keyed as actor/* in info
            actor_loss = info.get('actor/total_loss', info.get('actor/pg_loss', None))
            if actor_loss is not None:
                losses.append(float(actor_loss))

        if len(losses) >= 2:
            # At least one loss should differ from the first
            diffs = [abs(l - losses[0]) for l in losses[1:]]
            assert max(diffs) > 1e-8, (
                f"Actor loss remained constant across updates: {losses}. "
                "Q-based advantages may not be influencing optimization."
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
