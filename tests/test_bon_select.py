"""
Tests for the pi05 BoN-only change (docs/changes/2026-10-09-pi05-bon-only/):
  - q_helper.select_best_of_n vs a verbatim copy of the pre-change inline rule in
    OGPOPi05Agent.sample_actions_BON (subsample / min), and mean by hand
  - unclipped Euler ODE (sample_flow_actions_ode with +-inf bounds) vs a hand Euler loop
  - ogpo.yaml defaults of the new keys

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_bon_select.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.modules.pg_helper import sample_flow_actions_ode
from ogpo.agents.modules.q_helper import select_best_of_n


def _pre_change_inline(q_values, subsample_bon, sub_rng, num_qs):
    """Verbatim copy of ogpo/agents/ogpo_pi05.py:577-582 at 021fb48 (B, N from q_values)."""
    _, B, N = q_values.shape
    if subsample_bon:
        idxs = jax.random.randint(sub_rng, (2, B, N), 0, num_qs)
        min_qs = jnp.take_along_axis(q_values, idxs, axis=0).min(axis=0)
    else:
        min_qs = q_values.min(axis=0)
    best = jnp.argmax(min_qs, axis=1)
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_select_matches_pre_change_subsample(seed):
    num_qs, B, N = 10, 16, 32
    q = jax.random.normal(jax.random.PRNGKey(seed), (num_qs, B, N))
    rng = jax.random.PRNGKey(100 + seed)
    new = select_best_of_n(q, 'subsample', rng)
    old = _pre_change_inline(q, True, rng, num_qs)
    assert new.shape == (B,)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


@pytest.mark.parametrize("seed", [0, 1])
def test_select_matches_pre_change_min(seed):
    q = jax.random.normal(jax.random.PRNGKey(seed), (10, 8, 32))
    new = select_best_of_n(q, 'min', None)
    old = _pre_change_inline(q, False, None, 10)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


def test_select_mean_by_hand():
    q = np.random.default_rng(0).normal(size=(10, 8, 32)).astype(np.float32)
    new = select_best_of_n(jnp.asarray(q), 'mean')
    np.testing.assert_array_equal(np.asarray(new), q.mean(axis=0).argmax(axis=1))


def test_select_mean_differs_from_min():
    # One head strongly disfavours candidate 0, which wins on the mean: min and mean must disagree.
    q = np.zeros((3, 1, 2), np.float32)
    q[:, 0, 0] = [10.0, 10.0, -5.0]   # mean 5.0, min -5.0
    q[:, 0, 1] = [1.0, 1.0, 1.0]      # mean 1.0, min 1.0
    assert int(select_best_of_n(jnp.asarray(q), 'mean')[0]) == 0
    assert int(select_best_of_n(jnp.asarray(q), 'min')[0]) == 1


def test_select_bad_method_and_missing_rng():
    q = jnp.zeros((2, 1, 3))
    with pytest.raises(ValueError):
        select_best_of_n(q, 'max')
    with pytest.raises(ValueError):
        select_best_of_n(q, 'subsample', None)


def _linear_actor(obs, x, t, is_encoded=True):
    # Pushes samples well outside [-1, 1] so a final clip would be visible.
    return 3.0 * x + 2.0 + t


def _hand_euler(x0, flow_steps):
    """Euler loop with the helper's op order (x + v * dt), so equality can be exact."""
    x = x0
    dt = 1.0 / flow_steps
    for i in range(flow_steps):
        t = jnp.full((x.shape[0], 1), i / flow_steps)
        x = x + _linear_actor(None, x, t) * dt
    return x


def test_ode_unclipped_equals_hand_euler():
    B, D, steps = 6, 70, 10
    rng = jax.random.PRNGKey(3)
    out = sample_flow_actions_ode(
        actor_fn=_linear_actor, observations=jnp.zeros((B, 1)), rng=rng, flow_steps=steps, act_dim=D,
        act_min=-jnp.inf, act_max=jnp.inf, clip_intermediate=False, clip_value=1.0, is_encoded=True)
    ref = _hand_euler(jax.random.normal(rng, (B, D)), steps)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=0, atol=0)
    assert float(jnp.abs(out).max()) > 1.0  # nothing clipped


def test_ode_clipped_path_unchanged():
    B, D, steps = 4, 70, 10
    rng = jax.random.PRNGKey(5)
    out = sample_flow_actions_ode(
        actor_fn=_linear_actor, observations=jnp.zeros((B, 1)), rng=rng, flow_steps=steps, act_dim=D,
        act_min=-1.0, act_max=1.0, clip_intermediate=False, clip_value=1.0, is_encoded=True)
    ref = jnp.clip(_hand_euler(jax.random.normal(rng, (B, D)), steps), -1.0, 1.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_new_config_defaults():
    from ogpo.utils.config_loader import config_to_agent_dict, load_config
    config = load_config('ogpo', {})
    assert config['training']['freeze_actor'] is False
    assert config['training']['first_round_policy'] == 'bon'
    agent = config_to_agent_dict(config)
    assert agent['pi05_bon_sampler'] == 'sde'
    assert agent['pi05_bon_agg'] == 'subsample'
    assert agent['subsample_bon'] is True  # common.yaml; default pi05 rule is the same 'subsample'


# --- verifier additions: edge shapes, ties, jit ---------------------------------------------------------------

@pytest.mark.parametrize("num_qs,B,N", [(2, 1, 32), (2, 5, 1), (10, 1, 1), (3, 7, 2)])
@pytest.mark.parametrize("seed", [0, 1])
def test_select_edge_shapes_match_pre_change(num_qs, B, N, seed):
    q = jax.random.normal(jax.random.PRNGKey(seed), (num_qs, B, N))
    rng = jax.random.PRNGKey(200 + seed)
    for method, sub in (('subsample', True), ('min', False)):
        new = select_best_of_n(q, method, rng)
        old = _pre_change_inline(q, sub, rng, num_qs)
        assert new.shape == (B,)
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
    np.testing.assert_array_equal(np.asarray(select_best_of_n(q, 'mean')), np.asarray(q).mean(0).argmax(1))


def test_select_ties_first_index():
    q = jnp.zeros((4, 3, 5))
    for method in ('min', 'subsample', 'mean'):
        np.testing.assert_array_equal(np.asarray(select_best_of_n(q, method, jax.random.PRNGKey(0))), [0, 0, 0])
    # quantised q-values (many exact ties): still equal to the pre-change rule
    qq = jnp.round(jax.random.normal(jax.random.PRNGKey(4), (10, 16, 32)))
    rng = jax.random.PRNGKey(9)
    np.testing.assert_array_equal(np.asarray(select_best_of_n(qq, 'subsample', rng)),
                                  np.asarray(_pre_change_inline(qq, True, rng, 10)))
    np.testing.assert_array_equal(np.asarray(select_best_of_n(qq, 'min')),
                                  np.asarray(_pre_change_inline(qq, False, None, 10)))


@pytest.mark.parametrize("method", ['min', 'subsample', 'mean'])
def test_select_under_jit_matches_eager(method):
    q = jax.random.normal(jax.random.PRNGKey(11), (10, 8, 32))
    rng = jax.random.PRNGKey(12)
    jitted = jax.jit(lambda q, r: select_best_of_n(q, method, r))
    np.testing.assert_array_equal(np.asarray(jitted(q, rng)), np.asarray(select_best_of_n(q, method, rng)))
    np.testing.assert_array_equal(np.asarray(jitted(q, rng)),
                                  np.asarray(jax.jit(lambda q, r: _pre_change_inline(q, method == 'subsample', r, 10))(q, rng))
                                  if method != 'mean' else np.asarray(q).mean(0).argmax(1))


def test_clip_inf_bounds_identity():
    x = jnp.array([-1e30, -3.0, -1.0, -0.0, 0.0, 0.5, 2.0, 1e30, jnp.inf, -jnp.inf], jnp.float32)
    for f in (lambda x: jnp.clip(x, -jnp.inf, jnp.inf), jax.jit(lambda x: jnp.clip(x, -jnp.inf, jnp.inf))):
        y = f(x)
        np.testing.assert_array_equal(np.asarray(y), np.asarray(x))
        assert np.array_equal(np.signbit(np.asarray(y)), np.signbit(np.asarray(x)))  # -0.0 kept


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
