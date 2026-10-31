"""
Test: Does distrax reparameterization affect log_prob gradients in the SDE chain?

Claim: When you sample x = dist.sample() and compute dist.log_prob(x), distrax's
reparameterization trick makes gradients of log_prob flow through x back to the
distribution parameters, zeroing out the score function gradient.

Result: The claim is mathematically TRUE — the gradient IS zeroed out. But this
does NOT affect OGPO because:
  1. old_lp from sample_flow_actions_sde uses TARGET params (not grad_params)
  2. new_lp is computed via compute_flow_log_prob on DETACHED chains
  3. Only new_lp is differentiated, and it has correct gradients
"""

import jax
import jax.numpy as jnp
import distrax

# ──────────────────────────────────────────────────────────────────────
# Test 1: Single-step — reparam zeroes out d(log_prob)/d(mu)
# ──────────────────────────────────────────────────────────────────────
def test_single_step_reparam_zeros_gradient():
    """
    sample x ~ N(mu, sigma), compute log p(x; mu, sigma).
    With reparam: d(log_prob)/d(mu) = 0  (x depends on mu, cancels the score)
    With stop_grad: d(log_prob)/d(mu) = (x - mu)/sigma^2  (the score function)
    """
    mu = jnp.array([1.0, 2.0, 3.0])
    sigma = jnp.array([0.5, 0.5, 0.5])
    key = jax.random.PRNGKey(42)

    # WITH reparameterization (current distrax behavior)
    def lp_with_reparam(mu):
        dist = distrax.Normal(mu, sigma)
        x = dist.sample(seed=key)  # x = mu + sigma * eps (reparam)
        return dist.log_prob(x).sum()

    grad_reparam = jax.grad(lp_with_reparam)(mu)

    # WITHOUT reparameterization (stop_gradient on sample)
    def lp_without_reparam(mu):
        dist = distrax.Normal(mu, sigma)
        x = jax.lax.stop_gradient(dist.sample(seed=key))
        return dist.log_prob(x).sum()

    grad_no_reparam = jax.grad(lp_without_reparam)(mu)

    # The actual sample for reference
    dist = distrax.Normal(mu, sigma)
    x = dist.sample(seed=key)
    expected_score = (x - mu) / sigma**2

    print("=== Test 1: Single-step reparam effect ===")
    print(f"  mu = {mu}")
    print(f"  x  = {x}")
    print(f"  grad WITH reparam:    {grad_reparam}  (should be ~0)")
    print(f"  grad WITHOUT reparam: {grad_no_reparam}")
    print(f"  expected score:       {expected_score}")
    print(f"  reparam grad is zero: {jnp.allclose(grad_reparam, 0.0, atol=1e-6)}")
    print(f"  no-reparam matches score: {jnp.allclose(grad_no_reparam, expected_score, atol=1e-6)}")

    assert jnp.allclose(grad_reparam, 0.0, atol=1e-6), \
        f"Reparam gradient should be zero, got {grad_reparam}"
    assert jnp.allclose(grad_no_reparam, expected_score, atol=1e-6), \
        f"Non-reparam gradient should be score function"
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────
# Test 2: Multi-step SDE chain — reparam zeroes out gradient through
#          the ENTIRE chain, not just one step
# ──────────────────────────────────────────────────────────────────────
def test_multi_step_chain_reparam_zeros_gradient():
    """
    Simulate a mini SDE chain:  x_{i+1} = x_i + W @ x_i * dt + sigma * eps_i
    Accumulate log_prob at each step.
    Show gradient of total log_prob w.r.t. W is zero with reparam.
    """
    act_dim = 4
    flow_steps = 5
    dt = 1.0 / flow_steps
    sigma = 0.1
    key = jax.random.PRNGKey(0)

    W = jax.random.normal(jax.random.PRNGKey(1), (act_dim, act_dim)) * 0.1

    def chain_logprob_with_reparam(W):
        rng = key
        rng, init_key = jax.random.split(rng)
        x = jax.random.normal(init_key, (act_dim,))
        total_lp = jnp.float32(0.0)

        for i in range(flow_steps):
            rng, step_key = jax.random.split(rng)
            vel = W @ x
            mean_next = x + vel * dt
            dist = distrax.Normal(mean_next, sigma)
            x = dist.sample(seed=step_key)  # reparam: x = mean_next + sigma * eps
            total_lp = total_lp + dist.log_prob(x).sum()

        return total_lp

    def chain_logprob_without_reparam(W):
        rng = key
        rng, init_key = jax.random.split(rng)
        x = jax.random.normal(init_key, (act_dim,))
        total_lp = jnp.float32(0.0)

        for i in range(flow_steps):
            rng, step_key = jax.random.split(rng)
            vel = W @ x
            mean_next = x + vel * dt
            dist = distrax.Normal(mean_next, sigma)
            x = jax.lax.stop_gradient(dist.sample(seed=step_key))
            total_lp = total_lp + dist.log_prob(x).sum()

        return total_lp

    grad_reparam = jax.grad(chain_logprob_with_reparam)(W)
    grad_no_reparam = jax.grad(chain_logprob_without_reparam)(W)

    print("=== Test 2: Multi-step chain reparam effect ===")
    print(f"  grad WITH reparam norm:    {jnp.linalg.norm(grad_reparam):.8f}  (should be ~0)")
    print(f"  grad WITHOUT reparam norm: {jnp.linalg.norm(grad_no_reparam):.8f}  (should be non-zero)")
    print(f"  reparam grad is zero: {jnp.allclose(grad_reparam, 0.0, atol=1e-5)}")

    assert jnp.allclose(grad_reparam, 0.0, atol=1e-5), \
        f"Multi-step reparam gradient should be zero, norm = {jnp.linalg.norm(grad_reparam)}"
    assert jnp.linalg.norm(grad_no_reparam) > 0.01, \
        f"Non-reparam gradient should be non-zero"
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────
# Test 3: Forward VALUES are identical — reparam only affects gradients
# ──────────────────────────────────────────────────────────────────────
def test_forward_values_identical():
    """
    The log_prob VALUE from sample_flow_actions_sde is numerically identical
    regardless of reparameterization. Only the gradient computation differs.
    """
    mu = jnp.array([1.0, -0.5, 2.0])
    sigma = jnp.array([0.3, 0.3, 0.3])
    key = jax.random.PRNGKey(99)

    dist = distrax.Normal(mu, sigma)
    x = dist.sample(seed=key)
    x_sg = jax.lax.stop_gradient(x)

    lp_reparam = dist.log_prob(x).sum()
    lp_no_reparam = dist.log_prob(x_sg).sum()

    print("=== Test 3: Forward values are identical ===")
    print(f"  log_prob with reparam:    {lp_reparam:.8f}")
    print(f"  log_prob without reparam: {lp_no_reparam:.8f}")
    print(f"  identical: {jnp.allclose(lp_reparam, lp_no_reparam, atol=1e-7)}")

    assert jnp.allclose(lp_reparam, lp_no_reparam, atol=1e-7)
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────
# Test 4: OGPO workflow — sample with target params, recompute log_prob
#          with current params. The gradient that matters is through
#          compute_flow_log_prob, which operates on DETACHED chains.
# ──────────────────────────────────────────────────────────────────────
def test_ogpo_workflow_gradient_is_correct():
    """
    Simulates the OGPO gradient flow:
      1. Sample chain from target policy (SDE) → chains, old_lp
      2. Recompute log_prob of same chain under current policy → new_lp
      3. Loss = f(new_lp, old_lp)  — gradient only through new_lp

    Shows that the gradient through new_lp (via compute_flow_log_prob pattern)
    is the correct score function, and that old_lp's gradient doesn't matter
    because it comes from a different parameter set.
    """
    act_dim = 4
    flow_steps = 5
    dt = 1.0 / flow_steps
    sigma = 0.1
    key = jax.random.PRNGKey(7)

    W_target = jax.random.normal(jax.random.PRNGKey(1), (act_dim, act_dim)) * 0.1
    W_current = W_target + 0.01 * jax.random.normal(jax.random.PRNGKey(2), (act_dim, act_dim))

    # Step 1: Sample chain from target policy (like _sample_actions with use_target=True)
    def sample_chain(W, rng):
        rng, init_key = jax.random.split(rng)
        x = jax.random.normal(init_key, (act_dim,))
        chain = [x]
        total_lp = jnp.float32(0.0)

        for i in range(flow_steps):
            rng, step_key = jax.random.split(rng)
            vel = W @ x
            mean_next = x + vel * dt
            dist = distrax.Normal(mean_next, sigma)
            x = dist.sample(seed=step_key)  # reparam (doesn't matter, no grad taken)
            total_lp = total_lp + dist.log_prob(x).sum()
            chain.append(x)

        return jnp.stack(chain), total_lp

    chain, old_lp = sample_chain(W_target, key)

    # Step 2: Recompute log_prob under current policy (like compute_flow_log_prob)
    # This takes the DETACHED chain and evaluates it under W_current
    def recompute_logprob(W_current, chain):
        total_lp = jnp.float32(0.0)
        for i in range(flow_steps):
            x = chain[i]  # detached — no gradient through chain
            x_next = chain[i + 1]  # detached
            vel = W_current @ x
            mean_next = x + vel * dt
            dist = distrax.Normal(mean_next, sigma)
            total_lp = total_lp + dist.log_prob(x_next).sum()
        return total_lp

    new_lp = recompute_logprob(W_current, chain)

    # Step 3: Gradient of new_lp w.r.t. W_current
    grad_new_lp = jax.grad(recompute_logprob)(W_current, chain)

    print("=== Test 4: OGPO workflow gradient ===")
    print(f"  old_lp (from target, no grad needed): {old_lp:.4f}")
    print(f"  new_lp (from current, grad needed):   {new_lp:.4f}")
    print(f"  grad(new_lp) norm: {jnp.linalg.norm(grad_new_lp):.6f}  (should be non-zero)")

    assert jnp.linalg.norm(grad_new_lp) > 0.01, \
        "Gradient of recomputed log_prob should be non-zero (it's the score function)"
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────
# Test 5: Full SDE chain with neural-network-like velocity — confirm
#          reparam zeroing even with nonlinear dynamics
# ──────────────────────────────────────────────────────────────────────
def test_nonlinear_velocity_reparam():
    """
    Use a 2-layer MLP as velocity field. Reparam still zeroes the gradient
    of chain log_prob w.r.t. MLP params.
    """
    act_dim = 8
    hidden_dim = 16
    flow_steps = 10
    dt = 1.0 / flow_steps
    sigma = 0.05
    key = jax.random.PRNGKey(42)

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    params = {
        'W1': jax.random.normal(k1, (act_dim, hidden_dim)) * 0.1,
        'b1': jnp.zeros(hidden_dim),
        'W2': jax.random.normal(k2, (hidden_dim, act_dim)) * 0.1,
        'b2': jnp.zeros(act_dim),
    }

    def velocity(x, params):
        h = jax.nn.relu(x @ params['W1'] + params['b1'])
        return h @ params['W2'] + params['b2']

    def chain_logprob_reparam(params):
        rng = key
        rng, init_key = jax.random.split(rng)
        x = jax.random.normal(init_key, (act_dim,))
        total_lp = jnp.float32(0.0)

        for i in range(flow_steps):
            rng, step_key = jax.random.split(rng)
            vel = velocity(x, params)
            mean_next = x + vel * dt
            dist = distrax.Normal(mean_next, sigma)
            x = dist.sample(seed=step_key)
            total_lp = total_lp + dist.log_prob(x).sum()

        return total_lp

    def chain_logprob_stopgrad(params):
        rng = key
        rng, init_key = jax.random.split(rng)
        x = jax.random.normal(init_key, (act_dim,))
        total_lp = jnp.float32(0.0)

        for i in range(flow_steps):
            rng, step_key = jax.random.split(rng)
            vel = velocity(x, params)
            mean_next = x + vel * dt
            dist = distrax.Normal(mean_next, sigma)
            x = jax.lax.stop_gradient(dist.sample(seed=step_key))
            total_lp = total_lp + dist.log_prob(x).sum()

        return total_lp

    grad_reparam = jax.grad(chain_logprob_reparam)(params)
    grad_stopgrad = jax.grad(chain_logprob_stopgrad)(params)

    reparam_norms = {k: jnp.linalg.norm(v) for k, v in grad_reparam.items()}
    stopgrad_norms = {k: jnp.linalg.norm(v) for k, v in grad_stopgrad.items()}

    print("=== Test 5: Nonlinear (MLP) velocity field ===")
    for k in params:
        print(f"  {k}: reparam norm = {reparam_norms[k]:.8f}, stopgrad norm = {stopgrad_norms[k]:.6f}")

    all_zero = all(jnp.allclose(v, 0.0, atol=1e-4) for v in grad_reparam.values())
    any_nonzero = any(stopgrad_norms[k] > 0.01 for k in params)

    assert all_zero, f"Reparam gradients should all be ~0: {reparam_norms}"
    assert any_nonzero, f"Stop-grad gradients should be non-zero: {stopgrad_norms}"
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────
# Test 6: Using actual sample_flow_actions_sde and compute_flow_log_prob
#          from the codebase — verify values match and OGPO gradient works
# ──────────────────────────────────────────────────────────────────────
def test_actual_sde_functions():
    """
    Use the real pg_helper functions to confirm:
    1. log_prob from sample_flow_actions_sde matches compute_flow_log_prob
       (when randn_clip is disabled to avoid the known numerical mismatch)
    2. Gradient of compute_flow_log_prob w.r.t. velocity params is non-zero
       (this is the gradient OGPO actually uses)
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from ogpo.agents.modules.pg_helper import sample_flow_actions_sde, compute_flow_log_prob

    batch_size = 4
    act_dim = 6
    obs_dim = 8
    flow_steps = 5
    sigma = 0.1

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    vel_params = {
        'W': jax.random.normal(k1, (obs_dim + act_dim + 1, act_dim)) * 0.01,
        'b': jnp.zeros(act_dim),
    }

    obs = jax.random.normal(jax.random.PRNGKey(10), (batch_size, obs_dim))

    def actor_fn(obs, actions, t, params=None, is_encoded=False):
        p = params if params is not None else vel_params
        inp = jnp.concatenate([obs, actions, t], axis=-1)
        return inp @ p['W'] + p['b']

    def noise_fn(obs, t, params=None):
        return sigma * jnp.ones((obs.shape[0], act_dim))

    # Use large randn_clip to effectively disable it → values should match exactly
    rng = jax.random.PRNGKey(42)
    actions, chains, lp_sde, sigmas = sample_flow_actions_sde(
        actor_fn=actor_fn, noise_fn=noise_fn, observations=obs, rng=rng,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=sigma,
        randn_clip_value=1e6,
        act_min=-1e6, act_max=1e6,
        clip_intermediate=False,
    )

    lp_recomputed, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn, noise_fn=noise_fn, observations=obs, chains=chains,
        flow_steps=flow_steps, use_constant_noise=True, constant_noise_std=sigma,
        clip_intermediate=False,
    )

    print("=== Test 6: Actual SDE functions ===")
    print(f"  lp from SDE sampling:     {lp_sde}")
    print(f"  lp from recompute:        {lp_recomputed}")
    print(f"  match: {jnp.allclose(lp_sde, lp_recomputed, atol=1e-3)}")

    # Gradient of recomputed log_prob (the OGPO path)
    def loss_fn(vel_params):
        lp, _, _ = compute_flow_log_prob(
            actor_fn=lambda obs, actions, t, params=None, is_encoded=False:
                jnp.concatenate([obs, actions, t], axis=-1) @ vel_params['W'] + vel_params['b'],
            noise_fn=noise_fn, observations=obs, chains=chains,
            flow_steps=flow_steps, use_constant_noise=True, constant_noise_std=sigma,
            clip_intermediate=False,
        )
        return lp.mean()

    grad_recompute = jax.grad(loss_fn)(vel_params)
    grad_norms = {k: jnp.linalg.norm(v) for k, v in grad_recompute.items()}
    print(f"  grad(recomputed_lp) norms: {grad_norms}  (should be non-zero)")

    assert jnp.allclose(lp_sde, lp_recomputed, atol=1e-3), \
        f"Log-prob values should match: sde={lp_sde}, recomputed={lp_recomputed}"
    assert any(v > 0.001 for v in grad_norms.values()), \
        f"Recomputed log_prob gradient should be non-zero: {grad_norms}"
    print("  PASSED\n")


if __name__ == '__main__':
    print("=" * 70)
    print("Reparameterization trick & log_prob gradient analysis")
    print("=" * 70)
    print()

    test_single_step_reparam_zeros_gradient()
    test_multi_step_chain_reparam_zeros_gradient()
    test_forward_values_identical()
    test_ogpo_workflow_gradient_is_correct()
    test_nonlinear_velocity_reparam()
    test_actual_sde_functions()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print()
    print("CONCLUSION:")
    print("  The friend is mathematically correct: distrax reparameterization")
    print("  causes d(log_prob(sample()))/d(params) = 0, which differs from")
    print("  the score function you'd get with stop_gradient on the sample.")
    print()
    print("  However, this does NOT affect OGPO because:")
    print("  1. old_lp comes from sample_flow_actions_sde with TARGET params")
    print("     (not grad_params), so its gradient w.r.t. grad_params is 0 anyway")
    print("  2. new_lp comes from compute_flow_log_prob on DETACHED chains")
    print("     (stored in buffer), giving correct score function gradients")
    print("  3. The PPO loss = f(new_lp - old_lp) only differentiates new_lp")
    print("=" * 70)
