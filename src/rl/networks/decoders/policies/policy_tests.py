import jax
import jax.numpy as jnp
import flax.nnx as nn
from normal_policy import NormalPolicyDecoder
from learned_std_normal_policy import LearnedStdTanhNormalPolicyDecoder, LearnedStdNormalPolicyDecoder


def test_policies():
    print("=== Starting Policy Tests ===")

    # 1. Setup Data
    batch_size = 8
    obs_dim = 16
    action_dim = 4
    hidden_dims = (32, 32)

    # Fake observations
    key = jax.random.key(0)
    obs = jax.random.normal(key, (batch_size, obs_dim))
    dummy_obs = jnp.zeros(obs_dim)
    dummy_action = jnp.zeros(action_dim)

    # RNGs for NNX
    rngs = nn.Rngs(0)

    # ---------------------------------------------------------
    # Test 1: NormalPolicy (Fixed Std)
    # ---------------------------------------------------------
    print("\n[1/3] Testing NormalPolicy...")
    policy_1 = NormalPolicyDecoder(
        observation=dummy_obs,
        action=dummy_action,
        hidden_dims=hidden_dims,
        std=0.5,
        rngs=rngs
    )

    # Forward pass
    dist_1 = policy_1(obs)

    # Check 1: Output type
    assert hasattr(dist_1, 'sample'), "Output must be a distribution with .sample()"

    # Check 2: Sample shape
    sample_key = jax.random.key(1)
    samples_1 = dist_1.sample(seed=sample_key)
    assert samples_1.shape == (batch_size, action_dim), \
        f"Expected shape {(batch_size, action_dim)}, got {samples_1.shape}"

    # Check 3: Log Prob shape
    log_probs_1 = dist_1.log_prob(samples_1)
    assert log_probs_1.shape == (batch_size,), \
        f"Expected log_prob shape {(batch_size,)}, got {log_probs_1.shape}"

    print("      -> Shapes correct.")
    print("      -> Log prob computable.")

    # ---------------------------------------------------------
    # Test 2: LearnedStdNormalPolicy
    # ---------------------------------------------------------
    print("\n[2/3] Testing LearnedStdNormalPolicy...")
    policy_2 = LearnedStdNormalPolicyDecoder(
        observation=dummy_obs,
        action=dummy_action,
        hidden_dims=hidden_dims,
        rngs=rngs
    )

    dist_2 = policy_2(obs)
    samples_2 = dist_2.sample(seed=sample_key)
    log_probs_2 = dist_2.log_prob(samples_2)

    assert samples_2.shape == (batch_size, action_dim)
    assert log_probs_2.shape == (batch_size,)

    # Check clipping logic (inspecting internal scale isn't easy on compiled dists,
    # but we verify it runs without NaN)
    assert not jnp.any(jnp.isnan(samples_2)), "Samples contain NaNs!"

    print("      -> Forward pass successful.")
    print("      -> Shapes correct.")
    print("      -> Log prob computable.")

    # ---------------------------------------------------------
    # Test 3: LearnedStdTanhNormalPolicy (The complex one)
    # ---------------------------------------------------------
    print("\n[3/3] Testing LearnedStdTanhNormalPolicy (Bounded)...")

    low_limit = -2.0
    high_limit = 2.0

    policy_3 = LearnedStdTanhNormalPolicyDecoder(
        observation=dummy_obs,
        action=dummy_action,
        hidden_dims=hidden_dims,
        low=low_limit,
        high=high_limit,
        rngs=rngs
    )

    dist_3 = policy_3(obs)

    # A. Check Mode
    mode = dist_3.mode()
    assert mode.shape == (batch_size, action_dim)
    print("      -> Mode computation successful.")

    # B. Check Bounds strictly
    # We draw many samples to ensure none escape the bounds
    n_samples = 1000
    many_samples = dist_3.sample(seed=sample_key, sample_shape=(n_samples,))
    # Shape: (n_samples, batch_size, action_dim)

    min_val = jnp.min(many_samples)
    max_val = jnp.max(many_samples)

    print(f"      -> Sample Range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"      -> Target Range: [{low_limit}, {high_limit}]")

    # Allow a tiny float tolerance for the check
    tol = 1e-5
    assert min_val >= low_limit - tol, f"Sample {min_val} violated lower bound {low_limit}"
    assert max_val <= high_limit + tol, f"Sample {max_val} violated upper bound {high_limit}"

    # C. Check Bijector Math (Manual Verification)
    # y = shift + scale * tanh(x)
    # If tanh returns 1.0 -> y should be high
    # If tanh returns -1.0 -> y should be low

    # Access the bijector chain
    bijector = dist_3.bijector

    # Test extreme values input to the bijector (simulate the output of the base Normal)
    inf_input = jnp.array([100.0])  # effectively +infinity for tanh
    neg_inf_input = jnp.array([-100.0])  # effectively -infinity for tanh

    out_high = bijector.forward(inf_input)
    out_low = bijector.forward(neg_inf_input)

    assert jnp.allclose(out_high, high_limit, atol=1e-4), f"Bijector high failed: got {out_high}, expected {high_limit}"
    assert jnp.allclose(out_low, low_limit, atol=1e-4), f"Bijector low failed: got {out_low}, expected {low_limit}"

    print("      -> Bijector logic verified.")

    # D. Check Log Prob on boundaries (Stability check)
    # Tanh distributions can be unstable near boundaries. TFP handles this usually.
    # We check if log_prob is finite for a value inside the domain.
    valid_sample = jnp.ones((batch_size, action_dim)) * ((high_limit + low_limit) / 2)
    lp = dist_3.log_prob(valid_sample)
    assert jnp.all(jnp.isfinite(lp)), "Log probs should be finite for valid samples"

    print("      -> Log probability numerical stability verified.")

    print("\n=== All Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_policies()