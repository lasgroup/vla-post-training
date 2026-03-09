import numpy as np
import pytest

from src.envs.dsrl_vector_env import DSRLVectorEnv


def _env_stub(action_dim: int = 7, action_horizon: int = 10) -> DSRLVectorEnv:
    env = DSRLVectorEnv.__new__(DSRLVectorEnv)
    env._policy_action_dim = action_dim
    env._policy_action_horizon = action_horizon
    return env


def test_expand_noise_from_compact_batch_shape():
    env = _env_stub(action_dim=4, action_horizon=6)
    noise = np.arange(12, dtype=np.float32).reshape(3, 4)

    expanded = env._expand_noise_to_horizon(noise, batch_size=3)

    assert expanded.shape == (3, 6, 4)
    np.testing.assert_allclose(expanded[:, 0, :], noise)
    np.testing.assert_allclose(expanded[:, -1, :], noise)


def test_expand_noise_from_single_horizon_repeats_to_policy_horizon():
    env = _env_stub(action_dim=3, action_horizon=5)
    noise = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)

    expanded = env._expand_noise_to_horizon(noise, batch_size=1)

    assert expanded.shape == (1, 5, 3)
    np.testing.assert_allclose(expanded[0, :, :], np.array([[1.0, 2.0, 3.0]] * 5))


def test_expand_noise_from_flattened_batch_shape():
    env = _env_stub(action_dim=2, action_horizon=4)
    # Flattened horizon=3 compact latent representation.
    noise = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)

    expanded = env._expand_noise_to_horizon(noise, batch_size=1)

    assert expanded.shape == (1, 4, 2)
    np.testing.assert_allclose(expanded[0, 0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(expanded[0, 1], np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(expanded[0, 2], np.array([5.0, 6.0], dtype=np.float32))
    # Last step should pad with the final compact latent vector.
    np.testing.assert_allclose(expanded[0, 3], np.array([5.0, 6.0], dtype=np.float32))


def test_expand_noise_raises_on_action_dim_mismatch():
    env = _env_stub(action_dim=4, action_horizon=5)
    bad_noise = np.ones((2, 1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Noise action dim"):
        env._expand_noise_to_horizon(bad_noise, batch_size=2)
