"""Grouping semantics for the group-relative AWR actor step.

The one thing in update_actor_group.py that is silently wrong if it is wrong:
`jnp.repeat(x, G, axis=0)` must lay each state's G copies out contiguously so
that a later `q.reshape(-1, G)` recovers the groups. If those two disagree the
advantage is computed across states instead of within one, which still trains
and still logs plausible numbers.

Run: .venv/bin/python tests/awr/test_group_advantage.py
"""

import jax.numpy as jnp
import numpy as np


G = 8
B = 5


def test_repeat_reshape_roundtrip():
    """repeat(axis=0) then reshape(-1, G) must recover the originating state."""
    state_id = jnp.arange(B)
    repeated = jnp.repeat(state_id, repeats=G, axis=0)
    grouped = repeated.reshape(-1, G)
    # Every row is one state repeated G times.
    assert grouped.shape == (B, G)
    for b in range(B):
        assert np.all(np.asarray(grouped[b]) == b), f"row {b} mixes states: {grouped[b]}"


def test_group_advantage_is_zero_mean_within_state():
    """A_i = Q_i - mean_j Q_j sums to zero inside each group, not across the batch."""
    rng = np.random.default_rng(0)
    # Per-state offsets far larger than the within-state spread: this is the
    # critic level error the group baseline is supposed to remove.
    offsets = rng.normal(0, 50, size=(B, 1))
    spread = rng.normal(0, 1.0, size=(B, G))
    q_value = jnp.asarray((offsets + spread).reshape(-1))

    q_grouped = q_value.reshape(-1, G)
    group_mean = jnp.mean(q_grouped, axis=-1, keepdims=True)
    advantage = (q_grouped - group_mean).reshape(-1)

    per_group_sum = np.asarray(advantage.reshape(-1, G).sum(axis=-1))
    assert np.allclose(per_group_sum, 0.0, atol=1e-4), per_group_sum
    # The 50-unit per-state offsets are gone; only the unit-scale spread is left.
    assert np.abs(advantage).max() < 10.0, np.abs(advantage).max()


def test_weight_share_is_a_distribution():
    """group_weight_share must sum to 1 across task groups."""
    num_groups = 4
    rng = np.random.default_rng(1)
    task_id = jnp.asarray(rng.integers(0, num_groups, size=B))
    group_task_id = jnp.repeat(task_id, repeats=G, axis=0)
    score = jnp.asarray(rng.uniform(0.1, 5.0, size=B * G))

    onehot = jnp.zeros((B * G, num_groups)).at[jnp.arange(B * G), group_task_id].set(1.0)
    share = jnp.sum(onehot * score[:, None], axis=0) / jnp.maximum(jnp.sum(score), 1e-12)

    assert np.isclose(float(jnp.sum(share)), 1.0, atol=1e-5), share


def test_subsample_keeps_backward_size():
    """batch_size/G states expanded G ways is back to batch_size chunks."""
    batch_size = 256
    reduced = batch_size // G
    assert reduced * G == batch_size, (reduced, G, batch_size)
    assert reduced == 32


def test_awr_default_stays_on_buffer_path():
    """awr.sh must not change: group_advantage defaults off."""
    from src.training.config import AdvantageWeightedSFTLearnerConfig

    cfg = AdvantageWeightedSFTLearnerConfig()
    assert cfg.group_advantage is False
    assert cfg.group_size == 8
    assert cfg.group_noise_level == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
