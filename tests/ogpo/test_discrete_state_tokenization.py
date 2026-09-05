"""TokenizePrompt must pair each row with its OWN state.

``discrete_state_input=True`` (pi0.5's native proprio channel) writes the state into
the prompt as digitized text. openpi's transform was written for the offline
per-sample dataloader; the online stack calls it batched in two places:

  collection    prompt = [E env strings],  state = (E, 8)
  buffer insert prompt = "<task>",          state = (T+1, 8)

Before the fix both silently rendered EVERY row's state into EVERY prompt as a
bracketed numpy array ("State: [140 140 ...] [12 12 ...]"), and the buffer variant
also blew past max_token_len and truncated. No exception either way.

The last test is the load-bearing one: with discrete_state_input=False the patch must
be a bit-exact no-op, so the existing state-blind arms stay valid comparators.
"""

import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


MAX_LEN = 200
PROMPTS = ["pick up the black bowl", "open the top drawer"]
STATES = np.array([[0.1] * 8, [-0.9] * 8], dtype=np.float32)


@pytest.fixture(scope="module")
def tok():
    return _tokenizer.PaligemmaTokenizer(MAX_LEN)


@pytest.fixture(scope="module")
def tp_on(tok):
    return _transforms.TokenizePrompt(tok, discrete_state_input=True)


@pytest.fixture(scope="module")
def tp_off(tok):
    return _transforms.TokenizePrompt(tok, discrete_state_input=False)


def test_batched_prompts_pair_with_own_state(tp_on, tok):
    """Collection path: E envs, E states."""
    out = tp_on({"prompt": list(PROMPTS), "state": STATES})
    tokens = out["tokenized_prompt"]
    assert tokens.shape == (len(PROMPTS), MAX_LEN)
    for i, (p, s) in enumerate(zip(PROMPTS, STATES, strict=True)):
        np.testing.assert_array_equal(tokens[i], tok.tokenize(p, s)[0])
    assert not np.array_equal(tokens[0], tokens[1])


def test_scalar_prompt_with_per_timestep_state(tp_on, tok):
    """Buffer-insert path: one task string, T+1 observations."""
    states = np.stack([np.full(8, v, dtype=np.float32) for v in (0.1, 0.2, 0.3)])
    out = tp_on({"prompt": PROMPTS[0], "state": states})
    tokens, mask = out["tokenized_prompt"], out["tokenized_prompt_mask"]
    assert tokens.shape == (len(states), MAX_LEN)
    for t, s in enumerate(states):
        np.testing.assert_array_equal(tokens[t], tok.tokenize(PROMPTS[0], s)[0])
    assert len({row.tobytes() for row in tokens}) == len(states), "rows must differ"
    assert mask.sum(axis=1).max() < MAX_LEN, "prompt must not fill the token budget"


def test_unbatched_stays_rank_one(tp_on):
    """Offline dataloader path: must keep the (max_len,) shape."""
    out = tp_on({"prompt": PROMPTS[0], "state": STATES[0]})
    assert out["tokenized_prompt"].shape == (MAX_LEN,)


def test_batch_mismatch_raises(tp_on):
    with pytest.raises(ValueError, match="batch mismatch"):
        tp_on({"prompt": list(PROMPTS), "state": STATES[:1]})


@pytest.mark.parametrize(
    "prompt,state",
    [
        (list(PROMPTS), STATES),          # collection
        (PROMPTS[0], STATES),             # buffer insert
        (PROMPTS[0], STATES[0]),          # offline dataloader
    ],
)
def test_state_blind_path_is_unchanged(tp_off, tok, prompt, state):
    """discrete_state_input=False must reproduce pre-patch behavior exactly."""
    out = tp_off({"prompt": prompt, "state": state})
    if isinstance(prompt, list):
        expected = np.stack([tok.tokenize(p, None)[0] for p in prompt])
    else:
        expected = tok.tokenize(prompt, None)[0]
    np.testing.assert_array_equal(out["tokenized_prompt"], expected)


@pytest.mark.parametrize("per_row", [False, True])
def test_buffer_accepts_both_prompt_ranks(per_row):
    """The observation ring must take a (T+1, max_len) prompt column as well as the
    (max_len,) one it broadcasts today -- that rank change is the only downstream
    effect of the fix (replay_buffer.py:113)."""
    from src.rl.replay_buffer import ShardedReplayBuffer

    n_obs, n_txn = 6, 5
    dummy = {
        "observations": {
            "image": {"base_0_rgb": np.zeros((1, 4, 4, 3), np.uint8)},
            "image_mask": {"base_0_rgb": np.zeros((1,), bool)},
            "state": np.zeros((1, 32), np.float32),
            "tokenized_prompt": np.zeros((1, MAX_LEN), np.int32),
            "tokenized_prompt_mask": np.zeros((1, MAX_LEN), bool),
        },
        "actions": np.zeros((1, 10, 32), np.float32),
        "reward": np.zeros((1,), np.float32),
    }
    prompt = (
        np.stack([np.full(MAX_LEN, t, np.int32) for t in range(n_obs)])
        if per_row
        else np.zeros(MAX_LEN, np.int32)
    )
    buf = ShardedReplayBuffer(dummy_data=dummy, max_capacity=1000, freeze_dict=False)
    buf.insert(
        {
            "observations": {
                "image": {"base_0_rgb": np.zeros((n_obs, 4, 4, 3), np.uint8)},
                "image_mask": {"base_0_rgb": np.ones((n_obs,), bool)},
                "state": np.zeros((n_obs, 32), np.float32),
                "tokenized_prompt": prompt,
                "tokenized_prompt_mask": np.ones((n_obs, MAX_LEN), bool),
            },
            "obs_index": np.arange(n_txn),
            "next_obs_index": np.arange(n_txn) + 1,
            "actions": np.zeros((n_txn, 10, 32), np.float32),
            "reward": np.zeros((n_txn,), np.float32),
        }
    )
    # num_obs must come from a real observation leaf, not from the max_len axis.
    assert buf.obs_total == n_obs and buf.size == n_txn
    stored = buf.obs_storage["tokenized_prompt"][:n_obs]
    assert len({row.tobytes() for row in stored}) == (n_obs if per_row else 1)
