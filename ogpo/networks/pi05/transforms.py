"""pi05 input/output transforms, matching openpi's policy pipeline:

  LiberoInputs -> Normalize(quantile) -> ResizeImages(224) -> TokenizePrompt -> PadStatesAndActions(32)
  ... model ...
  Unnormalize(quantile) -> LiberoOutputs (first 7 dims)

`ROBOTS` holds the per-checkpoint differences (pi05_libero vs pi05_droid_jointpos). For droid, openpi's
DroidInputs/DroidOutputs + DeltaActions/AbsoluteActions(make_bool_mask(7, -1)) make the model space a joint delta
relative to the state at the start of the chunk (dims 0-6) plus an absolute gripper; the agent applies the delta
(`OGPOPi05Agent._to_env_actions` / `_normalize_env_actions`). Both keep openpi's normalise-then-zero-pad order.

Everything is numpy on host; the model sees only fixed-shape arrays.
"""

import json
import pathlib

import numpy as np
from PIL import Image

from ogpo.networks.pi05.pi05 import IMAGE_KEYS

IMAGE_RESOLUTION = 224
MODEL_ACTION_DIM = 32
MAX_TOKEN_LEN = 200  # pi05 default

# Per-robot action contract (agent.pi05_robot).
#   env_action_dim: real action dims (the rest of the 32 are zero padding)
#   asset_id: norm-stats dir under <checkpoint>/assets
#   delta_dims: leading dims predicted relative to the window-start state (droid: 7 joints; gripper absolute)
#   clip_env_actions: eval clips env actions to [-1, 1]; LIBERO's are normalised EEF deltas, droid's are joint radians
ROBOTS = {
    'libero': dict(env_action_dim=7, asset_id='physical-intelligence/libero', delta_dims=0, clip_env_actions=True),
    'droid': dict(env_action_dim=8, asset_id='droid', delta_dims=7, clip_env_actions=False),
}
ENV_ACTION_DIM = ROBOTS['libero']['env_action_dim']  # LIBERO value; the agent uses its own profile


def load_norm_stats(assets_dir, asset_id="physical-intelligence/libero"):
    """Reads `<assets_dir>/<asset_id>/norm_stats.json` -> {'state': {...}, 'actions': {...}} of np arrays."""
    path = pathlib.Path(assets_dir) / asset_id / "norm_stats.json"
    stats = json.loads(path.read_text())["norm_stats"]
    return {k: {s: np.asarray(v, dtype=np.float32) for s, v in d.items() if v is not None} for k, d in stats.items()}


def normalize_quantile(x, stats):
    q01, q99 = stats["q01"][..., : x.shape[-1]], stats["q99"][..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x, stats):
    q01, q99 = stats["q01"], stats["q99"]
    if (dim := q01.shape[-1]) < x.shape[-1]:
        return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def pad_to_dim(x, dim):
    pad = dim - x.shape[-1]
    if pad <= 0:
        return x
    return np.concatenate([x, np.zeros((*x.shape[:-1], pad), dtype=x.dtype)], axis=-1)


def _resize_with_pad_pil(image, height, width):
    """openpi_client.image_tools.resize_with_pad for one uint8 (H, W, 3) image."""
    image = Image.fromarray(np.asarray(image))
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.asarray(image)
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    canvas.paste(resized, (max(0, int((width - resized_width) / 2)), max(0, int((height - resized_height) / 2))))
    return np.asarray(canvas)


def resize_images(images):
    """(..., H, W, 3) uint8 -> (..., 224, 224, 3) uint8."""
    if images.shape[-3:-1] == (IMAGE_RESOLUTION, IMAGE_RESOLUTION):
        return images
    flat = images.reshape(-1, *images.shape[-3:])
    out = np.stack([_resize_with_pad_pil(im, IMAGE_RESOLUTION, IMAGE_RESOLUTION) for im in flat])
    return out.reshape(*images.shape[:-3], *out.shape[-3:])


def to_model_images(base_uint8, wrist_uint8):
    """uint8 224x224 base/wrist -> (images dict in [-1, 1], image_masks dict). Right wrist is a masked zero image."""
    base = base_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0
    wrist = wrist_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0
    # LiberoInputs pads with a uint8 zero image, i.e. -1 after conversion.
    right = np.full_like(base, -1.0)
    batch = base.shape[:-3]
    images = dict(zip(IMAGE_KEYS, (base, wrist, right)))
    image_masks = {
        "base_0_rgb": np.ones(batch, dtype=bool),
        "left_wrist_0_rgb": np.ones(batch, dtype=bool),
        "right_wrist_0_rgb": np.zeros(batch, dtype=bool),
    }
    return images, image_masks


def state_to_model(state, norm_stats):
    """Raw 8-D LIBERO state -> quantile-normalised, zero-padded to 32 (critic input only)."""
    return pad_to_dim(normalize_quantile(state.astype(np.float32), norm_stats["state"]), MODEL_ACTION_DIM)


def actions_to_env(model_actions, norm_stats):
    """(..., H, 32) normalised model actions -> (..., H, 7) env actions."""
    return unnormalize_quantile(np.asarray(model_actions, dtype=np.float32), norm_stats["actions"])[..., :ENV_ACTION_DIM]


def actions_to_model(env_actions, norm_stats):
    """(..., H, 7) env actions -> (..., H, 32) normalised, zero-padded."""
    return pad_to_dim(normalize_quantile(env_actions.astype(np.float32), norm_stats["actions"]), MODEL_ACTION_DIM)


class PromptTokenizer:
    """openpi PaligemmaTokenizer.tokenize with state=None (pi05_libero uses discrete_state_input=False)."""

    def __init__(self, tokenizer_path, max_len=MAX_TOKEN_LEN):
        import sentencepiece

        self._max_len = max_len
        with open(tokenizer_path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt):
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        if len(tokens) > self._max_len:
            raise ValueError(f"Prompt has {len(tokens)} tokens > max_len {self._max_len}")
        mask = [True] * len(tokens) + [False] * (self._max_len - len(tokens))
        tokens = tokens + [0] * (self._max_len - len(tokens))
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)
