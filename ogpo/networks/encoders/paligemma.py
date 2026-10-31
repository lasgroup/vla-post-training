"""PaliGemma frozen vision encoder for OGPO (pure JAX/Flax).

Loads the SigLIP So400m/14 vision tower from a Pi0/Pi0.5 checkpoint and uses
it as a frozen image encoder. The architecture is imported from the openpi
codebase.

Output: mean-pooled image tokens (B, 2048) concatenated with proprioceptive
state → (B, 2048 + state_dim).

Usage:
    encode_fn = load_paligemma_encoder(
        checkpoint_path="/data/user_data/.../pi05_libero/params",
    )
    encoded = encode_fn(observations, images)  # numpy in, numpy out
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


# ---------------------------------------------------------------------------
# SigLIP architecture (minimal port from openpi/models/siglip.py)
# ---------------------------------------------------------------------------

def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    y, x = jnp.mgrid[:h, :w]
    assert width % 4 == 0
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }
        _, _, d = x.shape
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        return x, out


class SigLIPEncoder(nn.Module):
    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}
        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
        else:
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class SigLIPViT(nn.Module):
    """SigLIP So400m/14 Vision Transformer."""
    num_classes: int | None = None
    patch_size: tuple = (14, 14)
    width: int = 1152
    depth: int = 27
    mlp_dim: int = 4304
    num_heads: int = 16
    posemb: str = "learn"
    pool_type: str = "none"
    scan: bool = True
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction
        x = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Positional embeddings
        x = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        n, _, c = x.shape
        x = x.astype(self.dtype_mm)

        x, _ = SigLIPEncoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            scan=self.scan,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)

        # pool_type="none" → return all patch tokens
        # Apply head projection if num_classes is set (1152 → 2048)
        if self.num_classes:
            x = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head")(x)

        return x  # (B, 256, num_classes) or (B, 256, width)


# ---------------------------------------------------------------------------
# Image preprocessing (matches PaliGemma/SigLIP expectations)
# ---------------------------------------------------------------------------

def preprocess_images(images, target_size=224):
    """Resize and normalize images for SigLIP.

    Supports multi-camera inputs: if C = 3 · num_cams (e.g. transport's
    shouldercamera0+shouldercamera1 concatenated on the channel axis),
    cameras are split and returned as (B · num_cams, target_size, target_size, 3).

    Args:
        images: (B, H, W, C) numpy uint8 or float [0, 255]. C must be a multiple of 3.
        target_size: Target image size (224 for PaliGemma).

    Returns:
        (B · num_cams, target_size, target_size, 3) float32 in [-1, 1], where
        num_cams = C // 3.
    """
    from PIL import Image as PILImage

    images = np.asarray(images)
    B, H, W, C = images.shape
    if C % 3 != 0:
        raise ValueError(f"preprocess_images expects C divisible by 3 (got C={C})")
    num_cams = C // 3

    if num_cams > 1:
        # (B, H, W, num_cams·3) -> (B·num_cams, H, W, 3)
        images = images.reshape(B, H, W, num_cams, 3)
        images = images.transpose(0, 3, 1, 2, 4)       # (B, num_cams, H, W, 3)
        images = images.reshape(B * num_cams, H, W, 3)

    if H != target_size or W != target_size:
        # Convert to uint8 for PIL resize (on CPU, avoids JAX GPU memory)
        if images.dtype != np.uint8:
            imgs_uint8 = np.clip(images, 0, 255).astype(np.uint8)
        else:
            imgs_uint8 = images
        n_imgs = imgs_uint8.shape[0]
        resized = np.empty((n_imgs, target_size, target_size, 3), dtype=np.uint8)
        for i in range(n_imgs):
            resized[i] = np.array(
                PILImage.fromarray(imgs_uint8[i]).resize(
                    (target_size, target_size), PILImage.BILINEAR
                )
            )
        images = resized

    # To float [0, 1] then normalize to [-1, 1] (SigLIP convention)
    images = images.astype(np.float32) / 255.0
    images = (images - 0.5) / 0.5
    return images


# ---------------------------------------------------------------------------
# Checkpoint loading and encode_fn factory
# ---------------------------------------------------------------------------

def _load_siglip_params(checkpoint_path):
    """Load SigLIP vision tower params from a Pi0/Pi0.5 orbax checkpoint.

    Robust to the checkpoint's saved device mesh: builds an abstract target
    pytree that pins every leaf to a single CPU device, then uses orbax's
    construct_restore_args helper to translate that into the per-leaf
    ArrayRestoreArgs the restorer expects. This works for:
      - pi05_base: replicated/single-device → trivially restored
      - pi05_libero: saved with 8-way sharding → re-routed onto host

    Args:
        checkpoint_path: Path to params dir (e.g., ".../pi05_libero/params").

    Returns:
        Dict of params for the SigLIP ViT (Flax format).
    """
    import orbax.checkpoint as ocp
    import jax

    print(f"[PaliGemma] Loading checkpoint: {checkpoint_path}")
    ckptr = ocp.PyTreeCheckpointer()

    # Build an abstract target tree from the on-disk metadata, with every
    # leaf pinned to host (CPU) so we don't need to match the saved mesh.
    metadata = ckptr.metadata(checkpoint_path)
    # Newer orbax versions wrap the actual pytree inside a StepMetadata
    # (or similar) container; unwrap to get the raw tree of leaf metadatas.
    for attr in ('item_metadata', 'metadata', 'tree'):
        inner = getattr(metadata, attr, None)
        if inner is not None:
            metadata = inner
            break

    cpu_device = jax.devices('cpu')[0]
    cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)

    def _is_array_leaf(x):
        return hasattr(x, 'shape') and hasattr(x, 'dtype')

    def _to_abstract(leaf):
        if _is_array_leaf(leaf):
            return jax.ShapeDtypeStruct(
                shape=leaf.shape, dtype=leaf.dtype, sharding=cpu_sharding
            )
        return leaf

    target = jax.tree.map(_to_abstract, metadata, is_leaf=_is_array_leaf)
    restore_args = ocp.checkpoint_utils.construct_restore_args(target)
    result = ckptr.restore(checkpoint_path, restore_args=restore_args)

    # Extract vision tower params, then drop refs to the rest of the
    # checkpoint (Gemma LM, action expert, etc.) so they get GC'd —
    # important for pi05_libero which carries multi-GB of LM weights.
    img_params = result['params']['PaliGemma']['img']
    del result, target, restore_args, metadata
    import gc
    gc.collect()

    # Strip 'value' suffix if present (NNX format)
    from flax import traverse_util
    flat = traverse_util.flatten_dict(img_params)
    if all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
        img_params = traverse_util.unflatten_dict(flat)

    return img_params


def _load_siglip_params_from_npz(npz_path):
    """Load SigLIP params from the official PaliGemma npz checkpoint.

    Args:
        npz_path: Path to pt_224.npz file.

    Returns:
        Dict of params for the SigLIP ViT.
    """
    from flax import traverse_util

    print(f"[PaliGemma] Loading npz: {npz_path}")
    with open(npz_path, 'rb') as f:
        flat_params = dict(np.load(f, allow_pickle=False))

    # The npz has keys like "params/img/..." — extract just the img subtree
    img_flat = {}
    for k, v in flat_params.items():
        parts = k.split("/")
        # Keys are like "params/img/Transformer/..."
        if len(parts) >= 3 and parts[0] == "params" and parts[1] == "img":
            new_key = "/".join(parts[2:])
            img_flat[new_key] = v

    # Unflatten
    img_params = traverse_util.unflatten_dict(
        {tuple(k.split("/")): v for k, v in img_flat.items()}
    )
    return img_params


def load_paligemma_encoder(
    checkpoint_path: str = "/data/user_data/sarveshp/physical_intelligence/pi05_libero/params",
    img_proj_dim: int = 0,
    state_proj_dim: int = 0,
    target_size: int = 224,
    encoder_device: int | None = None,
):
    """Load PaliGemma SigLIP vision tower and return a frozen encode function.

    Args:
        checkpoint_path: Path to Pi0/Pi0.5 orbax checkpoint params dir,
            or path to PaliGemma npz file.
        img_proj_dim: Unused (kept for API compatibility with MinViT path).
        state_proj_dim: Unused (kept for API compatibility).
        target_size: Image resize target (224 for PaliGemma).
        encoder_device: GPU index to run the encoder on. If None, uses default
            device. Set to a different GPU than training to avoid memory
            contention (e.g., encoder_device=1 while training on GPU 0).

    Returns:
        encode_fn(observations, images) -> encoded numpy array.
    """
    # Load params
    if checkpoint_path.endswith('.npz'):
        siglip_params = _load_siglip_params_from_npz(checkpoint_path)
    else:
        siglip_params = _load_siglip_params(checkpoint_path)

    # Determine device for the encoder
    if encoder_device is not None:
        devices = jax.devices("gpu")
        if encoder_device < len(devices):
            enc_device = devices[encoder_device]
            print(f"[PaliGemma] Using GPU {encoder_device} for encoding")
        else:
            enc_device = jax.devices()[0]
            print(f"[PaliGemma] GPU {encoder_device} not available, using default device")
    else:
        enc_device = jax.devices()[0]

    # Place params on the encoder device
    siglip_params = jax.device_put(siglip_params, enc_device)

    # Create the SigLIP model (So400m/14 architecture)
    # num_classes=2048 adds the head projection (1152→2048)
    siglip = SigLIPViT(
        num_classes=2048,
        patch_size=(14, 14),
        width=1152,
        depth=27,
        mlp_dim=4304,
        num_heads=16,
        posemb="learn",
        pool_type="none",
        scan=True,
        dtype_mm="float32",
    )

    frozen_params = {'params': siglip_params}

    # Verify shapes by running a dummy forward pass on the encoder device
    dummy_img = jax.device_put(jnp.zeros((1, target_size, target_size, 3)), enc_device)
    try:
        out = siglip.apply(frozen_params, dummy_img, train=False)
        output_dim = out.shape[-1]
        print(f"[PaliGemma] SigLIP output: {out.shape} → mean pool → (B, {output_dim})")
    except Exception as e:
        print(f"[PaliGemma] Warning: shape verification failed: {e}")
        output_dim = 2048

    print(f"[PaliGemma] Final encoded dim: {output_dim} (img) + state_dim (proprio)")

    # JIT-compile the forward pass, pinned to encoder device.
    # Pass params as argument (not closure) to avoid embedding ~500MB as XLA constants.
    @functools.partial(jax.jit, device=enc_device)
    def _encode_images(params, pixel_values):
        tokens = siglip.apply(params, pixel_values, train=False)
        return tokens.mean(axis=1)  # (B, 2048)

    def encode_fn(observations, images):
        """Encode images + state into a flat feature vector.

        Supports multi-camera inputs: if `images` has C = 3 · num_cams (e.g. 6
        channels for transport's two cameras), SigLIP is applied per-camera and
        the resulting features are concatenated along the feature axis.

        Args:
            observations: numpy array — proprioceptive state. (B, state_dim) or (state_dim,).
            images: numpy array — pixel observations in [0, 255]. (B, H, W, C) or (H, W, C).
                    C must be a multiple of 3.

        Returns:
            numpy float32 array with encoded features of shape
            (B, num_cams · img_feat_dim + state_dim) or squeezed batch dim.
        """
        squeeze = images.ndim == 3
        if squeeze:
            images = images[np.newaxis]
            observations = observations[np.newaxis]
        B = images.shape[0]
        num_cams = images.shape[-1] // 3

        # Preprocess: resize to 224x224, normalize to [-1, 1]. For multi-camera
        # this returns (B·num_cams, 224, 224, 3).
        pixel_values = preprocess_images(images, target_size=target_size)

        # Place input on encoder device, then run SigLIP + mean pool
        pixel_jax = jax.device_put(jnp.array(pixel_values), enc_device)
        img_features = np.asarray(_encode_images(frozen_params, pixel_jax))  # (B·num_cams, D)

        # Concatenate cameras along feature axis
        if num_cams > 1:
            img_features = img_features.reshape(B, num_cams * img_features.shape[-1])

        # Concatenate with proprioceptive state
        encoded = np.concatenate([img_features, observations], axis=-1).astype(np.float32)

        if squeeze:
            encoded = encoded[0]
        return encoded

    return encode_fn
