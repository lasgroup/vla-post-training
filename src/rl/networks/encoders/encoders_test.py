import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from functools import partial

# Import your encoders
# Adjust imports if your folder structure is slightly different (e.g. strict src.rl...)
from cnn_encoder import CNNEncoder
from impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from resnet_encoderv1 import ResNetSmall
from resnet_encoderv2 import ResNetv2_Small
from spatial_softmax import SpatialSoftmax
from encoders import ImageEncoder, MLPEncoder, BaseEncoder
from src.rl.networks.mlp import MLP

rngs = nnx.Rngs(42)

# =========================================
# HELPERS
# =========================================

def create_dummy_obs(batch_size=4, height=64, width=64, channels=3):
    """Creates a dummy observation dictionary."""
    return {
        'pixels_left': jnp.zeros((batch_size, height, width, channels), dtype=jnp.uint8),
        'pixels_right': jnp.zeros((batch_size, height, width, channels), dtype=jnp.uint8),
        'state': jnp.zeros((batch_size, 10)),
        'velocity': jnp.zeros((batch_size, 10))
    }


rngs = nnx.Rngs(42)


# =========================================
# 1. SPATIAL SOFTMAX TESTS
# =========================================

def test_spatial_softmax_shape():
    print("\n[Test] Spatial Softmax Shape")
    batch_size = 3
    h, w, c = 10, 10, 8

    # Create coordinate grids manually to mock what the layer does internally
    # (Just to initialize the layer, actual values don't matter for shape test)
    pos_x, pos_y = jnp.meshgrid(
        jnp.linspace(-1., 1., h),
        jnp.linspace(-1., 1., w)
    )
    pos_x = pos_x.reshape(h * w)
    pos_y = pos_y.reshape(h * w)

    layer = SpatialSoftmax(height=h, width=w, channel=c,
                           pos_x=pos_x, pos_y=pos_y,
                           temperature=1.0, rngs=rngs)

    # Input feature map: [Batch, Height, Width, Channels]
    feature_map = jnp.zeros((batch_size, h, w, c))

    out = layer(feature_map)

    # Expected output: [Batch, Channels * 2] (x and y for each channel)
    assert out.shape == (batch_size, c * 2)
    print("  -> Output shape is correct:", out.shape)


def test_spatial_softmax_logic():
    """Test if spatial softmax correctly identifies a 'hot' pixel."""
    print("\n[Test] Spatial Softmax Logic")
    h, w, c = 5, 5, 1

    pos_x, pos_y = jnp.meshgrid(
        jnp.linspace(-1., 1., h),
        jnp.linspace(-1., 1., w)
    )
    pos_x = pos_x.reshape(h * w)
    pos_y = pos_y.reshape(h * w)

    layer = SpatialSoftmax(height=h, width=w, channel=c,
                           pos_x=pos_x, pos_y=pos_y,
                           temperature=0.1,  # Low temp for argmax-like behavior
                           rngs=rngs)

    # Create a feature map with a clear maximum at (0, 0) (top-left)
    # Note: In standard image coords, top-left is usually (-1, -1) in this normalization
    feature_map = jnp.zeros((1, h, w, c))
    feature_map = feature_map.at[0, 0, 0, 0].set(100.0)  # Spike at top-left

    out = layer(feature_map)

    # Check if output coords match the top-left coordinate defined in pos_x/pos_y
    # The grid was flattened, index 0 corresponds to (0,0) in grid
    expected_x = pos_x[0]
    expected_y = pos_y[0]

    # Output is [Batch, C*2] -> [1, 2] -> [x, y]
    assert jnp.allclose(out[0, 0], expected_x, atol=1e-2)
    assert jnp.allclose(out[0, 1], expected_y, atol=1e-2)
    print("  -> Correctly localized feature peak.")


# =========================================
# 2. CNN ENCODER TESTS
# =========================================

def test_cnn_encoder_shape():
    print("\n[Test] CNNEncoder Shape")
    obs = create_dummy_obs(batch_size=2, height=64, width=64, channels=3)

    # Default: 4 layers with strides (2, 1, 1, 1), features (32, 32, 32, 32)
    # Input: 64x64
    # L1 (stride 2): 32x32
    # L2 (stride 1): 30x30 (valid padding)
    # L3 (stride 1): 28x28
    # L4 (stride 1): 26x26
    # Final Channel: 32

    model = CNNEncoder(input_example=obs,
                       features=(32, 32, 32, 32),
                       strides=(2, 1, 1, 1),
                       padding='VALID',
                       image_keys=['pixels_left', 'pixels_right'],
                       rngs=rngs)

    out = model(obs)

    # CNNEncoder implementation loop ends.
    # It seems to return the raw convolution map unless it has a flatten/dense at the end.
    # Based on your file, it returns: `x.reshape((*x.shape[:-3], -1))` ?
    # Let's check the behavior dynamically.

    print("  -> Output shape:", out.shape)

    # Check batch dimension
    assert out.shape[0] == 2

    # Check if it is flattened (2D) or spatial (4D)
    # Your Impala encoder flattens, let's see if CNNEncoder flattens.
    # If the code snippet `x = jnp.reshape(x, (*x.shape[:-2], -1))` is in __call__,
    # and `CNNEncoder` loops over layers...
    # Wait, the `CNNEncoder` snippet provided had `_prepare_input` doing reshape?
    # No, `_prepare_input` flattens frame stack.

    # Assuming standard CNNEncoder just returns the feature map or a flattened vector.
    # If it returns a feature map (B, H, W, C), rank is 4.
    if out.ndim == 4:
        print("  -> Returned Spatial Map")
    elif out.ndim == 2:
        print("  -> Returned Flattened Embedding")
        assert out.shape[1] > 0


# =========================================
# 3. IMPALA ENCODER TESTS
# =========================================

def test_impala_encoder_shape():
    print("\n[Test] ImpalaEncoder Shape")
    obs = create_dummy_obs(batch_size=2, height=64, width=64)

    # Standard Impala
    model = ImpalaEncoder(
        input_example=obs, rngs=rngs, image_keys=['pixels_left', 'pixels_right'], )

    out = model(obs)

    print("  -> Output shape:", out.shape)
    assert out.shape[0] == 2
    assert out.ndim == 2  # Impala usually flattens at the end

    # Check Large vs Small
    model_small = SmallerImpalaEncoder(input_example=obs, rngs=rngs, image_keys=['pixels_left', 'pixels_right'])
    out_small = model_small(obs)
    print("  -> Small Impala Output shape:", out_small.shape)
    assert out_small.shape[0] == 2


# =========================================
# 4. RESNET ENCODER V1 TESTS
# =========================================

def test_resnet_encoderv1_shape():
    print("\n[Test] ResNetEncoder (V1/Small) Shape")
    obs = create_dummy_obs(batch_size=2, height=64, width=64)

    # ResNetSmall is a partial of ResNetEncoder
    # It likely uses GroupNorm or BatchNorm.
    # NNX handles state (batch stats) automatically in the graph if we use stateful run,
    # or inside the module if we use standard __call__.

    # Note: Your ResNet implementation expects `stage_sizes`.
    # ResNetSmall usually sets this via partial.

    model = ResNetSmall(input_example=obs, rngs=rngs, image_keys=['pixels_left', 'pixels_right'])

    # Run in training mode (updates stats if BN, though RL often uses GroupNorm/LayerNorm)
    out = model(obs, train=True)

    print("  -> Output shape:", out.shape)
    assert out.shape[0] == 2

    # ResNet usually performs Global Average Pooling at the end
    # so output should be (Batch, Features)
    assert out.ndim == 2


# =========================================
# 4. RESNET ENCODER V2 TESTS
# =========================================

def test_resnet_encoderv2_shape():
    print("\n[Test] ResNetEncoder (V2/Small) Shape")
    obs = create_dummy_obs(batch_size=2, height=64, width=64, channels=3)

    # ResNetSmall is a partial of ResNetEncoder
    # It likely uses GroupNorm or BatchNorm.
    # NNX handles state (batch stats) automatically in the graph if we use stateful run,
    # or inside the module if we use standard __call__.

    # Note: Your ResNet implementation expects `stage_sizes`.
    # ResNetSmall usually sets this via partial.

    model = ResNetv2_Small(input_example=obs, rngs=rngs, image_keys=['pixels_left', 'pixels_right'])

    # Run in training mode (updates stats if BN, though RL often uses GroupNorm/LayerNorm)
    out = model(obs, train=True)

    print("  -> Output shape:", out.shape)
    assert out.shape[0] == 2

    # ResNet usually performs Global Average Pooling at the end
    # so output should be (Batch, Features)
    assert out.ndim == 2


# =========================================
# TEST: BASE ENCODER (MERGING)
# =========================================

def test_base_encoder_merge():
    print("\n[Test] BaseEncoder Merge (Image + State)")

    batch_size = 2
    obs = create_dummy_obs(batch_size=batch_size)

    # 1. Define Image Encoder Factory
    #    We use a simple CNN wrapped in ImageEncoder with a bottleneck
    image_latent_dim = 50

    # Inner CNN factory
    def cnn_factory(obs_dict, rng):
        return CNNEncoder(input_example=obs_dict,
                          features=(16, 32),
                          strides=(2, 2),
                          image_keys=['pixels_left', 'pixels_right'],
                          rngs=rng)

    # Outer ImageEncoder factory (as expected by BaseEncoder)
    def image_encoder_def(obs_dict, rng):
        return ImageEncoder(dummy_obs=obs_dict,
                            encoder_def=cnn_factory,
                            latent_dim=image_latent_dim,
                            use_bottleneck=True,
                            rngs=rng)

    # 2. Define MLP Encoder Factory
    #    We want to encode 'state' and 'velocity'
    state_keys = ['state', 'velocity']  # 10 + 3 = 13 dim
    mlp_latent_dim = 20

    # Inner MLP factory (takes flat vector)
    def mlp_factory(flat_input, rng):
        return MLP(input=flat_input,
                   hidden_dims=[32, mlp_latent_dim],  # Output 20
                   activations=nnx.relu,
                   rngs=rng)

    # Outer MLPEncoder factory
    def mlp_encoder_def(obs_dict, rng):
        return MLPEncoder(dummy_obs=obs_dict,
                          encoder_def=mlp_factory,
                          state_vector_keys=state_keys,
                          rngs=rng)

    # 3. Instantiate BaseEncoder
    #    This is the component we are testing
    model = BaseEncoder(dummy_obs=obs,
                        mlp_encoder_def=mlp_encoder_def,
                        image_encoder_def=image_encoder_def,
                        rngs=rngs)

    # 4. Run Forward Pass
    out = model(obs, training=True)

    print("  -> Output shape:", out.shape)

    # 5. Verify Shapes
    #    Expected: [Batch, Image_Latent + MLP_Latent]
    #    Expected: [2, 50 + 20] = [2, 70]
    expected_dim = image_latent_dim + mlp_latent_dim
    assert out.shape == (batch_size, expected_dim)
    print(
        f"  -> Successfully merged embeddings: {image_latent_dim} (Image) + {mlp_latent_dim} (State) = {out.shape[1]}")


def test_base_encoder_defaults():
    print("\n[Test] BaseEncoder Defaults (No Encoders)")
    # If no encoders are provided, it should flatten and concat everything
    batch_size = 2
    obs = create_dummy_obs(batch_size=batch_size)

    model = BaseEncoder(dummy_obs=obs,
                        mlp_encoder_def=None,
                        image_encoder_def=None,
                        rngs=rngs)

    out = model(obs)

    # Total dim:
    # pixels: 64*64*3 = 12288
    # state: 10
    # velocity: 3
    # Total: 12301

    print("  -> Output shape:", out.shape)

    # Note: Depending on dictionary ordering (if not FrozenDict/Ordered), concatenation order might vary,
    # but the total size should be deterministic.
    expected_dim = (64 * 64 * 6) + 10 + 10
    assert out.shape == (batch_size, expected_dim)
    print("  -> Successfully flattened raw observations.")

# =========================================
# MAIN
# =========================================


if __name__ == "__main__":
    # Manually running tests
    try:
        test_spatial_softmax_shape()
        test_spatial_softmax_logic()
        test_cnn_encoder_shape()
        test_impala_encoder_shape()
        test_resnet_encoderv1_shape()
        test_base_encoder_merge()
        test_base_encoder_defaults()
        print("\nAll Encoder Tests Passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback

        traceback.print_exc()
