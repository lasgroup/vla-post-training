import jax
import jax.numpy as jnp
import flax.nnx as nn
from src.rl.networks.mlp import MLP
from src.rl.networks.normal_tanh_policy import NormalTanhPolicy
from src.rl.networks.values.state_action_value import StateActionValue
from src.rl.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from src.rl.networks.encoders.resnet_encoderv1 import ResNetEncoder
import numpy as np

def test_mlp():
    print("Testing MLP...")
    rngs = nn.Rngs(0)
    mlp = MLP(hidden_dims=(64, 64), activations=nn.relu, rngs=rngs)
    x = jnp.ones((1, 32))
    y = mlp(x)
    print(f"MLP output shape: {y.shape}")
    assert y.shape == (1, 64)

def test_policy():
    print("Testing NormalTanhPolicy...")
    rngs = nn.Rngs(0)
    policy = NormalTanhPolicy(hidden_dims=(64, 64), action_dim=5, rngs=rngs)
    x = jnp.ones((1, 32))
    dist = policy(x)
    action = dist.mode()
    print(f"Policy action shape: {action.shape}")
    assert action.shape == (1, 5)

def test_value():
    print("Testing StateActionValue...")
    rngs = nn.Rngs(0)
    # StateActionValue(hidden_dims, activations, final_fc_dim, rngs)
    value_net = StateActionValue(hidden_dims=(64, 64), rngs=rngs) 
    obs = jnp.ones((1, 32))
    action = jnp.ones((1, 5))
    v = value_net(obs, action)
    print(f"Value output shape: {v.shape}")
    assert v.shape == (1,)

def test_impala():
    print("Testing ImpalaEncoder...")
    rngs = nn.Rngs(0)
    # ImpalaEncoder(nn_scale, rngs)
    encoder = ImpalaEncoder(nn_scale=1, rngs=rngs)
    # Input image (Batch, H, W, C)
    img = jnp.zeros((1, 64, 64, 3), dtype=jnp.uint8) # uint8 0-255
    emb = encoder(img)
    print(f"Impala embedding shape: {emb.shape}")
    # Output shape depends on architecture
    

def test_smaller_impala():
    print("Testing SmallerImpalaEncoder...")
    rngs = nn.Rngs(0)
    encoder = SmallerImpalaEncoder(nn_scale=1, rngs=rngs)
    img = jnp.zeros((1, 64, 64, 3), dtype=jnp.uint8)
    emb = encoder(img)
    print(f"SmallerImpala embedding shape: {emb.shape}")
    
def test_resnet():
    print("Testing ResNetEncoder...")
    rngs = nn.Rngs(0)
    # ResNetEncoder(stage_sizes, block_cls, ...)
    # Need block_cls. Import from resnet_encoderv1
    from src.rl.networks.encoders.resnet_encoderv1 import ResNetBlock
    encoder = ResNetEncoder(stage_sizes=(2, 2, 2), block_cls=ResNetBlock, rngs=rngs)
    img = jnp.zeros((1, 64, 64, 5, 3), dtype=jnp.uint8)
    emb = encoder(img)
    print(f"ResNet embedding shape (RGB): {emb.shape}")
    
    # Test with 1-channel input (grayscale)
    img_gray = jnp.zeros((1, 64, 64, 5, 1), dtype=jnp.uint8)
    # We need a fresh encoder for this test because the previous one is initialized for 3 channels
    encoder_gray = ResNetEncoder(stage_sizes=(2, 2, 2), block_cls=ResNetBlock, rngs=nn.Rngs(1))
    emb_gray = encoder_gray(img_gray)
    print(f"ResNet embedding shape (Gray): {emb_gray.shape}")

if __name__ == "__main__":
    test_mlp()
    test_policy()
    test_value()
    try:
        test_smaller_impala()
    except Exception as e:
        print(f"SmallerImpala failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_impala()
    except Exception as e:
        print(f"Impala failed: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        test_resnet()
    except Exception as e:
        print(f"ResNet failed: {e}")
        import traceback
        traceback.print_exc()

    print("Verification complete!")
