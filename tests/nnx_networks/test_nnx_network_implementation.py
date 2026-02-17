
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from src.rl.networks.mlp import MLP
from src.rl.networks.normal_tanh_policy import NormalTanhPolicy
from src.rl.networks.values.state_action_value import StateActionValue
from src.rl.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from src.rl.networks.encoders.resnet_encoderv1 import ResNetEncoder, ResNetBlock, BottleneckResNetBlock
from src.rl.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
import numpy as np

def test_mlp():
    print("Testing MLP...")
    rngs = nnx.Rngs(0)
    mlp = MLP(hidden_dims=(64, 64), activations=nnx.relu, rngs=rngs)
    x = jnp.ones((1, 32))
    print(f"MLP Input shape: {x.shape}")
    y = mlp(x)
    print(f"MLP output shape: {y.shape}")
    assert y.shape == (1, 64)

def test_policy():
    print("Testing NormalTanhPolicy...")
    rngs = nnx.Rngs(0)
    policy = NormalTanhPolicy(hidden_dims=(64, 64), action_dim=5, rngs=rngs)
    x = jnp.ones((1, 32))
    print(f"Policy Input shape: {x.shape}")
    dist = policy(x)
    action = dist.mode()
    print(f"Policy action shape: {action.shape}")
    assert action.shape == (1, 5)

def test_value():
    print("Testing StateActionValue...")
    rngs = nnx.Rngs(0)
    # StateActionValue(hidden_dims, activations, final_fc_dim, rngs)
    value_net = StateActionValue(hidden_dims=(64, 64), rngs=rngs) 
    obs = jnp.ones((1, 32))
    action = jnp.ones((1, 5))
    print(f"Value Input obs shape: {obs.shape}, action shape: {action.shape}")
    v = value_net(obs, action)
    print(f"Value output shape: {v.shape}")
    assert v.shape == (1,)

def test_impala():
    print("Testing ImpalaEncoder...")
    rngs = nnx.Rngs(0)
    # ImpalaEncoder(nn_scale, rngs)
    encoder = ImpalaEncoder(nn_scale=1, rngs=rngs)
    # Input image (Batch, H, W, Stack, C) -> (1, 64, 64, 5, 3) to mimic 4D content but in 5D shape
    img = jnp.zeros((1, 64, 64, 5, 3), dtype=jnp.uint8) 
    print(f"Impala Input shape: {img.shape}")
    emb = encoder(img)
    print(f"Impala embedding shape: {emb.shape}")
    # Calculation: 
    # 64x64 -> Stack1 -> 32x32 -> Stack2 -> 16x16 -> Stack3 -> 8x8 
    # Output is 8*8*32 = 2048.
    assert emb.shape[-1] == 2048

def test_smaller_impala():
    print("Testing SmallerImpalaEncoder...")
    rngs = nnx.Rngs(0)
    encoder = SmallerImpalaEncoder(nn_scale=1, rngs=rngs)
    # Input image (Batch, H, W, Stack, C)
    img = jnp.zeros((1, 64, 64, 5, 3), dtype=jnp.uint8)
    print(f"SmallerImpala Input shape: {img.shape}")
    emb = encoder(img)
    print(f"SmallerImpala embedding shape: {emb.shape}")
    # Same spatial reduction (3 stacks with max pool) -> 8x8
    # Last stack channel = 32.
    assert emb.shape[-1] == 2048
    
def test_resnet():
    print("Testing ResNetEncoder...")
    rngs = nnx.Rngs(0)
    # ResNetEncoder(stage_sizes, block_cls, ...)
    encoder = ResNetEncoder(stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock, rngs=rngs)
    # (Batch, H, W, Stack, C)
    img = jnp.zeros((1, 224, 224, 5, 3), dtype=jnp.uint8)
    print(f"ResNet Input shape: {img.shape}")
    emb = encoder(img)
    print(f"ResNet embedding shape: {emb.shape}")

def test_resnet_bottleneck():
    print("Testing ResNetEncoder with Bottleneck...")
    rngs = nnx.Rngs(0)
    encoder = ResNetEncoder(stage_sizes=(3, 4, 6, 3), block_cls=BottleneckResNetBlock, rngs=rngs)
    img = jnp.zeros((1, 224, 224, 5, 3), dtype=jnp.uint8)
    print(f"ResNet Bottleneck Input shape: {img.shape}")
    emb = encoder(img)
    print(f"ResNet Bottleneck embedding shape: {emb.shape}")

def test_resnet_v2():
    print("Testing ResNetV2Encoder...")
    rngs = nnx.Rngs(0)
    # ResNetV2Encoder(stage_sizes, ...)
    encoder = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), rngs=rngs)
    img = jnp.zeros((1, 224, 224, 5, 3), dtype=jnp.uint8)
    print(f"ResNetV2 Input shape: {img.shape}")
    emb = encoder(img)
    print(f"ResNetV2 embedding shape: {emb.shape}")
    

if __name__ == "__main__":
    test_mlp()
    test_policy()
    test_value()
    try:
        test_impala()
    except Exception as e:
        print(f"Impala failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_smaller_impala()
    except Exception as e:
        print(f"SmallerImpala failed: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        test_resnet()
    except Exception as e:
        print(f"ResNet failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_resnet_bottleneck()
    except Exception as e:
        print(f"ResNet Bottleneck failed: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        test_resnet_v2()
    except Exception as e:
        print(f"ResNetV2 failed: {e}")
        import traceback
        traceback.print_exc()

    print("Verification complete!")
