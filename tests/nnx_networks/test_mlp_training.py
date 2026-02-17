
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from src.rl.networks.mlp import MLP
import numpy as np

def train_mlp():
    print("Initializing MLP Training Test...")
    
    # 1. Data Generation (Linear Regression: y = 2x + 1)
    rng = jax.random.PRNGKey(0)
    key_x, key_noise = jax.random.split(rng)
    X = jax.random.normal(key_x, (100, 10))  # 100 samples, 10 features
    W_true = jax.random.normal(key_noise, (10, 1))
    Y = X @ W_true + 0.1 * jax.random.normal(key_noise, (100, 1))
    
    # Check data shape
    print(f"Data shapes - X: {X.shape}, Y: {Y.shape}")

    # 2. Model Initialization
    rngs = nnx.Rngs(0)
    # Lazy init: input dim determined at first call
    model = MLP(hidden_dims=(32, 1), activations=nnx.relu, rngs=rngs)
    
    # 3. Optimizer
    # We need to initialize the model parameters first by running a dummy input
    # before constructing the optimizer, because the optimizer needs the parameters.
    dummy_input = jnp.ones((1, 10))
    _ = model(dummy_input) # Trigger lazy init
    
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=0.01))

    # 4. Training Loop
    @nnx.jit
    def train_step(model, optimizer, x_batch, y_batch):
        def loss_fn(model):
            pred = model(x_batch)
            loss = jnp.mean((pred - y_batch) ** 2)
            return loss
        
        grad = nnx.grad(loss_fn)(model)
        optimizer.update(grad)
        return loss_fn(model)

    print("Starting training loop...")
    for step in range(100):
        loss = train_step(model, optimizer, X, Y)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    print(f"Final Loss: {loss:.4f}")
    print("Training test passed!")

if __name__ == "__main__":
    train_mlp()
