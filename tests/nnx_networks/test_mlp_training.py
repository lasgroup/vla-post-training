
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from src.rl.networks.mlp import MLP
import numpy as np

def train_mlp():
    print("Initializing MLP Training Test...")
    
    # 1. Data Generation (Non-Linear Regression: y = sin(X @ W) + 0.5 * cos(X @ W))
    rng = jax.random.PRNGKey(0)
    key_x, key_noise = jax.random.split(rng)
    X = jax.random.normal(key_x, (200, 10))  # 200 samples, 10 features
    W_true = jax.random.normal(key_noise, (10, 1))
    
    # Non-linear target
    h = X @ W_true
    Y = jnp.sin(h) + 0.5 * jnp.cos(h) + 0.05 * jax.random.normal(key_noise, (200, 1))
    
    # Check data shape
    print(f"Data shapes - X: {X.shape}, Y: {Y.shape}")

    # 2. Model Initialization
    rngs = nnx.Rngs(0)
    # Lazy init: input dim determined at first call
    # Increased capacity for non-linear task
    model = MLP(hidden_dims=(64, 64, 1), activations=nnx.relu, rngs=rngs)
    
    # 3. Optimizer
    # We need to initialize the model parameters first by running a dummy input
    # before constructing the optimizer, because the optimizer needs the parameters.
    dummy_input = jnp.ones((1, 10))
    _ = model(dummy_input) # Trigger lazy init
    
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.005))

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
    for step in range(500):
        loss = train_step(model, optimizer, X, Y)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    print(f"Final Loss: {loss:.4f}")
    print("Training test passed!")

if __name__ == "__main__":
    train_mlp()
