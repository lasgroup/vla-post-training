"""Minimal SAC on dm_control to validate the critic + actor update logic.

Usage:
    python test_sac_dmc.py                          # pendulum swingup (default)
    python test_sac_dmc.py --env cartpole-swingup
    python test_sac_dmc.py --steps 100000

Requires: dm_control, jax, flax, optax
    pip install dm_control
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import time
from typing import Any, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


# =========================================================================
# 1. DMC wrapper
# =========================================================================

class DMCEnv:
    """Thin wrapper: flat numpy obs, actions rescaled from [-1,1]."""

    def __init__(self, domain: str, task: str, seed: int = 0):
        from dm_control import suite
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        obs_spec = self._env.observation_spec()
        self.obs_dim = sum(int(np.prod(v.shape)) for v in obs_spec.values())
        act_spec = self._env.action_spec()
        self.act_dim = int(np.prod(act_spec.shape))
        self._act_min = act_spec.minimum.astype(np.float32)
        self._act_max = act_spec.maximum.astype(np.float32)

    def _flatten_obs(self, timestep) -> np.ndarray:
        return np.concatenate(
            [np.asarray(v, dtype=np.float32).ravel() for v in timestep.observation.values()]
        )

    def reset(self) -> np.ndarray:
        return self._flatten_obs(self._env.reset())

    def step(self, action: np.ndarray):
        lo, hi = self._act_min, self._act_max
        scaled = np.clip(lo + (action + 1.0) * 0.5 * (hi - lo), lo, hi)
        ts = self._env.step(scaled)
        return self._flatten_obs(ts), float(ts.reward or 0.0), ts.last()


# =========================================================================
# 2. Replay buffer
# =========================================================================

@dataclasses.dataclass
class ReplayBuffer:
    capacity: int
    obs_dim: int
    act_dim: int

    def __post_init__(self):
        c = self.capacity
        self.obs = np.zeros((c, self.obs_dim), np.float32)
        self.act = np.zeros((c, self.act_dim), np.float32)
        self.rew = np.zeros(c, np.float32)
        self.next_obs = np.zeros((c, self.obs_dim), np.float32)
        self.done = np.zeros(c, np.float32)
        self._ptr = self._size = 0

    @property
    def size(self):
        return self._size

    def add(self, obs, act, rew, next_obs, done):
        i = self._ptr % self.capacity
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i] = next_obs, float(done)
        self._ptr += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, rng: np.random.Generator):
        idx = rng.integers(0, self._size, size=batch_size)
        return (
            jnp.asarray(self.obs[idx]),
            jnp.asarray(self.act[idx]),
            jnp.asarray(self.rew[idx]),
            jnp.asarray(self.next_obs[idx]),
            jnp.asarray(self.done[idx]),
        )


# =========================================================================
# 3. Networks
# =========================================================================

class Critic(nnx.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(obs_dim + act_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, obs, act):
        x = nnx.relu(self.fc1(jnp.concatenate([obs, act], -1)))
        return self.out(nnx.relu(self.fc2(x))).squeeze(-1)


class Actor(nnx.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(obs_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.mean_head = nnx.Linear(hidden, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden, act_dim, rngs=rngs)

    def __call__(self, obs, rng):
        x = nnx.relu(self.fc2(nnx.relu(self.fc1(obs))))
        mean = self.mean_head(x)
        log_std = jnp.clip(self.log_std_head(x), -5.0, 2.0)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, mean.shape)
        raw = mean + std * noise
        action = jnp.tanh(raw)
        log_prob = jnp.sum(
            -0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std
            - jnp.log(1.0 - action ** 2 + 1e-6),
            axis=-1,
        )
        return action, log_prob


# =========================================================================
# 4. State: split static (not traced) from dynamic (all arrays)
# =========================================================================

class SACStatic(NamedTuple):
    """Non-array objects that stay outside JIT."""
    actor_def: Any  # nnx.GraphDef
    q1_def: Any
    q2_def: Any
    actor_tx: Any   # optax.GradientTransformation
    q1_tx: Any
    q2_tx: Any
    alpha_tx: Any
    target_entropy: float
    tau: float


@dataclasses.dataclass
class SACDynamic:
    """All array-like state — safe to pass through JIT."""
    actor_params: nnx.State
    actor_opt: optax.OptState
    q1_params: nnx.State
    q1_opt: optax.OptState
    q2_params: nnx.State
    q2_opt: optax.OptState
    q1_target: nnx.State
    q2_target: nnx.State
    log_alpha: jax.Array
    alpha_opt: optax.OptState


def init_sac(
    obs_dim: int,
    act_dim: int,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_alpha: float = 3e-4,
    tau: float = 0.005,
    seed: int = 0,
) -> tuple[SACStatic, SACDynamic]:
    actor = Actor(obs_dim, act_dim, rngs=nnx.Rngs(seed))
    actor_def, actor_params = nnx.split(actor)
    actor_tx = optax.adam(lr_actor)

    q1 = Critic(obs_dim, act_dim, rngs=nnx.Rngs(seed + 1))
    q1_def, q1_params = nnx.split(q1)
    q1_tx = optax.adam(lr_critic)

    q2 = Critic(obs_dim, act_dim, rngs=nnx.Rngs(seed + 2))
    q2_def, q2_params = nnx.split(q2)
    q2_tx = optax.adam(lr_critic)

    log_alpha = jnp.array(0.0, dtype=jnp.float32)
    alpha_tx = optax.adam(lr_alpha)

    static = SACStatic(
        actor_def=actor_def, q1_def=q1_def, q2_def=q2_def,
        actor_tx=actor_tx, q1_tx=q1_tx, q2_tx=q2_tx, alpha_tx=alpha_tx,
        target_entropy=-float(act_dim), tau=tau,
    )
    dynamic = SACDynamic(
        actor_params=actor_params, actor_opt=actor_tx.init(actor_params),
        q1_params=q1_params, q1_opt=q1_tx.init(q1_params),
        q2_params=q2_params, q2_opt=q2_tx.init(q2_params),
        q1_target=jax.tree.map(jnp.copy, q1_params),
        q2_target=jax.tree.map(jnp.copy, q2_params),
        log_alpha=log_alpha, alpha_opt=alpha_tx.init(log_alpha),
    )
    return static, dynamic


# =========================================================================
# 5. Update functions  (static captured via functools.partial)
# =========================================================================

def _update_critic_impl(
    static: SACStatic,
    dyn: SACDynamic,
    rng: jax.Array,
    obs: jax.Array,
    act: jax.Array,
    rew: jax.Array,
    next_obs: jax.Array,
    done: jax.Array,
    gamma: float = 0.99,
) -> tuple[SACDynamic, dict]:
    alpha = jnp.exp(dyn.log_alpha)

    # next actions from current actor
    actor = nnx.merge(static.actor_def, dyn.actor_params)
    next_act, next_log_prob = actor(next_obs, rng)

    # target Q
    q1_t = nnx.merge(static.q1_def, dyn.q1_target)
    q2_t = nnx.merge(static.q2_def, dyn.q2_target)
    q_next = jnp.minimum(q1_t(next_obs, next_act), q2_t(next_obs, next_act))
    td_target = jax.lax.stop_gradient(
        rew + gamma * (1.0 - done) * (q_next - alpha * next_log_prob)
    )

    # Q1
    def q1_loss(p):
        return jnp.mean(jnp.square(nnx.merge(static.q1_def, p)(obs, act) - td_target))

    q1_l, q1_g = jax.value_and_grad(q1_loss)(dyn.q1_params)
    q1_u, q1_o = static.q1_tx.update(q1_g, dyn.q1_opt, dyn.q1_params)
    q1_p = optax.apply_updates(dyn.q1_params, q1_u)

    # Q2
    def q2_loss(p):
        return jnp.mean(jnp.square(nnx.merge(static.q2_def, p)(obs, act) - td_target))

    q2_l, q2_g = jax.value_and_grad(q2_loss)(dyn.q2_params)
    q2_u, q2_o = static.q2_tx.update(q2_g, dyn.q2_opt, dyn.q2_params)
    q2_p = optax.apply_updates(dyn.q2_params, q2_u)

    # Polyak
    tau = static.tau
    q1_targ = jax.tree.map(lambda t, o: tau * o + (1 - tau) * t, dyn.q1_target, q1_p)
    q2_targ = jax.tree.map(lambda t, o: tau * o + (1 - tau) * t, dyn.q2_target, q2_p)

    new_dyn = dataclasses.replace(
        dyn,
        q1_params=q1_p, q1_opt=q1_o,
        q2_params=q2_p, q2_opt=q2_o,
        q1_target=q1_targ, q2_target=q2_targ,
    )
    return new_dyn, {"q1_loss": q1_l, "q2_loss": q2_l, "td_target": jnp.mean(td_target)}


def _update_actor_alpha_impl(
    static: SACStatic,
    dyn: SACDynamic,
    rng: jax.Array,
    obs: jax.Array,
) -> tuple[SACDynamic, dict]:
    # actor loss
    def actor_loss(actor_params, log_alpha):
        alpha = jnp.exp(log_alpha)
        act, log_prob = nnx.merge(static.actor_def, actor_params)(obs, rng)
        q1 = nnx.merge(static.q1_def, dyn.q1_params)(obs, act)
        q2 = nnx.merge(static.q2_def, dyn.q2_params)(obs, act)
        return jnp.mean(alpha * log_prob - jnp.minimum(q1, q2)), log_prob

    (a_loss, log_prob), a_grad = jax.value_and_grad(actor_loss, argnums=0, has_aux=True)(
        dyn.actor_params, dyn.log_alpha
    )
    a_u, a_o = static.actor_tx.update(a_grad, dyn.actor_opt, dyn.actor_params)
    a_p = optax.apply_updates(dyn.actor_params, a_u)

    # alpha loss
    def alpha_loss(log_alpha):
        return -jnp.mean(jnp.exp(log_alpha) * (jax.lax.stop_gradient(log_prob) + static.target_entropy))

    al_l, al_g = jax.value_and_grad(alpha_loss)(dyn.log_alpha)
    al_u, al_o = static.alpha_tx.update(al_g, dyn.alpha_opt)
    new_log_alpha = optax.apply_updates(dyn.log_alpha, al_u)

    new_dyn = dataclasses.replace(dyn, actor_params=a_p, actor_opt=a_o, log_alpha=new_log_alpha, alpha_opt=al_o)
    return new_dyn, {
        "actor_loss": a_loss, "alpha": jnp.exp(new_log_alpha),
        "alpha_loss": al_l, "log_prob": jnp.mean(log_prob),
    }


def make_update_fns(static: SACStatic, gamma: float = 0.99):
    """Return JIT-compiled update functions with static captured via partial."""

    @jax.jit
    def update_critic(dyn, rng, obs, act, rew, next_obs, done):
        return _update_critic_impl(static, dyn, rng, obs, act, rew, next_obs, done, gamma)

    @jax.jit
    def update_actor_alpha(dyn, rng, obs):
        return _update_actor_alpha_impl(static, dyn, rng, obs)

    return update_critic, update_actor_alpha


@jax.jit
def _select_action(actor_def, actor_params, obs, rng):
    return nnx.merge(actor_def, actor_params)(obs, rng)[0]


def select_action(static: SACStatic, dyn: SACDynamic, obs: jax.Array, rng: jax.Array):
    return _select_action(static.actor_def, dyn.actor_params, obs, rng)


# =========================================================================
# 6. Training loop
# =========================================================================

def train(
    domain: str = "pendulum",
    task: str = "swingup",
    total_steps: int = 50_000,
    warmup: int = 1_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    eval_every: int = 5_000,
    eval_episodes: int = 5,
    seed: int = 0,
    target_return: float | None = None,
):
    env = DMCEnv(domain, task, seed=seed)
    eval_env = DMCEnv(domain, task, seed=seed + 100)
    print(f"Env: {domain}-{task}  obs_dim={env.obs_dim}  act_dim={env.act_dim}")

    buf = ReplayBuffer(capacity=100_000, obs_dim=env.obs_dim, act_dim=env.act_dim)
    np_rng = np.random.default_rng(seed)
    rng = jax.random.key(seed)

    static, dyn = init_sac(env.obs_dim, env.act_dim, seed=seed)
    update_critic, update_actor_alpha = make_update_fns(static, gamma)

    obs = env.reset()
    ep_return, ep_len = 0.0, 0
    best_eval = -float("inf")
    t0 = time.time()
    c_info = {}

    for step in range(1, total_steps + 1):
        # ── collect ──────────────────────────────────────────────────
        if step < warmup:
            action = np_rng.uniform(-1, 1, size=env.act_dim).astype(np.float32)
        else:
            rng, act_rng = jax.random.split(rng)
            action = np.asarray(select_action(static, dyn, jnp.asarray(obs), act_rng))

        next_obs, reward, done = env.step(action)
        buf.add(obs, action, reward, next_obs, done)
        ep_return += reward
        ep_len += 1
        obs = next_obs if not done else env.reset()
        if done:
            ep_return, ep_len = 0.0, 0

        # ── update ───────────────────────────────────────────────────
        if step >= warmup:
            o, a, r, no, d = buf.sample(batch_size, np_rng)
            rng, c_rng, a_rng = jax.random.split(rng, 3)
            dyn, c_info = update_critic(dyn, c_rng, o, a, r, no, d)
            dyn, a_info = update_actor_alpha(dyn, a_rng, o)

        # ── eval ─────────────────────────────────────────────────────
        if step % eval_every == 0:
            eval_rets = []
            for _ in range(eval_episodes):
                rng, eval_rng = jax.random.split(rng)
                e_obs, e_ret = eval_env.reset(), 0.0
                for _ in range(1000):
                    eval_rng, sel_rng = jax.random.split(eval_rng)
                    e_act = np.asarray(select_action(static, dyn, jnp.asarray(e_obs), sel_rng))
                    e_obs, e_rew, e_done = eval_env.step(e_act)
                    e_ret += e_rew
                    if e_done:
                        break
                eval_rets.append(e_ret)
            mean_ret = np.mean(eval_rets)
            best_eval = max(best_eval, mean_ret)
            elapsed = time.time() - t0
            q1l = float(c_info.get("q1_loss", 0.0))
            alph = float(jnp.exp(dyn.log_alpha))
            print(
                f"step {step:>6d} | eval {mean_ret:7.1f} (best {best_eval:7.1f}) | "
                f"α {alph:5.3f} | q_loss {q1l:7.3f} | buf {buf.size:>6d} | {elapsed:5.0f}s"
            )
            if target_return is not None and mean_ret >= target_return:
                print(f"\n✅ Solved! Target {target_return} reached at step {step}.")
                return True, best_eval

    solved = target_return is not None and best_eval >= target_return
    print(f"\n{'✅ Solved!' if solved else '❌ Not solved.'}  Best eval: {best_eval:.1f}")
    return solved, best_eval


# =========================================================================
# 7. Main
# =========================================================================

DEFAULTS = {
    "pendulum-swingup":  dict(total_steps=50_000,  target_return=600),
    "cartpole-swingup":  dict(total_steps=100_000, target_return=600),
    "reacher-easy":      dict(total_steps=100_000, target_return=800),
    "cheetah-run":       dict(total_steps=200_000, target_return=400),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="pendulum-swingup")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    domain, task = args.env.split("-", 1)
    d = DEFAULTS.get(args.env, {})
    train(
        domain=domain, task=task,
        total_steps=args.steps or d.get("total_steps", 100_000),
        target_return=args.target or d.get("target_return"),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()