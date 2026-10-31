"""Shared CPU stubs for the pi05 runner tests (not collected: leading underscore).

TaskEnv     episode fully determined by (task, reset seed); state = [task, seed, t, 0...].
StubAgent   records every call into a shared log; each update returns a NEW instance.
            strict=True: the replaced instance raises on any later use, as `donate_agent` frees it.
Logger      records (prefix, step, payload).
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

IMG = 4
H = 3  # stub chunk length; episode lengths 3..6 -> slots finish mid-chunk at different times


def ep_len(seed):
    return 3 + seed % 4


class TaskEnv:
    def __init__(self, task, reg):
        self.task, self.reg, self.t, self.s = task, reg, 0, None
        reg['built'].append(self)

    def _obs(self):
        st = np.zeros(8, np.float32)
        st[0], st[1], st[2] = self.task, self.s, self.t
        img = np.full((IMG, IMG, 3), (7 * self.s + self.t) % 256, np.uint8)
        return {'image': img, 'wrist_image': img.copy(), 'state': st}

    def reset(self, seed=None):
        self.reg['resets'].append((self.task, seed))
        self.s, self.t = seed, 0
        return self._obs(), {}

    def step(self, action):
        assert action.shape == (7,)
        assert self.t < ep_len(self.s), "stepped a finished episode"
        self.reg['actions'].append((self.task, self.s, self.t, np.array(action)))
        self.t += 1
        end = self.t >= ep_len(self.s)
        succ = float(self.s % 2 == 0) if end else 0.0
        return self._obs(), -1.0, bool(end and succ), bool(end and not succ), {'success': succ}


def reg():
    return {'built': [], 'resets': [], 'actions': []}


class StaleAgentError(RuntimeError):
    pass


def _copy(d):
    return None if d is None else {k: np.array(v) for k, v in d.items()}


class StubAgent:
    """Chunk row j = [task, seed, t_replan + h, key-dependent, 0...] from the critic features (= raw state)."""
    env_action_dim = 7  # libero profile (OGPOPi05Agent.env_action_dim / clip_env_actions)
    clip_env_actions = True

    def __init__(self, log, strict=False, gen=0):
        self.log, self.strict, self.gen, self.dead = log, strict, gen, False
        self.actor_params = jnp.zeros(1)
        self.critic_network = SimpleNamespace(params=jnp.zeros(1))

    def _use(self, name):
        if self.dead:
            raise StaleAgentError(f"{name} called on agent gen {self.gen} after it was replaced by an update")

    def _next(self):
        new = StubAgent(self.log, self.strict, self.gen + 1)
        if self.strict:
            self.dead = True
        return new

    @staticmethod
    def _key(key):
        return np.asarray(key).copy()

    def encode(self, imgs, states, task_ids=None):
        self._use('encode')
        self.log.append(('encode', np.asarray(imgs)[:, 0, 0, 0, 0].copy(), np.asarray(states).copy(),
                         None if task_ids is None else np.asarray(task_ids).copy()))
        return np.asarray(states, np.float32)

    def _chunk(self, feats, key):
        f = np.asarray(feats)
        out = np.zeros((f.shape[0], H, 7), np.float32)
        out[..., 0], out[..., 1] = f[:, None, 0], f[:, None, 1]
        out[..., 2] = f[:, None, 2] + np.arange(H)[None]
        out[..., 3] = (int(np.asarray(key).ravel()[-1]) % 997) / 997.0
        return out

    def sample_actions_BON(self, imgs, feats, key, task_ids=None):
        self._use('sample_actions_BON')
        self.log.append(('bon', np.asarray(imgs)[:, 0, 0, 0, 0].copy(), np.asarray(feats).copy(), self._key(key),
                         None if task_ids is None else np.asarray(task_ids).copy()))
        return self._chunk(feats, key)

    def sample_actions_ode(self, imgs, key, task_ids=None, critic_obs=None):
        self._use('sample_actions_ode')
        self.log.append(('ode', np.asarray(imgs)[:, 0, 0, 0, 0].copy(), self._key(key),
                         None if task_ids is None else np.asarray(task_ids).copy()))
        out = np.zeros((imgs.shape[0], H, 7), np.float32)
        out[..., 3] = (int(np.asarray(key).ravel()[-1]) % 997) / 997.0
        return out

    def update(self, batch, batch_success, bc_batch, success_flag, ppo_batch=None):
        self._use('update')
        self.log.append(('update', _copy(batch), _copy(batch_success), _copy(bc_batch), bool(success_flag),
                         _copy(ppo_batch)))
        info = {'actor/pg_loss': float(np.sum(bc_batch['actions'])), 'critic/td_loss': float(np.sum(batch['rewards'])),
                'actor/adv_std': float(self.gen)}
        return self._next(), info

    def critic_update(self, batch, batch_success, success_flag):
        self._use('critic_update')
        self.log.append(('critic_update', _copy(batch), _copy(batch_success), None, bool(success_flag), None))
        return self._next(), {'critic/td_loss': float(np.sum(batch['rewards']))}

    def grad_diagnostics(self, batch, bc_batch):
        self._use('grad_diagnostics')
        assert batch is not None and bc_batch is not None, "grad diagnostics need an actor step's batches"
        self.log.append(('diag', _copy(batch), _copy(bc_batch)))
        return {'grad_pg_norm': float(np.sum(batch['rewards'])), 'grad_bc_norm': float(np.sum(bc_batch['actions']))}


class Logger:
    def __init__(self):
        self.logs = []

    def log(self, data, prefix, step):
        self.logs.append((prefix, step, dict(data)))


TIMING_KEYS = {'sec_per_update', 'collect_minutes'}


def strip_timing(logs):
    return [(p, s, {k: v for k, v in d.items() if k not in TIMING_KEYS}) for p, s, d in logs]


def assert_logs_equal(a, b):
    a, b = strip_timing(a), strip_timing(b)
    assert [(p, s) for p, s, _ in a] == [(p, s) for p, s, _ in b]
    for (p, s, da), (_, _, db) in zip(a, b):
        assert da.keys() == db.keys(), (p, s, da.keys() ^ db.keys())
        for k in da:
            assert da[k] == db[k] or (np.isnan(da[k]) and np.isnan(db[k])), (p, s, k, da[k], db[k])


def _eq(x, y):
    if isinstance(x, dict) or isinstance(y, dict):
        if x is None or y is None:
            return x is y
        return x.keys() == y.keys() and all(_eq(x[k], y[k]) for k in x)
    if x is None or y is None:
        return x is y
    if isinstance(x, (bool, str)):
        return x == y
    return np.array_equal(np.asarray(x), np.asarray(y))


def assert_call_logs_equal(a, b):
    assert [c[0] for c in a] == [c[0] for c in b]
    for i, (ca, cb) in enumerate(zip(a, b)):
        assert len(ca) == len(cb)
        for j, (x, y) in enumerate(zip(ca, cb)):
            assert _eq(x, y), (i, ca[0], j)


def assert_buffers_equal(rb_a, rb_b):
    assert rb_a.size == rb_b.size and set(rb_a._dict) == set(rb_b._dict)
    for k in rb_a._dict:
        assert np.array_equal(rb_a[k][:rb_a.size], rb_b[k][:rb_b.size]), k
    assert np.array_equal(rb_a._traj_success_mask[:rb_a.size], rb_b._traj_success_mask[:rb_b.size])


def episodes_in_buffer(rb):
    """Split the buffer into episodes, checking each is contiguous and self-consistent.

    -> list of (task, seed). Every transition: obs = (task, seed, t), next obs t + 1, task_ids = task,
    action planned from that obs (action[:3] = (task, seed, t)), terminal only on the last step.
    """
    obs, nobs = rb['observations'][:rb.size], rb['next_observations'][:rb.size]
    acts, tids, term = rb['actions'][:rb.size], rb['task_ids'][:rb.size], rb['terminals'][:rb.size]
    imgs, nimgs = rb['images'][:rb.size], rb['next_images'][:rb.size]
    i, eps = 0, []
    while i < rb.size:
        task, s = int(obs[i, 0]), int(obs[i, 1])
        n = ep_len(s)
        for t in range(n):
            j = i + t
            assert (obs[j, 0], obs[j, 1], obs[j, 2]) == (task, s, t), (j, obs[j, :3])
            assert (nobs[j, 0], nobs[j, 1], nobs[j, 2]) == (task, s, t + 1)
            assert tids[j] == task
            assert (acts[j, 0], acts[j, 1], acts[j, 2]) == (task, s, t), (j, acts[j, :3])
            assert imgs[j, 0, 0, 0, 0] == (7 * s + t) % 256 and nimgs[j, 0, 0, 0, 0] == (7 * s + t + 1) % 256
            assert term[j] == (t == n - 1)
        eps.append((task, s))
        i += n
    return eps
