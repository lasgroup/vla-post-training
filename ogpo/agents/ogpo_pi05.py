"""OGPO with a pi0.5 action expert as the policy.

Mirrors the reference OGPO path in `ogpo.py` (vanilla advantages, PPO on SDE
chains from the target actor, success-buffer BC anchor, TD critic with an SDE
next action, success-buffer critic update, best-of-N rollouts), with the MLP
actor replaced by a pi05 checkpoint (`pi05_robot`: libero = pi05_libero, droid = pi05_droid_jointpos):

  * PaliGemma (SigLIP + Gemma-2B) is frozen (bf16); only the action expert and
    its heads are trained, with vla-post-training's AdamW settings.
  * The prefix runs once per state; its KV cache is shared by every flow step
    and every group / best-of-N sample.
  * The policy chunk is `pi05_action_horizon` x 32 (quantile-normalised, padded); the first
    `horizon_length` steps x `env_action_dim` real dims (libero 7, droid 8) are executed. SDE noise, logprob and
    the critic only see those executed dims (`dim_mask`).
  * Env actions (buffer, BC targets, what the env executes) are absolute. droid's model space is a joint delta
    (dims 0-6) relative to the state at the window / chunk start, recovered from the critic features
    (`_window_state`); `_to_env_actions` / `_normalize_env_actions` convert. libero has no delta dims.
  * OGPO's flow convention (t: 0 noise -> 1 data, v = a - x0) is mapped onto
    openpi's (tau = 1 - t, v_pi = x0 - a) in `_actor_fn`.

Critic input: [pooled Gemma prefix (`pi05_critic_pooling`: 2048 / 4096 / 6144) |
normalised state (8)] via OGPO's two-tier encoder, plus the normalised executed chunk, into
`num_qs` heads: MLP over `value_hidden_dims`, or BroNet (`pi05_critic_arch: bronet`, as vla-post-training).
`pi05_critic_arch: bronet_vla` drops the two-tier encoder: `num_qs` independent BroNets on the raw
[prefix mean 2048 | state 8 | action], the architecture of vla-post-training's `BroNetStateActionCritic`.

Best-of-N acting (`sample_actions_BON`): `best_of_n` candidates from the target actor, scored by the
target critic. `pi05_bon_sampler`: sde (SDE chains, as OGPO) | ode (unclipped Euler ODE, as
vla-post-training's BoN). `pi05_bon_agg`: head reduction min | subsample | mean (`q_helper.select_best_of_n`);
the pi05 agent does not read `subsample_bon`.

SARSA-V critic (`use_value_fn: true`, `v_training_mode: sarsa`; vla-post-training's BoN critic update): a
state-value ensemble V (`value_network`, `v_ensemble_size` heads, same architecture as the critic without the
action input, own optimiser and EMA target). Per batch, a Q step Q(s, a_buf) <- r + gamma^H mask v_agg(V_tgt(s'))
(no policy sample, no prefix pass on next_images), then a V step V(s) <- v_agg(Q_tgt(s, a_buf)) with the target
critic after the Q step's Polyak update. `value_network` is None otherwise (default path unchanged).

AWR (`pi05_actor_objective: awr`; Peng et al. 2019, Alg. 1): the actor objective is advantage-weighted flow
matching on replay-buffer windows, mean_b[min(exp(A_b / awr_beta), awr_weight_max) * CFM_b], with
A = R^λ(s) - V(s). R^λ is the TD(λ) return the runner writes into the buffer each round (`lambda_returns`,
bootstrapped with the previous round's V); V regresses onto it (`v_training_mode: lambda_return`,
`value_update`), then the actor steps (`actor_update`). No Q, PPO, SDE chains or BC anchor: the critic network is
still built and checkpointed but never trained or read. The runner owns the round schedule
(`run_pi05_round_rl`) and collects with `sample_actions_explore` (one unclipped ODE sample, online actor).

Group AWR (`pi05_actor_objective: group_awr`): OGPO with the PPO term swapped for the AWR paper's loss. Same G SDE
samples from the target actor, same group-relative advantages (`compute_ogpo_advantages`), same critic, BC anchor,
collection and eval; the actor term is mean_n[min(exp(A_n / awr_beta), awr_weight_max) * CFM_n] on the sampled chunks
(`_group_awr_loss`). No log-prob, ratio or entropy.

Data parallel (`utils.sharding.shard_pi05_agent` sets the static `mesh`): the PPO states subsampled from the sharded
TD batch are constrained back to `P('data')` (the gather returns them replicated, i.e. every device would run every
state's chains). `zero_actor_state` (ZeRO-1, opt-in): the actor's optimiser state and init copy stay sharded
leaf-wise (`utils.sharding.zero_spec`) across updates; params used in forward / backward stay replicated.
"""

import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax.sharding import NamedSharding, PartitionSpec as P

from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.agents.modules.maintenance import create_lr_schedule, create_optimizer
from ogpo.agents.modules.pg_helper import (
    compute_awr_weights,
    compute_flow_log_prob,
    compute_ogpo_advantages,
    compute_ppo_loss,
    sample_flow_actions_ode,
    sample_flow_actions_sde,
)
from ogpo.agents.modules.q_helper import (
    aggregate_q_values,
    bon_scores,
    compute_td_loss,
    compute_td_target,
    reduce_q_over_samples,
    select_best_of_n,
)
from ogpo.networks import Value
from ogpo.networks.pi05 import transforms as T
from ogpo.networks.pi05.pi05 import IMAGE_KEYS, Pi05, load_pi05_params, merge_params, split_params
from ogpo.utils.sharding import zero_shardings

PREFIX_DIM = 2048
STATE_DIM = 8
# Critic feature width per `pi05_critic_pooling` mode (number of pooled 2048-D segments).
POOLING_SEGMENTS = {'all': 1, 'segmented': 2, 'per_camera': 3}


def _tile(tree, n, axis):
    return jax.tree.map(lambda x: jnp.concatenate([x] * n, axis=axis), tree)


def _per_task(stats, task_ids, num_tasks):
    """{name: (N,) values} + task ids (N,) -> {name/task{k}: masked mean} for K > 1 (NaN when a task is absent)."""
    if num_tasks <= 1 or task_ids is None:
        return {}
    out = {}
    for k in range(num_tasks):
        m = (task_ids == k).astype(jnp.float32)
        n = m.sum()
        for name, v in stats.items():
            out[f'{name}/task{k}'] = jnp.where(n > 0, (v * m).sum() / jnp.maximum(n, 1), jnp.nan)
    return out


class OGPOPi05Agent(flax.struct.PyTreeNode):
    rng: Any
    actor_params: Any
    target_actor_params: Any
    actor_opt_state: Any
    init_actor_params: Any  # pi05_libero weights, for drift logging only
    frozen_params: Any
    critic_network: TrainState
    consts: Any  # tokens / norm stats / dim mask as jnp arrays
    config: Any = nonpytree_field()
    model: Any = nonpytree_field()
    actor_tx: Any = nonpytree_field()
    value_network: Any = None  # SARSA-V state-value TrainState (use_value_fn), else None
    mesh: Any = nonpytree_field(default=None)  # data-parallel mesh (shard_pi05_agent); None = single device
    zero_actor_state: bool = nonpytree_field(default=False)  # ZeRO-1 actor opt state / init copy (shard_pi05_agent)
    robot: str = nonpytree_field(default='libero')  # T.ROBOTS key; create() sets it from the validated pi05_robot

    # ------------------------------------------------------------------
    # pi05 plumbing
    # ------------------------------------------------------------------

    @property
    def _chunk_shape(self):
        return (self.config['pi05_action_horizon'], T.MODEL_ACTION_DIM)

    @property
    def _flat_dim(self):
        h, d = self._chunk_shape
        return h * d

    @property
    def env_action_dim(self):
        return T.ROBOTS[self.robot]['env_action_dim']

    @property
    def clip_env_actions(self):
        return T.ROBOTS[self.robot]['clip_env_actions']

    def _window_state(self, critic_obs):
        """(N, F) critic features -> (N, 8) raw state at the window / chunk start (`encode` puts the normalised
        state last). The delta reference for droid's joint actions."""
        q01, q99 = self.consts['state_q01'], self.consts['state_q99']
        return (critic_obs[..., -STATE_DIM:] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def _model_images(self, images, drop_empty=False):
        """(B, 2, 224, 224, 3) uint8 -> openpi image / mask dicts. drop_empty: leave out the right-wrist slot (an all -1
        image, always masked)."""
        x = images.astype(jnp.float32) / 255.0 * 2.0 - 1.0
        b = images.shape[0]
        imgs = {'base_0_rgb': x[:, 0], 'left_wrist_0_rgb': x[:, 1], 'right_wrist_0_rgb': jnp.full_like(x[:, 0], -1.0)}
        masks = {'base_0_rgb': jnp.ones(b, bool), 'left_wrist_0_rgb': jnp.ones(b, bool), 'right_wrist_0_rgb': jnp.zeros(b, bool)}
        if drop_empty:
            del imgs['right_wrist_0_rgb'], masks['right_wrist_0_rgb']
        return imgs, masks

    def _prefix(self, images, task_ids=None, kv_only=False):
        """Frozen VLM pass -> (prefix_out, kv_cache, prefix_mask), no gradient.

        `consts['tokens']` is a (K, L) table, one tokenised prompt per task; `task_ids` (B,) int
        selects the row per sample (multi-task). None -> row 0 for the whole batch (single task).

        kv_only=True (callers that use only the KV cache for the action expert): prefix_out is returned as None, and with
        `pi05_drop_empty_camera` the always-masked right-wrist slot is left out of the pass (~1/3 fewer prefix tokens).
        Masked tokens are never attended to and do not advance positions (`Pi05.prefix`), so the valid tokens' KV and the
        velocities are unchanged. The pooled critic features (`encode`) average over every prefix token including the
        masked ones, so they always take the full prefix (kv_only=False).
        """
        b = images.shape[0]
        imgs, masks = self._model_images(images, drop_empty=kv_only and self.config.get('pi05_drop_empty_camera', False))
        if task_ids is None:
            tokens = jnp.broadcast_to(self.consts['tokens'][0], (b, self.consts['tokens'].shape[-1]))
            token_mask = jnp.broadcast_to(self.consts['token_mask'][0], tokens.shape)
        else:
            tokens = self.consts['tokens'][task_ids]
            token_mask = self.consts['token_mask'][task_ids]
        prefix_out, kv_cache, prefix_mask = self.model.apply(
            {'params': merge_params(self.frozen_params, self.target_actor_params)},
            imgs, masks, tokens, token_mask, method=Pi05.prefix)
        return jax.lax.stop_gradient((None if kv_only else prefix_out, kv_cache, prefix_mask))

    def _actor_fn(self, params, kv_cache, prefix_mask):
        """pg_helper-compatible actor_fn in OGPO's flow convention."""
        full = merge_params(self.frozen_params, params)
        h, d = self._chunk_shape

        def actor_fn(obs, x, t, params=None, is_encoded=True):
            n = x.shape[0]
            v_pi = self.model.apply(
                {'params': full}, kv_cache, prefix_mask, x.reshape(n, h, d), 1.0 - t[:, 0], method=Pi05.velocity)
            return -v_pi.reshape(n, h * d).astype(jnp.float32)

        return actor_fn

    def _sde_kwargs(self):
        c = self.config
        return dict(
            flow_steps=c['flow_steps'],
            use_constant_noise=c['use_constant_noise'],
            use_tapered_noise=c.get('use_tapered_noise', False),
            error_correct_sde_to_ode=c.get('error_correct_sde_to_ode', False),
            constant_noise_std=c['constant_noise_std'],
            min_noise_std=c['min_noise_std'],
            clip_intermediate=c['clip_intermediate_actions'],
            clip_value=c['denoised_clip_value'],
            ft_flow_steps=c.get('ft_flow_steps', 0),
            dim_mask=self.consts['dim_mask'],
        )

    def _sample_sde(self, params, kv_cache, prefix_mask, rng):
        """SDE chains (normalised model space). Mirrors `sample_actions_with_noise`."""
        n = prefix_mask.shape[0]
        c = self.config
        actions, chains, logprob, sigmas = sample_flow_actions_sde(
            actor_fn=self._actor_fn(params, kv_cache, prefix_mask), noise_fn=None,
            observations=jnp.zeros((n, 1)), rng=rng, act_dim=self._flat_dim,
            randn_clip_value=c['randn_clip_value'], act_min=c['act_min'], act_max=c['act_max'],
            is_encoded=True, params=None, **self._sde_kwargs())
        ft = c.get('ft_flow_steps', 0)
        effective_ft = ft if (0 < ft < c['flow_steps']) else c['flow_steps']
        logprob_steps = effective_ft + 1 if effective_ft == c['flow_steps'] else effective_ft
        if c['normalize_denoising_horizon']:
            logprob = logprob / logprob_steps
        if c['normalize_act_space_dimension']:
            logprob = logprob / self.consts['dim_mask'].sum()
        return actions, chains, logprob, sigmas

    def _log_prob(self, params, kv_cache, prefix_mask, chains, get_entropy):
        kw = self._sde_kwargs()
        kw.pop('clip_intermediate'), kw.pop('clip_value')
        return compute_flow_log_prob(
            actor_fn=self._actor_fn(params, kv_cache, prefix_mask), noise_fn=None,
            observations=jnp.zeros((chains.shape[0], 1)), chains=chains,
            normalize_horizon=self.config['normalize_denoising_horizon'],
            normalize_dim=self.config['normalize_act_space_dimension'],
            is_encoded=True, params=None, get_entropy=get_entropy,
            clip_intermediate=self.config['clip_intermediate_actions'],
            clip_value=self.config['denoised_clip_value'], **kw)

    def _critic_actions(self, flat_actions):
        """Model-space chunk (N, H_pi*32) -> executed, real dims (N, H*A)."""
        h, d = self._chunk_shape
        x = flat_actions.reshape(-1, h, d)[:, :self.config['horizon_length'], :self.env_action_dim]
        return x.reshape(x.shape[0], -1)

    def _normalize_env_actions(self, env_actions, critic_obs=None):
        """(B, T, A) env actions -> (B, T, A) model space (quantile-normalised). droid: dims 0-6 become deltas to the
        window-start state from `critic_obs` (B, F) first (openpi DeltaActions)."""
        a = self.env_action_dim
        k = T.ROBOTS[self.robot]['delta_dims']
        if k:
            assert critic_obs is not None, "delta actions need the window-start critic features"
            env_actions = env_actions.at[..., :k].add(-self._window_state(critic_obs)[:, None, :k])
        q01, q99 = self.consts['action_q01'][:a], self.consts['action_q99'][:a]
        return (env_actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def _to_env_actions(self, flat_actions, critic_obs=None):
        """(N, H_pi*32) model space -> (N, H, A) executed env actions. droid: dims 0-6 are deltas, so the chunk-start
        state from `critic_obs` (N, F) is added (openpi AbsoluteActions)."""
        h, d = self._chunk_shape
        a = self.env_action_dim
        k = T.ROBOTS[self.robot]['delta_dims']
        x = flat_actions.reshape(-1, h, d)[:, :self.config['horizon_length'], :a]
        q01, q99 = self.consts['action_q01'][:a], self.consts['action_q99'][:a]
        x = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        if k:
            assert critic_obs is not None, "delta actions need the chunk-start critic features"
            x = x.at[..., :k].add(self._window_state(critic_obs)[:, None, :k])
        return x

    def _target_q(self, critic_obs, actions, rng):
        """Target-critic q_agg / q_full for one action group. Mirrors `compute_q_for_group`."""
        q_vals = self.critic_network.select('target_critic')(critic_obs, actions, is_encoded=True)
        if self.config['q_agg'] == 'min':
            q_agg = q_vals.min(axis=0)
        elif self.config['q_agg'] == 'subsample':
            idxs = jax.random.randint(rng, (2,), 0, self.config['num_qs'])
            q_agg = q_vals[idxs].min(axis=0)
        else:
            q_agg = q_vals.mean(axis=0)
        return q_agg, q_vals

    def _state_value(self, observations):
        """(N, F) critic features -> (N,) online V, heads reduced by `v_agg` (mean | min; AWR's baseline / bootstrap)."""
        vs = self.value_network.select('value')(observations, is_encoded=True)
        return aggregate_q_values(vs, method=self.config['v_agg'])

    @jax.jit
    def state_values(self, observations):
        """Jitted `_state_value` for the runner's per-round λ-return pass."""
        return self._state_value(observations)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _bc_loss_per_sample(self, bc_batch, grad_params, rng):
        """(B,) masked flow-matching error per window; `bc_loss` is its mean, per-task diagnostics use it directly."""
        _, kv, pmask = self._prefix(bc_batch['images'], bc_batch.get('task_ids'), kv_only=True)
        # observations = the window-start critic features (droid delta reference); normalise, then zero-pad (openpi order)
        env_actions = self._normalize_env_actions(bc_batch['actions'], bc_batch.get('observations'))
        a = jnp.pad(env_actions, ((0, 0), (0, 0), (0, T.MODEL_ACTION_DIM - self.env_action_dim)))
        err = self._cfm_err(a.reshape(a.shape[0], -1), kv, pmask, grad_params, rng)
        return (err * bc_batch['valid'][..., None]).mean(axis=(1, 2))

    def _cfm_err(self, a, kv, pmask, grad_params, rng):
        """(N, H_pi*32) model-space chunks -> (N, H_pi, 32) squared velocity error at one (t, x0) draw per chunk.
        The flow-matching core shared by the BC anchor, `awr` and `group_awr`; callers mask and reduce."""
        b = a.shape[0]
        x0_rng, t_rng = jax.random.split(rng)
        x0 = jax.random.normal(x0_rng, a.shape)
        t = jax.random.uniform(t_rng, (b, 1))
        x_t = (1 - t) * x0 + t * a
        pred = self._actor_fn(grad_params, kv, pmask)(None, x_t, t)
        return jnp.square(pred - (a - x0)).reshape(b, *self._chunk_shape)

    def bc_loss(self, bc_batch, grad_params, rng):
        """Flow-matching BC on (B, H_pi, A) env-action windows, masked by `valid` (batch mean of the per-sample loss)."""
        return self._bc_loss_per_sample(bc_batch, grad_params, rng).mean()

    def actor_loss(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        pg_loss, ent, bc_loss, info = self._actor_terms(batch, bc_batch, grad_params, rng, ppo_batch)
        c = self.config
        return pg_loss - c['entropy_coeff'] * ent + c['bc_coeff'] * bc_loss, info

    def _actor_terms(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        """(pg_loss, entropy, bc_loss, info); split out so `grad_diagnostics` can differentiate each.

        PPO states come from `ppo_batch` when given (the runner's task-stratified batch), else from a random
        ppo_batch_size subsample of the TD batch (original behaviour).

        AWR (`pi05_actor_objective: awr`) returns before any of the PPO machinery runs: see `_awr_terms`.
        `group_awr` shares the samples and advantages below and swaps only the PPO term: see `_group_awr_loss`.
        """
        if self.config.get('pi05_actor_objective', 'ppo') == 'awr':
            return self._awr_terms(bc_batch, grad_params, rng)
        c = self.config
        G = c['group_num_samples']
        if ppo_batch is not None:
            images, c_obs, task_ids = ppo_batch['images'], ppo_batch['observations'], ppo_batch.get('task_ids')
        else:
            images = batch['images']
            c_obs = batch['observations']
            task_ids = batch.get('task_ids')
            if c['ppo_batch_size'] < c['batch_size']:
                rng, key = jax.random.split(rng)
                n_ppo = c['ppo_batch_size'] // self._accum_steps  # per micro-batch under grad accumulation
                idxs = jax.random.choice(key, images.shape[0], (n_ppo,), replace=False)
                images, c_obs = images[idxs], c_obs[idxs]
                task_ids = None if task_ids is None else task_ids[idxs]
                if self.mesh is not None:
                    # The gather returns the subsample replicated: without this every device would run all
                    # n_ppo states x G chains. n_ppo divides over the devices (main_pi05 checks ppo_batch_size).
                    shard = NamedSharding(self.mesh, P('data'))
                    images, c_obs = (jax.lax.with_sharding_constraint(x, shard) for x in (images, c_obs))
                    task_ids = None if task_ids is None else jax.lax.with_sharding_constraint(task_ids, shard)
        B = images.shape[0]

        _, kv, pmask = self._prefix(images, task_ids, kv_only=True)
        kv_g, pmask_g = _tile(kv, G, axis=1), _tile(pmask, G, axis=0)  # index g * B + b

        rng, sample_rng, adv_rng, bc_rng = jax.random.split(rng, 4)
        actions, chains, old_lp, sigmas = self._sample_sde(self.target_actor_params, kv_g, pmask_g, sample_rng)
        actions_g = self._critic_actions(actions).reshape(G, B, -1)

        q_agg, q_full = jax.vmap(lambda a_g, r_g: self._target_q(c_obs, a_g, r_g))(
            actions_g, jax.random.split(adv_rng, G))
        adv, adv_pre = compute_ogpo_advantages(q_agg, q_full, c)

        if c.get('pi05_actor_objective', 'ppo') == 'group_awr':
            # `rng` is left over from the 4-way split and unused by the PPO branch, so PPO's keys are unchanged.
            pg_loss, pg_info = self._group_awr_loss(grad_params, kv_g, pmask_g, actions, adv.reshape(-1), rng)
            ent = jnp.zeros((), jnp.float32)
        else:
            lp, entropy_rate, lp_info = self._log_prob(grad_params, kv_g, pmask_g, chains, get_entropy=True)
            pg_loss, pg_stats = compute_ppo_loss(lp, old_lp, adv.reshape(-1), c)
            ent = jnp.mean(entropy_rate)
            pg_info = {
                'lp': lp.mean(), 'lp_min': lp.min(), 'lp_max': lp.max(), 'old_lp': old_lp.mean(),
                **pg_stats, **{f'info_{k}': v for k, v in lp_info.items()},
            }

        K = self.consts['tokens'].shape[0]
        adv_g = adv.reshape(G, B)  # per-state stats over the group
        per_task = _per_task({
            'adv_abs': jnp.abs(adv_g).mean(axis=0), 'adv_zero_frac': (adv_g == 0).mean(axis=0),
            'q_values': adv_pre.mean(axis=0), 'q_group_std': jnp.std(adv_pre, axis=0),
        }, task_ids, K)
        if c['use_bc_regularization']:
            bc_per_sample = self._bc_loss_per_sample(bc_batch, grad_params, bc_rng)
            bc_loss = bc_per_sample.mean()
            per_task.update(_per_task({'bc_loss': bc_per_sample}, bc_batch.get('task_ids'), K))
        else:
            bc_loss = 0.0
        info = {
            **per_task,
            'pg_loss': pg_loss, 'bc_loss': bc_loss, 'entropy': ent,
            'adv': adv.mean(), 'adv_min': adv.min(), 'adv_max': adv.max(), 'adv_std': adv.std(),
            'adv_zero_frac': jnp.mean(adv == 0),  # conservative: heads disagreed on the sign
            'q_values': adv_pre.mean(), 'sigmas': sigmas.mean(),
            'actions': actions_g.mean(), 'actions_std': actions_g.std(),
            'actions_clipped_frac': jnp.mean(jnp.abs(actions_g) >= c['act_max']),
            'grpo_adv_group_std': jnp.std(adv_pre, axis=0).mean(),
            'grpo_adv_between_group_std': jnp.std(adv_pre.mean(axis=0)),
            **pg_info,
        }
        return pg_loss, ent, bc_loss, info

    def _group_awr_loss(self, grad_params, kv_g, pmask_g, actions, adv, rng):
        """`group_awr` actor term on OGPO's own samples: (loss, info).

        loss = mean_n[w_n * CFM_n], w = min(exp(A / awr_beta), awr_weight_max) (clip after the exp, no batch
        normalisation: Peng et al. 2019), A the OGPO group-relative advantage of SDE sample n (no gradient).
        CFM_n is the flow-matching error on the sampled model-space chunk, averaged over `awr_n_mc` (t, x0) draws and
        over the executed real dims (`dim_mask`, the dims the PPO log-prob sees). The draws are an unrolled loop, as
        `compute_flow_log_prob`'s steps: a vmap would broadcast the G-tiled KV cache awr_n_mc times.

        actions: (N, H_pi*32) model-space SDE samples, N = G*B (index g * B + b); adv: (N,).
        """
        c = self.config
        a = jax.lax.stop_gradient(actions)
        mask = self.consts['dim_mask'].reshape(self._chunk_shape)
        n_mc = c['awr_n_mc']
        cfm = 0.0
        for key in jax.random.split(rng, n_mc):
            err = self._cfm_err(a, kv_g, pmask_g, grad_params, key)
            cfm = cfm + (err * mask).sum(axis=(1, 2)) / mask.sum()
        cfm = cfm / n_mc
        w = jax.lax.stop_gradient(compute_awr_weights(adv, c['awr_beta'], c['awr_weight_max']))
        loss = jnp.mean(w * cfm)
        return loss, {
            'awr_loss': loss, 'cfm_loss': cfm.mean(),
            'awr_w_mean': w.mean(), 'awr_w_max': w.max(), 'awr_w_min': w.min(),
            'awr_w_clip_frac': jnp.mean(w >= c['awr_weight_max']),
            # effective sample size of the weights: 1 when uniform, 1/N when one sample carries all the weight
            'awr_ess': jnp.square(w.sum()) / (w.shape[0] * jnp.sum(jnp.square(w)) + 1e-12),
        }

    def _awr_terms(self, bc_batch, grad_params, rng):
        """AWR actor terms on replay-buffer windows: (awr_loss, 0, 0, info) in `_actor_terms`' layout.

        awr_loss = mean_b[w_b * CFM_b], w = min(exp(A / awr_beta), awr_weight_max) (clip after the exp, as the
        paper), A = R^λ(s) - V(s) at the window start with the current (just fitted) V; A and w carry no gradient.
        `awr_normalize_adv`: A <- (A - m) / (s + 1e-5) with the buffer-wide mean / std the runner computed with the
        same V (`awr_adv_mean` / `awr_adv_std` in the batch), as the reference's `_calc_adv`.
        CFM_b is `_bc_loss_per_sample`, the per-window flow-matching loss standing in for -log π(a|s).
        """
        c = self.config
        cfm = self._bc_loss_per_sample(bc_batch, grad_params, rng)  # (B,)
        v = jax.lax.stop_gradient(self._state_value(bc_batch['observations']))
        adv = jax.lax.stop_gradient(bc_batch['lambda_returns'] - v)
        adv_raw = adv
        if c.get('awr_normalize_adv', False):
            adv = jax.lax.stop_gradient((adv - bc_batch['awr_adv_mean']) / (bc_batch['awr_adv_std'] + 1e-5))
        w = jax.lax.stop_gradient(compute_awr_weights(adv, c['awr_beta'], c['awr_weight_max']))
        awr_loss = jnp.mean(w * cfm)
        zero = jnp.zeros((), jnp.float32)
        per_task = _per_task({'awr_adv': adv, 'awr_w': w, 'cfm_loss': cfm},
                             bc_batch.get('task_ids'), self.consts['tokens'].shape[0])
        info = {
            **per_task,
            'awr_loss': awr_loss, 'cfm_loss': cfm.mean(),
            'awr_adv_mean': adv.mean(), 'awr_adv_std': adv.std(), 'awr_adv_min': adv.min(), 'awr_adv_max': adv.max(),
            'awr_w_mean': w.mean(), 'awr_w_max': w.max(),
            'awr_w_clip_frac': jnp.mean(w >= c['awr_weight_max']),
            # effective sample size of the weights: 1 when uniform, 1/B when one window carries all the weight
            'awr_ess': jnp.square(w.sum()) / (w.shape[0] * jnp.sum(jnp.square(w)) + 1e-12),
            'awr_lambda_return_mean': bc_batch['lambda_returns'].mean(), 'awr_v_mean': v.mean(),
            'awr_adv_raw_mean': adv_raw.mean(), 'awr_adv_raw_std': adv_raw.std(),
        }
        return awr_loss, zero, zero, info

    def _sde_next_q(self, batch, sample_rng, agg_rng):
        """Default TD bootstrap: SDE next action(s) from the target actor at s' -> (next_q (B,), info)."""
        c = self.config
        _, kv, pmask = self._prefix(batch['next_images'], batch.get('task_ids'), kv_only=True)
        target_critic = self.critic_network.select('target_critic')
        vr_info = {}
        if c['q_variance_reduction']:
            # Mirrors OGPOAgent's q_variance_reduction path with one batched SDE call:
            # reduce over G samples per ensemble member, then aggregate over the ensemble.
            G, B = c['q_vr_num_samples'], pmask.shape[0]
            kv_g, pmask_g = _tile(kv, G, axis=1), _tile(pmask, G, axis=0)  # index g * B + b
            next_actions, _, _, _ = self._sample_sde(self.target_actor_params, kv_g, pmask_g, sample_rng)
            next_actions = self._critic_actions(next_actions).reshape(G, B, -1)
            next_qs_g = jax.vmap(
                lambda a: target_critic(batch['next_observations'], a, is_encoded=True))(next_actions)  # (G, num_qs, B)
            next_qs = reduce_q_over_samples(next_qs_g, method=c['q_vr_reduction'])
            q_std = next_qs_g.std(axis=0)
            vr_info = {'q_vr_sample_std': q_std.mean(), 'q_vr_sample_std_max': q_std.max(),
                       'next_actions_std': next_actions.std(axis=0).mean()}  # across-sample, as the reference
        else:
            next_actions, _, _, _ = self._sample_sde(self.target_actor_params, kv, pmask, sample_rng)
            next_actions = self._critic_actions(next_actions)
            next_qs = target_critic(batch['next_observations'], next_actions, is_encoded=True)
        next_q = aggregate_q_values(next_qs, method=c['q_agg'], rng=agg_rng, num_qs=c['num_qs'])

        return next_q, {'next_actions_std': next_actions.std(), **vr_info}

    def critic_loss(self, batch, grad_params, rng):
        c = self.config
        sample_rng, agg_rng = jax.random.split(rng)
        batch_actions = self._normalize_env_actions(batch['actions'], batch['observations']).reshape(
            batch['actions'].shape[0], -1)

        if self.value_network is not None:
            # SARSA-V: bootstrap on the target V ensemble at s' (cached features; no policy sample, no prefix pass).
            next_vs = self.value_network.select('target_value')(batch['next_observations'], is_encoded=True)
            next_q = aggregate_q_values(next_vs, method=c['v_agg'], rng=agg_rng, num_qs=c['v_ensemble_size'])
            next_info = {'next_v_head_std': next_vs.std(axis=0).mean()}
        else:
            next_q, next_info = self._sde_next_q(batch, sample_rng, agg_rng)

        target_q = compute_td_target(batch['rewards'], batch['masks'], next_q, c['discount'], c['horizon_length'])
        q = self.critic_network.select('critic')(batch['observations'], batch_actions, params=grad_params, is_encoded=True)
        td_loss, _ = compute_td_loss(q, target_q, valid_mask=batch['valid'][..., -1], loss_type='mse')
        K = self.consts['tokens'].shape[0]
        per_task = _per_task({
            'q_mean': q.mean(axis=0), 'td_err_sq': jnp.square(q.mean(axis=0) - target_q),
            'target_q': target_q,
        }, batch.get('task_ids'), K)
        return td_loss, {
            **per_task,
            'td_loss': td_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
            'next_q_mean': next_q.mean(), **next_info,
            'target_q_mean': target_q.mean(), 'q_head_std': q.std(axis=0).mean(),
            **self._mc_stats(jax.lax.stop_gradient(q).mean(axis=0), batch),
        }

    def value_loss(self, batch, grad_params, rng):
        """SARSA-V V step: V(s) <- v_agg over heads of the target critic at the buffer action (vla-post-training's
        `train_value_step` with buffer actions). Masked MSE over V heads, as the Q TD loss.

        `v_training_mode: lambda_return` (AWR): every V head regresses onto the buffer's TD(λ) return at the window
        start (`batch['lambda_returns']`, fixed for the round), with the same masked MSE; no critic is read."""
        c = self.config
        if c.get('v_training_mode') == 'lambda_return':
            target_v = jax.lax.stop_gradient(batch['lambda_returns'])
        else:
            batch_actions = self._normalize_env_actions(batch['actions'], batch['observations']).reshape(
                batch['actions'].shape[0], -1)
            q_tgt = self.critic_network.select('target_critic')(batch['observations'], batch_actions, is_encoded=True)
            target_v = jax.lax.stop_gradient(aggregate_q_values(q_tgt, method=c['v_agg'], rng=rng, num_qs=c['num_qs']))
        v =self.value_network.select('value')(batch['observations'], params=grad_params, is_encoded=True)
        v_loss, _ = compute_td_loss(v, target_v, valid_mask=batch['valid'][..., -1], loss_type='mse')
        per_task = _per_task({'v_mean': v.mean(axis=0), 'v_target': target_v},
                             batch.get('task_ids'), self.consts['tokens'].shape[0])
        return v_loss, {
            **per_task,
            'v_loss': v_loss, 'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),
            'v_target_mean': target_v.mean(), 'v_head_std': v.std(axis=0).mean(),
        }

    @staticmethod
    def _mc_stats(q, batch):
        """Critic accuracy against the observed discounted return-to-go.

        mc_returns ignore the truncation bootstrap, so on truncated failures they are
        above the TD fixed point; compare the success / failure splits, not just the mean.
        NaN when a batch has no transition of that kind (the runner nan-averages).
        """
        if 'mc_returns' not in batch:
            return {}
        mc = batch['mc_returns']
        qc, mcc = q - q.mean(), mc - mc.mean()
        corr = (qc * mcc).mean() / (jnp.sqrt((qc ** 2).mean() * (mcc ** 2).mean()) + 1e-8)
        succ = batch['is_success'][:, 0] > 0.5

        def masked_mean(x, m):
            n = m.sum()
            return jnp.where(n > 0, (x * m).sum() / jnp.maximum(n, 1), jnp.nan)

        return {
            'q_mc_corr': corr, 'mc_mean': mc.mean(), 'q_minus_mc': (q - mc).mean(),
            'q_success': masked_mean(q, succ), 'mc_success': masked_mean(mc, succ),
            'q_fail': masked_mean(q, ~succ), 'mc_fail': masked_mean(mc, ~succ),
            'success_frac': succ.mean(),
        }

    # ------------------------------------------------------------------
    # Updates (mirror OGPOAgent._update with use_success_buffer_q)
    # ------------------------------------------------------------------

    @property
    def _accum_steps(self):
        return self.config.get('grad_accum_steps', 1)

    def _accum_grad(self, loss_fn, params, batches, rng):
        """Mean gradient of `loss_fn(params, micro_batches, key) -> (loss, info)` over K micro-batches.

        Micro-batch k takes rows k, k+K, ... so each keeps the data-parallel sharding. The K
        passes run sequentially in a scan, so peak memory is one micro-batch's. Equal to the
        full-batch gradient for per-sample mean losses (PPO, BC, TD), up to RNG draws.
        info: NaN-aware mean over micro-batches (MC splits are NaN when a micro-batch has no
        such transition), min / max for `*min` / `*max` keys.
        """
        K = self._accum_steps
        micro = jax.tree.map(lambda x: x.reshape(x.shape[0] // K, K, *x.shape[1:]).swapaxes(0, 1), batches)

        def body(g_sum, xs):
            g, info = jax.grad(loss_fn, has_aux=True)(params, *xs)
            return jax.tree.map(jnp.add, g_sum, g), info

        g_sum, infos = jax.lax.scan(body, jax.tree.map(jnp.zeros_like, params), (micro, jax.random.split(rng, K)))
        info = {k: jnp.nanmin(v, 0) if k.endswith('min') else jnp.nanmax(v, 0) if k.endswith('max')
                else jnp.nanmean(v, 0) for k, v in infos.items()}
        return jax.tree.map(lambda g: g / K, g_sum), info

    def _critic_step(self, critic_network, batch, rng):
        agent = self.replace(critic_network=critic_network)
        if self._accum_steps == 1:
            new_critic, info = critic_network.apply_loss_fn(lambda p: agent.critic_loss(batch, p, rng))
        else:
            grads, info = self._accum_grad(lambda p, b, k: agent.critic_loss(b, p, k), critic_network.params, batch, rng)
            leaves = jax.tree.leaves(grads)  # same grad/* stats as TrainState.apply_loss_fn
            info.update({'grad/max': jnp.max(jnp.stack([x.max() for x in leaves])),
                         'grad/min': jnp.min(jnp.stack([x.min() for x in leaves])),
                         'grad/norm': jnp.sum(jnp.stack([jnp.linalg.norm(x.reshape(-1)) for x in leaves]))})
            new_critic = critic_network.apply_gradients(grads=grads)
        tau = self.config['tau']
        new_critic.params['modules_target_critic'] = jax.tree.map(
            lambda p, tp: p * tau + tp * (1 - tau),
            new_critic.params['modules_critic'], new_critic.params['modules_target_critic'])
        return new_critic, info

    def _value_step(self, value_network, batch, rng):
        """SARSA-V V step (mirrors `_critic_step`): reads `self.critic_network`'s target critic."""
        agent = self.replace(value_network=value_network)
        if self._accum_steps == 1:
            new_value, info = value_network.apply_loss_fn(lambda p: agent.value_loss(batch, p, rng))
        else:
            grads, info = self._accum_grad(lambda p, b, k: agent.value_loss(b, p, k), value_network.params, batch, rng)
            leaves = jax.tree.leaves(grads)  # same grad/* stats as TrainState.apply_loss_fn
            info.update({'grad/max': jnp.max(jnp.stack([x.max() for x in leaves])),
                         'grad/min': jnp.min(jnp.stack([x.min() for x in leaves])),
                         'grad/norm': jnp.sum(jnp.stack([jnp.linalg.norm(x.reshape(-1)) for x in leaves]))})
            new_value = value_network.apply_gradients(grads=grads)
        tau = self.config['tau']
        new_value.params['modules_target_value'] = jax.tree.map(
            lambda p, tp: p * tau + tp * (1 - tau),
            new_value.params['modules_value'], new_value.params['modules_target_value'])
        return new_value, {f'v_{k}' if k.startswith('grad/') else k: v for k, v in info.items()}

    def _sarsa_step(self, nets, batch, rng):
        """SARSA-V on one batch: Q step (bootstrap on the current target V), then V step (target critic after the
        Q step's Polyak update), as vla-post-training's `_update_critics`. nets = (critic, value)."""
        critic_network, value_network = nets
        q_rng, v_rng = jax.random.split(rng)
        new_critic, q_info = self.replace(value_network=value_network)._critic_step(critic_network, batch, q_rng)
        new_value, v_info = self.replace(critic_network=new_critic)._value_step(value_network, batch, v_rng)
        return (new_critic, new_value), {**q_info, **v_info}

    def _critic_updates(self, batch, batch_success, success_flag, critic_rng, sb_rng):
        """RB TD step, then the success-buffer TD step if enabled and available. -> (critic, value, info).

        SARSA-V (`value_network` set): each step is a Q step then a V step (`_sarsa_step`). Otherwise the value
        network is None and passes through.
        """
        if self.value_network is not None:
            nets, critic_info = self._sarsa_step((self.critic_network, self.value_network), batch, critic_rng)
            if self.config.get('use_success_buffer_q', False):
                nets = jax.lax.cond(
                    success_flag,
                    lambda n: self._sarsa_step(n, batch_success, sb_rng)[0],
                    lambda n: n,
                    nets)
            return nets[0], nets[1], {f'critic/{k}': v for k, v in critic_info.items()}
        new_critic, critic_info = self._critic_step(self.critic_network, batch, critic_rng)
        if self.config.get('use_success_buffer_q', False):
            new_critic = jax.lax.cond(
                success_flag,
                lambda cn: self._critic_step(cn, batch_success, sb_rng)[0],
                lambda cn: cn,
                new_critic)
        return new_critic, self.value_network, {f'critic/{k}': v for k, v in critic_info.items()}

    def _actor_step(self, batch, bc_batch, ppo_batch, actor_rng):
        """One actor optimiser step + target-actor Polyak. -> (actor_params, target_actor_params, opt_state, info).

        Shared by `update` (PPO + BC) and `actor_update` (AWR); `actor_loss` dispatches on the objective."""
        if self._accum_steps == 1:
            (_, actor_info), grads = jax.value_and_grad(
                lambda p: self.actor_loss(batch, bc_batch, p, actor_rng, ppo_batch), has_aux=True)(self.actor_params)
        else:
            grads, actor_info = self._accum_grad(
                lambda p, b, k: self.actor_loss(b[0], b[1], p, k, b[2]), self.actor_params,
                (batch, bc_batch, ppo_batch), actor_rng)
        updates, new_opt_state = self.actor_tx.update(grads, self.actor_opt_state, self.actor_params)
        new_actor_params = optax.apply_updates(self.actor_params, updates)
        actor_tau = self.config['actor_tau']
        new_target = jax.tree.map(lambda p, tp: p * actor_tau + tp * (1 - actor_tau),
                                  new_actor_params, self.target_actor_params)
        if self.zero_actor_state and self.mesh is not None:
            # ZeRO-1: keep the input layout so donation reuses the buffers and nothing recompiles.
            rep = NamedSharding(self.mesh, P())
            new_actor_params = jax.lax.with_sharding_constraint(new_actor_params, rep)
            new_target = jax.lax.with_sharding_constraint(new_target, rep)
            new_opt_state = jax.lax.with_sharding_constraint(new_opt_state, zero_shardings(new_opt_state, self.mesh))
        actor_info['grad_norm'] = optax.global_norm(grads)
        return new_actor_params, new_target, new_opt_state, actor_info

    def _update_impl(self, batch, batch_success, bc_batch, success_flag, ppo_batch=None):
        new_rng, actor_rng, critic_rng, sb_rng = jax.random.split(self.rng, 4)
        new_actor_params, new_target, new_opt_state, actor_info = self._actor_step(batch, bc_batch, ppo_batch, actor_rng)

        # Critic sees the pre-update target actor, as in OGPOAgent._update.
        new_critic, new_value, critic_info = self._critic_updates(batch, batch_success, success_flag, critic_rng, sb_rng)

        info = {**{f'actor/{k}': v for k, v in actor_info.items()}, **critic_info}
        return self.replace(
            rng=new_rng, actor_params=new_actor_params, target_actor_params=new_target,
            actor_opt_state=new_opt_state, critic_network=new_critic, value_network=new_value), info

    def _critic_update_impl(self, batch, batch_success, success_flag):
        """Critic-only step for runs with `actor_update_interval > 1`.

        Same RNG split arity as `update`; the actor key is drawn and unused so both
        steps advance `self.rng` identically.
        """
        new_rng, _, critic_rng, sb_rng = jax.random.split(self.rng, 4)
        new_critic, new_value, critic_info = self._critic_updates(batch, batch_success, success_flag, critic_rng, sb_rng)
        return self.replace(rng=new_rng, critic_network=new_critic, value_network=new_value), critic_info

    def _actor_update_impl(self, bc_batch):
        """AWR policy step on a replay-buffer window batch (needs `lambda_returns`); critic and V untouched."""
        new_rng, actor_rng = jax.random.split(self.rng)
        new_actor_params, new_target, new_opt_state, actor_info = self._actor_step(None, bc_batch, None, actor_rng)
        return self.replace(rng=new_rng, actor_params=new_actor_params, target_actor_params=new_target,
                            actor_opt_state=new_opt_state), {f'actor/{k}': v for k, v in actor_info.items()}

    def _value_update_impl(self, batch):
        """AWR V step (`v_training_mode: lambda_return`): V(s) <- R^λ on a buffer batch, then target-V Polyak."""
        new_rng, v_rng = jax.random.split(self.rng)
        new_value, info = self._value_step(self.value_network, batch, v_rng)
        return self.replace(rng=new_rng, value_network=new_value), {f'value/{k}': v for k, v in info.items()}

    # `donate_agent` (config): the old agent (frozen + actor + target + opt state, 14 GiB/device) is dead after the
    # call, so XLA reuses its buffers instead of holding two copies (needed on 48 GB GPUs). Callers must rebind
    # (`agent, info = agent.update(...)`) and must not touch the old agent afterwards.
    _update = jax.jit(_update_impl)
    _update_donate = jax.jit(_update_impl, donate_argnums=0)
    _critic_update = jax.jit(_critic_update_impl)
    _critic_update_donate = jax.jit(_critic_update_impl, donate_argnums=0)
    _actor_update = jax.jit(_actor_update_impl)
    _actor_update_donate = jax.jit(_actor_update_impl, donate_argnums=0)
    _value_update = jax.jit(_value_update_impl)
    _value_update_donate = jax.jit(_value_update_impl, donate_argnums=0)

    def update(self, batch, batch_success, bc_batch, success_flag, ppo_batch=None):
        fn = self._update_donate if self.config.get('donate_agent', False) else self._update
        return fn(batch, batch_success, bc_batch, success_flag, ppo_batch)

    def critic_update(self, batch, batch_success, success_flag):
        fn = self._critic_update_donate if self.config.get('donate_agent', False) else self._critic_update
        return fn(batch, batch_success, success_flag)

    def actor_update(self, bc_batch):
        fn = self._actor_update_donate if self.config.get('donate_agent', False) else self._actor_update
        return fn(bc_batch)

    def value_update(self, batch):
        fn = self._value_update_donate if self.config.get('donate_agent', False) else self._value_update
        return fn(batch)

    def grad_diagnostics(self, batch, bc_batch):
        """Log-time only: PG vs BC gradient split and actor drift. No state change.

        grad_* are pre-clip norms of each term as it enters the total loss (BC scaled by
        bc_coeff). Each gradient is its own jit call so peak memory stays at one actor
        gradient; a fused one-forward/two-VJP version OOMed at G=32 with a 256 BC batch.
        """
        g_pg = self._term_grad(batch, bc_batch, 'pg')
        g_bc = self._term_grad(batch, bc_batch, 'bc')
        return self._diag_stats(g_pg, g_bc)

    @partial(jax.jit, static_argnames=('term',))
    def _term_grad(self, batch, bc_batch, term):
        rng = jax.random.fold_in(self.rng, 7)

        def loss(p, b, key):
            pg_loss, _, bc_loss, _ = self._actor_terms(*b, p, key)
            return (pg_loss if term == 'pg' else self.config['bc_coeff'] * bc_loss), {}

        if self._accum_steps == 1:
            return jax.grad(lambda p: loss(p, (batch, bc_batch), rng)[0])(self.actor_params)
        return self._accum_grad(loss, self.actor_params, (batch, bc_batch), rng)[0]

    @jax.jit
    def _diag_stats(self, g_pg, g_bc):
        n_pg, n_bc = optax.global_norm(g_pg), optax.global_norm(g_bc)
        dot = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(g_pg), jax.tree.leaves(g_bc)))

        def dist(a, b):
            return optax.global_norm(jax.tree.map(lambda x, y: x - y, a, b))

        init_norm = optax.global_norm(self.init_actor_params)
        return {
            'grad_pg_norm': n_pg, 'grad_bc_norm': n_bc, 'grad_cos_pg_bc': dot / (n_pg * n_bc + 1e-12),
            'grad_pg_share': n_pg / (n_pg + n_bc + 1e-12),
            'drift_online_target_rel': dist(self.actor_params, self.target_actor_params) / init_norm,
            'drift_online_init_rel': dist(self.actor_params, self.init_actor_params) / init_norm,
            'drift_target_init_rel': dist(self.target_actor_params, self.init_actor_params) / init_norm,
        }

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def _pool_prefix(self, prefix_out, prefix_mask):
        """(B, S, 2048) prefix hidden states -> (B, k * 2048) critic features.

        Prefix layout (`Pi05.embed_prefix`): len(IMAGE_KEYS) cameras x 256 tokens, then the
        language tokens. `prefix_mask` excludes the dummy third camera and prompt padding.
          all        : unmasked mean over every position (original behaviour).
          segmented  : masked means over image tokens and language tokens, each L2-normalised.
          per_camera : masked means over base, wrist and language tokens, each L2-normalised.
        """
        mode = self.config['pi05_critic_pooling']
        if mode == 'all':
            return prefix_out.mean(axis=1).astype(jnp.float32)
        x = prefix_out.astype(jnp.float32)
        m = prefix_mask.astype(jnp.float32)
        n_lang = self.consts['tokens'].shape[-1]
        n_img = x.shape[1] - n_lang
        n_cam = n_img // len(IMAGE_KEYS)
        if mode == 'segmented':
            spans = [(0, n_img), (n_img, x.shape[1])]
        else:  # per_camera: base (slot 0) and left wrist (slot 1); slot 2 is always masked here
            spans = [(0, n_cam), (n_cam, 2 * n_cam), (n_img, x.shape[1])]
        feats = []
        for lo, hi in spans:
            w = m[:, lo:hi, None]
            seg = (x[:, lo:hi] * w).sum(axis=1) / jnp.maximum(w.sum(axis=1), 1.0)
            feats.append(seg / (jnp.linalg.norm(seg, axis=-1, keepdims=True) + 1e-6))
        return jnp.concatenate(feats, axis=-1)

    @jax.jit
    def encode(self, images, states, task_ids=None):
        """(B, 2, 224, 224, 3) uint8, (B, 8) raw state[, (B,) task ids] -> critic obs (B, k * 2048 + 8)."""
        prefix_out, _, prefix_mask = self._prefix(images, task_ids)
        q01, q99 = self.consts['state_q01'], self.consts['state_q99']
        state = (states - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return jnp.concatenate([self._pool_prefix(prefix_out, prefix_mask), state], axis=-1)

    @jax.jit
    def sample_actions_BON(self, images, critic_obs, rng, task_ids=None):
        """Best-of-N over target-actor samples (`pi05_bon_sampler`), ranked by the target critic with
        `pi05_bon_agg`. Mirrors `OGPOAgent.sample_actions_BON`.

        images (B, 2, 224, 224, 3), critic_obs (B, k * 2048 + 8)[, task_ids (B,)] -> env actions (B, H, A).
        """
        B, N = images.shape[0], self.config['best_of_n']
        actions, q_values, sub_rng = self._bon_candidates(images, critic_obs, rng, task_ids)
        best = select_best_of_n(q_values, self.config['pi05_bon_agg'], sub_rng)
        best_actions = jax.vmap(lambda a, i: a[i])(actions.reshape(B, N, -1), best)
        return self._to_env_actions(best_actions, critic_obs)

    def _bon_candidates(self, images, critic_obs, rng, task_ids):
        """N target-actor candidates per state and their target-critic Q -> (actions (B * N, D), Q (num_qs, B, N),
        the rng for the head subsample)."""
        B, N = images.shape[0], self.config['best_of_n']
        action_rng, sub_rng = jax.random.split(rng)
        _, kv, pmask = self._prefix(images, task_ids, kv_only=True)
        # candidate index b * N + n, matching jnp.repeat in the reference
        kv_n = jax.tree.map(lambda x: jnp.repeat(x, N, axis=1), kv)
        pmask_n = jnp.repeat(pmask, N, axis=0)
        if self.config['pi05_bon_sampler'] == 'ode':
            actions = self._sample_ode(self.target_actor_params, kv_n, pmask_n, action_rng, clip=False)
        else:
            actions, _, _, _ = self._sample_sde(self.target_actor_params, kv_n, pmask_n, action_rng)
        q_values = self.critic_network.select('target_critic')(
            jnp.repeat(critic_obs, N, axis=0), self._critic_actions(actions), is_encoded=True,
        ).reshape(self.config['num_qs'], B, N)
        return actions, q_values, sub_rng

    @partial(jax.jit, static_argnames=('select',))
    def sample_actions_BON_select(self, images, critic_obs, rng, task_ids=None, select='best'):
        """Critic diagnostic: `sample_actions_BON` with the pick swapped -> (env actions (B, H, A), candidate Q
        (num_qs, B, N), picked index (B,)). select: best (`sample_actions_BON`'s pick, argmax of the `pi05_bon_agg`
        scores) | worst (argmin of the same scores) | random (uniform). Same rng splits as `sample_actions_BON`, so
        for a given key every rule sees the same N candidates."""
        B, N = images.shape[0], self.config['best_of_n']
        actions, q_values, sub_rng = self._bon_candidates(images, critic_obs, rng, task_ids)
        scores = bon_scores(q_values, self.config['pi05_bon_agg'], sub_rng)
        if select == 'best':
            idx = jnp.argmax(scores, axis=1)
        elif select == 'worst':
            idx = jnp.argmin(scores, axis=1)
        elif select == 'random':
            idx = jax.random.randint(jax.random.fold_in(sub_rng, 1), (B,), 0, N)
        else:
            raise ValueError(f"select={select!r}; expected best | worst | random")
        picked = jax.vmap(lambda a, i: a[i])(actions.reshape(B, N, -1), idx)
        return self._to_env_actions(picked, critic_obs), q_values, idx

    def _sample_ode(self, params, kv_cache, prefix_mask, rng, clip):
        """Euler ODE from x0 ~ N(0, I) (normalised model space).

        clip=True: final clip to [act_min, act_max] (eval). clip=False: unclipped, as openpi /
        vla-post-training sample (±inf bounds make `sample_flow_actions_ode`'s final clip an identity).
        """
        c = self.config
        return sample_flow_actions_ode(
            actor_fn=self._actor_fn(params, kv_cache, prefix_mask), observations=jnp.zeros((prefix_mask.shape[0], 1)),
            rng=rng, flow_steps=c['flow_steps'], act_dim=self._flat_dim,
            act_min=c['act_min'] if clip else -jnp.inf, act_max=c['act_max'] if clip else jnp.inf,
            clip_intermediate=False, clip_value=c['denoised_clip_value'], is_encoded=True)

    @jax.jit
    def sample_actions_ode(self, images, rng, task_ids=None, critic_obs=None):
        """Deterministic Euler ODE with the online actor (eval). -> env actions (B, H, A).
        critic_obs (B, F): the chunk-start features, needed for droid's delta actions."""
        _, kv, pmask = self._prefix(images, task_ids, kv_only=True)
        return self._to_env_actions(self._sample_ode(self.actor_params, kv, pmask, rng, clip=True), critic_obs)

    @jax.jit
    def sample_actions_plain(self, images, rng, task_ids=None, critic_obs=None):
        """One unclipped ODE sample from the target actor: vla-post-training's plain (pre-BoN) policy,
        used for round 0 with `training.first_round_policy=ode`. -> env actions (B, H, A); critic_obs as sample_actions_ode."""
        _, kv, pmask = self._prefix(images, task_ids, kv_only=True)
        return self._to_env_actions(self._sample_ode(self.target_actor_params, kv, pmask, rng, clip=False), critic_obs)

    @jax.jit
    def sample_actions_explore(self, images, rng, task_ids=None, critic_obs=None):
        """One unclipped ODE sample from the ONLINE actor: AWR's collection policy π_k (Alg. 1 line 4), every round.
        Same as `sample_actions_plain` but independent of `actor_tau`. -> env actions (B, H, A)."""
        _, kv, pmask = self._prefix(images, task_ids, kv_only=True)
        return self._to_env_actions(self._sample_ode(self.actor_params, kv, pmask, rng, clip=False), critic_obs)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, seed, prompt, config):
        """`prompt`: one task string, or a list of K strings (multi-task; row k of the token table = task k)."""
        c = config
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        assert prompts, "at least one prompt"
        if c['pi05_critic_pooling'] not in POOLING_SEGMENTS:
            raise ValueError(f"pi05_critic_pooling={c['pi05_critic_pooling']!r}; "
                             f"expected one of {list(POOLING_SEGMENTS)}")
        if c['q_variance_reduction'] and c['q_vr_num_samples'] < 2:
            raise ValueError(f"q_vr_num_samples={c['q_vr_num_samples']}; need >= 2 with q_variance_reduction")
        if c['q_vr_reduction'] not in ('mean', 'median', 'trimmed_mean'):
            raise ValueError(f"q_vr_reduction={c['q_vr_reduction']!r}; expected mean | median | trimmed_mean")
        if not isinstance(c.get('pi05_drop_empty_camera', False), bool):
            raise ValueError(f"pi05_drop_empty_camera={c['pi05_drop_empty_camera']!r}; expected true | false")
        if c['pi05_bon_sampler'] not in ('sde', 'ode'):
            raise ValueError(f"pi05_bon_sampler={c['pi05_bon_sampler']!r}; expected sde | ode")
        if c['pi05_bon_agg'] not in ('min', 'subsample', 'mean'):
            raise ValueError(f"pi05_bon_agg={c['pi05_bon_agg']!r}; expected min | subsample | mean")
        if not c['subsample_bon'] and c['pi05_bon_agg'] == 'subsample':
            raise ValueError("subsample_bon=false is not read by the pi05 agent; its BoN rule is pi05_bon_agg "
                             "(set pi05_bon_agg=min for the min-over-all-heads rule)")
        objective = c.get('pi05_actor_objective', 'ppo')  # .get: pre-AWR configs (and the CPU test stubs) lack it
        if objective not in ('ppo', 'awr', 'group_awr'):
            raise ValueError(f"pi05_actor_objective={objective!r}; expected ppo | awr | group_awr")
        if c['use_value_fn'] and c['v_training_mode'] == 'lambda_return' and objective != 'awr':
            raise ValueError("v_training_mode=lambda_return is AWR's V target; it needs pi05_actor_objective=awr")
        if objective == 'awr':
            if not c['use_value_fn'] or c['v_training_mode'] != 'lambda_return':
                raise ValueError("pi05_actor_objective=awr needs use_value_fn=true and v_training_mode=lambda_return "
                                 f"(got use_value_fn={c['use_value_fn']}, v_training_mode={c['v_training_mode']!r})")
            if c['awr_beta'] <= 0 or c['awr_weight_max'] <= 0:
                raise ValueError(f"awr_beta={c['awr_beta']}, awr_weight_max={c['awr_weight_max']}; both must be > 0")
            if c['awr_mode'] != 'symmetric':
                raise ValueError(f"awr_mode={c['awr_mode']!r}; the pi05 AWR is the paper's symmetric weight only")
            if not 0.0 <= c['awr_lambda'] <= 1.0:
                raise ValueError(f"awr_lambda={c['awr_lambda']}; expected a value in [0, 1]")
            if c['v_agg'] not in ('mean', 'min'):
                raise ValueError(f"v_agg={c['v_agg']!r}; AWR's V baseline and λ-return bootstrap need mean | min")
            if c['use_bc_regularization']:
                raise ValueError("pi05_actor_objective=awr has no BC anchor; set use_bc_regularization=false")
        if objective == 'group_awr':
            if c['awr_beta'] <= 0 or c['awr_weight_max'] <= 0:
                raise ValueError(f"awr_beta={c['awr_beta']}, awr_weight_max={c['awr_weight_max']}; both must be > 0")
            if c['awr_mode'] != 'symmetric':
                raise ValueError(f"awr_mode={c['awr_mode']!r}; group_awr uses the paper's symmetric weight only")
            if c['awr_n_mc'] < 1:
                raise ValueError(f"awr_n_mc={c['awr_n_mc']}; group_awr needs >= 1 (t, x0) draw per sample")
            if c['adv_strategy'] not in ('vanilla', 'conservative', 'penultimate'):
                # adv_strategy=awr makes compute_ogpo_advantages return weights, which would be exponentiated twice
                raise ValueError(f"adv_strategy={c['adv_strategy']!r} under group_awr; "
                                 "expected vanilla | conservative | penultimate")
            if c['entropy_coeff'] != 0:
                raise ValueError(f"entropy_coeff={c['entropy_coeff']}; group_awr computes no log-prob or entropy, set 0")
            if c['awr_normalize_adv']:
                raise ValueError("awr_normalize_adv reads the awr runner's buffer-wide statistics; group_awr has none "
                                 "(normalize_group=true standardises the group advantages instead)")
        if c['use_value_fn']:
            if c['v_training_mode'] not in ('sarsa', 'lambda_return'):
                raise ValueError(f"use_value_fn on the pi05 agent supports v_training_mode=sarsa | lambda_return; "
                                 f"got {c['v_training_mode']!r}")
            if c['q_variance_reduction']:
                raise ValueError("use_value_fn (SARSA-V) and q_variance_reduction both redefine the TD bootstrap; "
                                 "turn one off")
            if c['v_pair_with_q']:
                raise ValueError("v_pair_with_q is not supported by the pi05 SARSA-V critic (one shared target per batch)")
            if c['v_agg'] not in ('mean', 'min', 'subsample'):
                raise ValueError(f"v_agg={c['v_agg']!r}; expected mean | min | subsample")
            if c['v_ensemble_size'] < 2:
                # Value with one head returns (B,), not (1, B): the head reduction would average over the batch.
                raise ValueError(f"v_ensemble_size={c['v_ensemble_size']}; the SARSA-V ensemble needs >= 2 heads")
        if c['pi05_critic_arch'] not in ('mlp', 'bronet', 'bronet_vla'):
            raise ValueError(f"pi05_critic_arch={c['pi05_critic_arch']!r}; expected mlp | bronet | bronet_vla")
        # VLA's critic prefix is the unmasked mean over every prefix token (best_of_n_learner.py:233-234).
        if c['pi05_critic_arch'] == 'bronet_vla' and c['pi05_critic_pooling'] != 'all':
            raise ValueError(f"pi05_critic_arch=bronet_vla needs pi05_critic_pooling=all to match vla-post-training; "
                             f"got {c['pi05_critic_pooling']!r}")
        if c['pi05_robot'] not in T.ROBOTS:
            raise ValueError(f"pi05_robot={c['pi05_robot']!r}; expected one of {list(T.ROBOTS)}")
        robot = T.ROBOTS[c['pi05_robot']]
        ckpt = c['pi05_checkpoint']
        rng = jax.random.PRNGKey(seed)
        rng, critic_rng = jax.random.split(rng)

        model = Pi05(action_horizon=c['pi05_action_horizon'])
        frozen, trainable = split_params(load_pi05_params(f"{ckpt}/params"))
        norm_stats = T.load_norm_stats(f"{ckpt}/assets", asset_id=robot['asset_id'])
        tokenizer = T.PromptTokenizer(c['pi05_tokenizer'])
        tokenized = [tokenizer.tokenize(p) for p in prompts]
        tokens = np.stack([t for t, _ in tokenized])        # (K, L)
        token_mask = np.stack([m for _, m in tokenized])    # (K, L)

        h, d = c['pi05_action_horizon'], T.MODEL_ACTION_DIM
        assert c['horizon_length'] <= h
        dim_mask = np.zeros((h, d), np.float32)
        dim_mask[:c['horizon_length'], :robot['env_action_dim']] = 1.0
        consts = {
            'tokens': jnp.asarray(tokens), 'token_mask': jnp.asarray(token_mask),
            'dim_mask': jnp.asarray(dim_mask.reshape(-1)),
            'action_q01': jnp.asarray(norm_stats['actions']['q01']), 'action_q99': jnp.asarray(norm_stats['actions']['q99']),
            'state_q01': jnp.asarray(norm_stats['state']['q01'][:STATE_DIM]),
            'state_q99': jnp.asarray(norm_stats['state']['q99'][:STATE_DIM]),
        }

        actor_tx = optax.chain(
            optax.clip_by_global_norm(c['pi05_clip_grad_norm']),
            optax.adamw(c['pi05_lr'], b1=0.9, b2=0.95, eps=1e-8, weight_decay=c['pi05_weight_decay']))
        trainable = jax.tree.map(jnp.asarray, trainable)

        feat_dim = POOLING_SEGMENTS[c['pi05_critic_pooling']] * PREFIX_DIM
        # bronet_vla: no two-tier encoder, so each BroNet head reads the raw [prefix | state | action] concat.
        vla = c['pi05_critic_arch'] == 'bronet_vla'
        value_kwargs = dict(
            hidden_dims=c['value_hidden_dims'], layer_norm=c['layer_norm'],
            critic_loss_type='mse', q_min=c['q_min'], q_max=c['q_max'],
            obs_two_tier=not vla, two_tier_img_dim=feat_dim, two_tier_proprio_dim=STATE_DIM,
            two_tier_fused_dim=c.get('two_tier_fused_dim', 0), head_arch='bronet' if vla else c['pi05_critic_arch'],
            bronet_hidden_dim=c['pi05_bronet_hidden_dim'], bronet_depth=c['pi05_bronet_depth'])
        critic_def = Value(num_ensembles=c['num_qs'], **value_kwargs)
        critic_net_def = ModuleDict({'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)})
        ex = dict(observations=jnp.zeros((1, feat_dim + STATE_DIM)),
                  actions=jnp.zeros((1, c['horizon_length'] * robot['env_action_dim'])))
        critic_params = critic_net_def.init(critic_rng, critic=ex, target_critic=ex)['params']
        # Online critic optimiser as in OGPOAgent.reset_optimizers_with_lr (base lr = ppo_lr).
        def _critic_tx():
            return create_optimizer(
                lr_schedule=create_lr_schedule(c['critic_scheduler'], c['ppo_lr'], c['critic_warmup_steps'],
                                               c['critic_decay_steps'], c['critic_end_value']),
                clip_grad_norm=c['clip_grad_norm'], weight_decay=c['critic_weight_decay'])
        critic_state = TrainState.create(critic_net_def, critic_params, _critic_tx())
        # A copy, not an alias: donated buffers must be distinct.
        critic_state.params['modules_target_critic'] = jax.tree.map(jnp.copy, critic_state.params['modules_critic'])

        value_state = None
        if c['use_value_fn']:
            # SARSA-V state-value ensemble: the critic's architecture without the action input, own optimiser.
            rng, value_rng = jax.random.split(rng)
            value_def = Value(num_ensembles=c['v_ensemble_size'], **value_kwargs)
            value_net_def = ModuleDict({'value': value_def, 'target_value': copy.deepcopy(value_def)})
            ex_v = dict(observations=ex['observations'])
            value_params = value_net_def.init(value_rng, value=ex_v, target_value=ex_v)['params']
            value_state = TrainState.create(value_net_def, value_params, _critic_tx())
            value_state.params['modules_target_value'] = jax.tree.map(jnp.copy, value_state.params['modules_value'])

        n_train = sum(x.size for x in jax.tree.leaves(trainable))
        n_frozen = sum(x.size for x in jax.tree.leaves(frozen))
        n_critic = sum(x.size for x in jax.tree.leaves(critic_state.params['modules_critic']))
        print(f"[pi05] trainable {n_train / 1e6:.1f}M, frozen {n_frozen / 1e9:.2f}B, "
              f"{len(prompts)} task prompt(s): {prompts!r}")
        print(f"[pi05] critic {c['pi05_critic_arch']} x {c['num_qs']} heads: {n_critic / 1e6:.1f}M params")
        if c.get('pi05_drop_empty_camera', False):
            print("[pi05] pi05_drop_empty_camera: KV-only prefix passes skip the masked right-wrist slot (encode keeps it)")
        print(f"[pi05] robot {c['pi05_robot']}: {robot['env_action_dim']}-D env actions, "
              f"{robot['delta_dims']} delta dim(s), norm stats {ckpt}/assets/{robot['asset_id']}")
        if value_state is not None:
            n_value = sum(x.size for x in jax.tree.leaves(value_state.params['modules_value']))
            print(f"[pi05] SARSA-V value {c['pi05_critic_arch']} x {c['v_ensemble_size']} heads: {n_value / 1e6:.1f}M params "
                  f"(v_agg={c['v_agg']}, v_training_mode={c['v_training_mode']})")
        if objective == 'awr':
            print(f"[pi05] actor objective: awr (beta={c['awr_beta']}, weight_max={c['awr_weight_max']}, "
                  f"lambda={c['awr_lambda']}); the critic is built but never trained or read")
        if objective == 'group_awr':
            print(f"[pi05] actor objective: group_awr (beta={c['awr_beta']}, weight_max={c['awr_weight_max']}, "
                  f"n_mc={c['awr_n_mc']}, adv_strategy={c['adv_strategy']}); OGPO samples / advantages, AWR loss")
        return cls(
            rng=rng, actor_params=trainable, target_actor_params=jax.tree.map(jnp.copy, trainable),
            actor_opt_state=actor_tx.init(trainable), init_actor_params=jax.tree.map(jnp.copy, trainable),
            frozen_params=frozen, critic_network=critic_state,
            consts=consts, config=c, model=model, actor_tx=actor_tx, value_network=value_state, robot=c['pi05_robot'])
