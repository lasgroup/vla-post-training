from typing import Tuple, Callable, Optional, Dict, Any
import jax
import jax.numpy as jnp
import distrax


def flow_score_from_velocity(v: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return (t * v - x) / (1.0 - t)


def sde_drift_correction(v: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray,
                         sigma_base: jnp.ndarray, use_tapered_noise: bool) -> jnp.ndarray:
    tv_minus_x = t * v - x
    if use_tapered_noise:
        return 0.5 * sigma_base * sigma_base * tv_minus_x
    else:
        return 0.5 * sigma_base * sigma_base * tv_minus_x / (1.0 - t)


def sample_flow_actions_ode(
        actor_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        flow_steps: int,
        act_dim: int,
        act_min: float = -1.0,
        act_max: float = 1.0,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        noises: jnp.ndarray = None,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
    """Sample actions using deterministic ODE flow (Euler integration)."""
    batch_size = observations.shape[0]

    # Initialize with noise
    if noises is None:
        actions = jax.random.normal(rng, (batch_size, act_dim))
    else:
        actions = noises

    dt = 1.0 / flow_steps

    for i in range(flow_steps):
        t = jnp.full((batch_size, 1), i / flow_steps)
        vels = actor_fn(observations, actions, t, is_encoded=is_encoded)
        actions = actions + vels * dt

        if clip_intermediate:
            actions = jnp.clip(actions, -clip_value, clip_value)

    actions = jnp.clip(actions, act_min, act_max)
    return actions


def sample_flow_actions_ode_with_correction(
        actor_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        flow_steps: int,
        act_dim: int,
        constant_noise_std: float,
        act_min: float = -1.0,
        act_max: float = 1.0,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        noises: jnp.ndarray = None,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
    batch_size = observations.shape[0]

    if noises is None:
        actions = jax.random.normal(rng, (batch_size, act_dim))
    else:
        actions = noises

    dt = 1.0 / flow_steps
    correction_coeff = constant_noise_std / (2 * dt) if dt > 0 else 0.0

    for i in range(flow_steps):
        t = jnp.full((batch_size, 1), i / flow_steps)
        vels, z_pred = actor_fn(observations, actions, t, is_encoded=is_encoded, return_denoiser=True)

        # SDE-to-ODE correction
        vels = vels + correction_coeff * z_pred
        actions = actions + vels * dt

        if clip_intermediate:
            actions = jnp.clip(actions, -clip_value, clip_value)

    actions = jnp.clip(actions, act_min, act_max)
    return actions


def sample_flow_actions_sde(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        flow_steps: int,
        act_dim: int,
        use_constant_noise: bool = False,
        use_tapered_noise: bool = False,
        error_correct_sde_to_ode: bool = False,
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        randn_clip_value: float = 3.0,
        act_min: float = -1.0,
        act_max: float = 1.0,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        is_encoded: bool = False,
        params: Any = None,
        ft_flow_steps: int = 0,
        dim_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size = observations.shape[0]

    # Determine which transitions contribute to logprob
    effective_ft = ft_flow_steps if (0 < ft_flow_steps < flow_steps) else flow_steps
    ft_start = flow_steps - effective_ft  # first transition index to include

    rng, key = jax.random.split(rng)
    actions = jax.random.normal(key, (batch_size, act_dim))
    chains = [actions]
    sigmas = []

    # Initial log prob: p(x_0) ~ N(0, I)
    # Include only when computing over the full chain (cancels in ratio either way)
    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    lp_mask = 1.0 if dim_mask is None else dim_mask
    if ft_start == 0:
        logprob = (init_dist.log_prob(actions) * lp_mask).sum(-1)
    else:
        logprob = jnp.zeros(batch_size)

    dt = 1.0 / flow_steps

    for i in range(flow_steps):
        rng, key = jax.random.split(rng)
        t = jnp.full((batch_size, 1), i / flow_steps)
        is_last_step = (i == flow_steps - 1)

        # Get velocity
        if params is not None:
            vels = actor_fn(observations, actions, t, params=params, is_encoded=is_encoded)
        else:
            vels = actor_fn(observations, actions, t, is_encoded=is_encoded)

        apply_correction = error_correct_sde_to_ode and (
            use_tapered_noise or (use_constant_noise and not is_last_step)
        )
        if apply_correction:
            drift = vels + sde_drift_correction(
                vels, actions, t, constant_noise_std, use_tapered_noise
            )
        else:
            drift = vels

        # Compute mean of next action
        mean_next = actions + drift * dt

        if clip_intermediate:
            mean_next = jnp.clip(mean_next, -clip_value, clip_value)

        # Get noise std
        if use_tapered_noise:
            sigma = constant_noise_std * jnp.sqrt(jnp.maximum(1.0 - t, 0.0)) * jnp.ones(actions.shape)
        elif use_constant_noise:
            if is_last_step:
                # Skip noise on the final step: degenerate Dirac mass at mean_next.
                sigma = jnp.zeros(actions.shape)
            else:
                sigma = constant_noise_std * jnp.ones(actions.shape)
        else:
            if params is not None:
                sigma = noise_fn(observations, t, params=params)
            else:
                sigma = noise_fn(observations, t)
            sigma = jnp.maximum(sigma, min_noise_std)

        if dim_mask is not None:
            sigma = sigma * dim_mask

        sigmas.append(sigma)

        if use_constant_noise and is_last_step:
            actions = mean_next
            # No stochastic transition -> no logprob contribution this step.
        else:
            step_dist = distrax.Normal(mean_next, sigma)
            rng, noise_key = jax.random.split(rng)
            actions = step_dist.sample(seed=noise_key)
            actions = jnp.clip(actions,
                               mean_next - randn_clip_value * sigma,
                               mean_next + randn_clip_value * sigma)

            if i >= ft_start:
                if dim_mask is None:
                    logprob = logprob + step_dist.log_prob(actions).sum(-1)
                else:
                    safe_dist = distrax.Normal(mean_next, jnp.where(dim_mask > 0, sigma, 1.0))
                    logprob = logprob + (safe_dist.log_prob(actions) * dim_mask).sum(-1)

        if is_last_step:
            actions = jnp.clip(actions, act_min, act_max)

        chains.append(actions)

    chains = jnp.stack(chains, axis=1)
    sigmas = jnp.stack(sigmas, axis=1)

    return actions, chains, logprob, sigmas


def compute_flow_log_prob(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        chains: jnp.ndarray,
        flow_steps: int,
        use_constant_noise: bool = False,
        use_tapered_noise: bool = False,
        error_correct_sde_to_ode: bool = False,
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        normalize_horizon: bool = False,
        normalize_dim: bool = False,
        is_encoded: bool = False,
        params: Any = None,
        get_entropy: bool = False,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        ft_flow_steps: int = 0,
        dim_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict]:
    batch_size = chains.shape[0]
    act_dim = chains.shape[-1]
    dt = 1.0 / flow_steps
    lp_mask = jnp.ones(act_dim) if dim_mask is None else dim_mask
    norm_dim = act_dim
    if dim_mask is not None:
        assert not use_constant_noise, "dim_mask is only supported with tapered / noise-net sigmas"
        norm_dim = dim_mask.sum()  # normalisation over masked-in dims only

    # Determine which transitions contribute to logprob
    effective_ft = ft_flow_steps if (0 < ft_flow_steps < flow_steps) else flow_steps
    ft_start = flow_steps - effective_ft  # first transition index to include

    logprob = 0.0
    joint_entropy = 0.0
    logprob_steps = 0

    # Initial probability: p(x_0) ~ N(0, I)
    # Include only when computing over the full chain (cancels in ratio either way)
    init_dist = distrax.Normal(jnp.zeros((batch_size, chains.shape[-1])), 1.0)
    if ft_start == 0:
        logprob_init = (init_dist.log_prob(chains[:, 0]) * lp_mask).sum(-1)
        logprob = logprob + logprob_init
        logprob_steps += 1

        if get_entropy:
            entropy_init = (init_dist.entropy() * lp_mask).sum(-1)
            joint_entropy = joint_entropy + entropy_init

    # Pre-compute chain slices
    chains_prev = chains[:, :-1, :]  # [B, flow_steps, act_dim]
    chains_next = chains[:, 1:, :]   # [B, flow_steps, act_dim]

    # Compute velocities, drifts, and sigmas for all steps
    chains_vel = jnp.zeros_like(chains_prev)
    chains_drift = jnp.zeros_like(chains_prev)
    chains_stds = jnp.zeros_like(chains_prev)

    skip_last_under_constant = use_constant_noise  # sigma=0 on final step

    for i in range(flow_steps):
        t = jnp.full((batch_size, 1), i / flow_steps)
        x_t = chains[:, i, :]
        is_last_step = (i == flow_steps - 1)

        if params is not None:
            v = actor_fn(observations, x_t, t, params=params, is_encoded=is_encoded)
        else:
            v = actor_fn(observations, x_t, t, is_encoded=is_encoded)

        # Score correction (mirror sampler exactly)
        apply_correction = error_correct_sde_to_ode and (
            use_tapered_noise or (use_constant_noise and not is_last_step)
        )
        if apply_correction:
            drift = v + sde_drift_correction(
                v, x_t, t, constant_noise_std, use_tapered_noise
            )
        else:
            drift = v

        # Sigma schedule (mirror sampler exactly)
        if use_tapered_noise:
            sigma = constant_noise_std * jnp.sqrt(jnp.maximum(1.0 - t, 0.0)) * jnp.ones((batch_size, act_dim))
        elif use_constant_noise:
            if is_last_step:
                sigma = jnp.zeros((batch_size, act_dim))
            else:
                sigma = constant_noise_std * jnp.ones((batch_size, act_dim))
        else:
            if params is not None:
                sigma = noise_fn(observations, t, params=params)
            else:
                sigma = noise_fn(observations, t)
            sigma = jnp.maximum(sigma, min_noise_std)

        if dim_mask is not None:
            sigma = sigma * dim_mask

        chains_vel = chains_vel.at[:, i, :].set(v)
        chains_drift = chains_drift.at[:, i, :].set(drift)
        chains_stds = chains_stds.at[:, i, :].set(sigma)

    # Compute means for all transitions using drift (= velocity + optional correction)
    chains_mean = chains_prev + chains_drift * dt

    # Clip intermediate means to match SDE sampling behavior
    if clip_intermediate:
        chains_mean = jnp.clip(chains_mean, -clip_value, clip_value)

    if skip_last_under_constant:
        chains_stds_safe = chains_stds.at[:, -1, :].set(1.0)
        chains_dist = distrax.Normal(chains_mean, chains_stds_safe)
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)  # [B, flow_steps]
        mask = jnp.ones(flow_steps).at[-1].set(0.0)
        logprob_trans = logprob_trans * mask[None, :]
        # ft_start..flow_steps spans `effective_ft` indices; the last is masked.
        contributing_steps = effective_ft - 1
    elif dim_mask is not None:
        chains_dist = distrax.Normal(chains_mean, jnp.where(dim_mask > 0, chains_stds, 1.0))
        logprob_trans = (chains_dist.log_prob(chains_next) * dim_mask).sum(-1)
        contributing_steps = effective_ft
    else:
        chains_dist = distrax.Normal(chains_mean, chains_stds)
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)
        contributing_steps = effective_ft

    logprob_trans_ft = logprob_trans[:, ft_start:]
    logprob = logprob + logprob_trans_ft.sum(axis=-1)
    logprob_steps += contributing_steps

    if get_entropy:
        entropy_trans = (chains_dist.entropy() * lp_mask).sum(-1)  # [B, flow_steps]
        if skip_last_under_constant:
            # Mask out the dummy final-step entropy (where we set sigma=1.0).
            entropy_mask = jnp.ones(flow_steps).at[-1].set(0.0)
            entropy_trans = entropy_trans * entropy_mask[None, :]
        entropy_trans_ft = entropy_trans[:, ft_start:]
        joint_entropy = joint_entropy + entropy_trans_ft.sum(axis=-1)
        entropy_rate = joint_entropy / jnp.maximum(logprob_steps, 1) / norm_dim
    else:
        entropy_rate = None

    # Normalization
    if normalize_horizon:
        logprob = logprob / logprob_steps

    if normalize_dim:
        logprob = logprob / norm_dim

    info = {
        'vel_mean': chains_vel.mean(),
        'sigma_mean': chains_stds.mean(),
        'logprob_mean': logprob.mean(),
        'logprob_steps': logprob_steps,
    }

    if get_entropy:
        info['joint_entropy'] = joint_entropy.mean()
        info['entropy_rate'] = entropy_rate.mean()

    return logprob, entropy_rate, info


def compute_ppo_loss(log_probs: jnp.ndarray, old_log_probs: jnp.ndarray, advantages: jnp.ndarray, config) -> Tuple[jnp.ndarray, Dict]:
    """ Compute PPO clipped surrogate loss """
    log_ratio = log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = ((ratio - 1) - log_ratio).mean()

    lower_bound = 1.0 - config.clip_epsilon * config.clip_min_epsilon_multiplier
    upper_bound = 1.0 + config.clip_epsilon * config.clip_min_epsilon_multiplier

    ratio_clipped_upper = jnp.mean(ratio > upper_bound)
    ratio_clipped_lower = jnp.mean(ratio < lower_bound)
    ratio_clipped_total = ratio_clipped_upper + ratio_clipped_lower

    clipped_ratio = jnp.clip(ratio, lower_bound, upper_bound)

    if config.adv_clip_min is not None:
        advantages = jnp.clip(advantages, config.adv_clip_min)

    # PPO loss
    pg_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    stats = {
        'ratio': ratio.mean(),
        'ratio_std': ratio.std(),
        'ratio_min': ratio.min(),
        'ratio_max': ratio.max(),
        'approx_kl': approx_kl,
        'ratio_clipped_upper': ratio_clipped_upper,
        'ratio_clipped_lower': ratio_clipped_lower,
        'ratio_clipped_total': ratio_clipped_total,
    }

    return pg_loss, stats


def _safe_max(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    x_min = x.min(axis=axis)
    x_max = x.max(axis=axis)
    return x_min * (x_min > 0) + x_max * (x_max < 0)

def _penultimate_safe_max(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    x_sorted = jnp.sort(x, axis=axis)  # sorted ascending along M
    x_min = x_sorted.take(0, axis=axis)
    x_max = x_sorted.take(-1, axis=axis)
    x_min2 = x_sorted.take(1, axis=axis)    # second smallest
    x_max2 = x_sorted.take(-2, axis=axis)  # second largest
    return x_min2 * (x_min > 0) + x_max2 * (x_max < 0)

def compute_ogpo_advantages(q_agg: jnp.ndarray, q_full: jnp.ndarray, config) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute advantages with different strategies for OGPO."""
    if config.adv_strategy == 'vanilla':
        if config.normalize_group:
            advantages = (q_agg - q_agg.mean(axis=0, keepdims=True)) / (q_agg.std(axis=0, keepdims=True) + 1e-8)
        else:
            advantages = q_agg - q_agg.mean(axis=0, keepdims=True)
        return advantages, q_agg

    elif config.adv_strategy == 'conservative':
        # Per-Q-network advantage: A_{i,m} = Q_m(s,a_i) - (1/G) sum_{i'} Q_m(s,a_{i'})
        adv_per_q = q_full - q_full.mean(axis=0, keepdims=True)  # [G, M, batch]
        return _safe_max(adv_per_q, axis=1), q_full.mean(axis=1)
        
    elif config.adv_strategy == 'penultimate':
        # Per-Q-network advantage: A_{i,m} = Q_m(s,a_i) - (1/G) sum_{i'} Q_m(s,a_{i'})
        adv_per_q = q_full - q_full.mean(axis=0, keepdims=True)  # [G, M, batch]
        return _penultimate_safe_max(adv_per_q, axis=1), q_full.mean(axis=1)

    if config.adv_strategy == 'awr':
        advantages = q_agg - q_agg.mean(axis=0, keepdims=True)
        weights = compute_awr_weights(advantages, config.awr_beta, config.get('awr_weight_max', 20.0),
                                      config.get('awr_mode', 'symmetric'))
        return weights, q_agg
    else:
        raise ValueError(f"Unknown adv_strategy: {config.adv_strategy}")


# ===================== chi2-Pessimistic OGPO (chiPO) =====================

def compute_chi_po_beta(q_full: jnp.ndarray, config) -> jnp.ndarray:
    """Anneal chi_po beta with Q confidence. q_full shape [G, M, batch]. Ensemble axis = 1."""
    q_std = jnp.std(q_full, axis=1).mean()  # std over ensemble M, mean over G and batch
    return config.chi_po_beta_base * (q_std / config.chi_po_q_std_target)


def compute_kl_reg_beta(q_full: jnp.ndarray, config) -> jnp.ndarray:
    """Anneal kl_reg beta with Q confidence. Same shape/semantics as compute_chi_po_beta."""
    q_std = jnp.std(q_full, axis=1).mean()
    return config.kl_reg_beta_base * (q_std / config.chi_po_q_std_target)


def compute_chi_po_advantages(
    q_full: jnp.ndarray,
    chi2_ratio: Optional[jnp.ndarray],
    beta: Optional[jnp.ndarray],
    config,
    log_ratio: Optional[jnp.ndarray] = None,
    beta_kl: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    q_mean = q_full.mean(axis=1)   # [G, batch]
    q_min = q_full.min(axis=1)     # [G, batch]

    # Pessimism blend: chi_po wins if both active (chi2_ratio - 1 drift vs log_ratio drift)
    if chi2_ratio is not None:
        pessimism_weight = jax.nn.sigmoid(config.chi_po_ensemble_alpha * (chi2_ratio - 1.0))
    else:
        alpha_kl = config.get('kl_reg_ensemble_alpha', config.chi_po_ensemble_alpha)
        pessimism_weight = jax.nn.sigmoid(alpha_kl * log_ratio)
    q_targ = (1.0 - pessimism_weight) * q_mean + pessimism_weight * q_min

    # Penalty: beta_chi * ratio + beta_kl * log_ratio
    penalty = jnp.zeros_like(q_targ)
    if chi2_ratio is not None and beta is not None:
        penalty = penalty + beta * chi2_ratio
    if log_ratio is not None and beta_kl is not None:
        penalty = penalty + beta_kl * log_ratio

    penalized_q = q_targ - penalty
    baseline = penalized_q.mean(axis=0, keepdims=True)
    advantages = penalized_q - baseline
    if config.normalize_group:
        advantages = advantages / (advantages.std(axis=0, keepdims=True) + 1e-8)
    return jax.lax.stop_gradient(advantages), q_targ


def compute_chi_po_ppo_loss(
    log_probs: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    beta: Optional[jnp.ndarray],
    config,
    beta_kl: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict]:
    log_ratio = log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = ((ratio - 1) - log_ratio).mean()

    eps = config.clip_epsilon * config.clip_min_epsilon_multiplier
    chi_po_active = beta is not None
    kl_reg_active = beta_kl is not None
    if chi_po_active:
        upper_bound = jnp.minimum(1.0 + eps, 1.0 + config.chi_po_R_max / beta)
    elif kl_reg_active:
        r_max_kl = config.get('kl_reg_R_max', config.chi_po_R_max)
        upper_bound = jnp.minimum(1.0 + eps, 1.0 + r_max_kl / beta_kl)
    else:
        upper_bound = 1.0 + eps
    lower_bound = 1.0 - eps
    clipped_ratio = jnp.clip(ratio, lower_bound, upper_bound)

    if config.adv_clip_min is not None:
        advantages = jnp.clip(advantages, config.adv_clip_min)

    pg_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    stats = {
        'ratio': ratio.mean(),
        'ratio_std': ratio.std(),
        'ratio_min': ratio.min(),
        'ratio_max': ratio.max(),
        'approx_kl': approx_kl,
        'ratio_clipped_upper': jnp.mean(ratio > upper_bound),
        'ratio_clipped_lower': jnp.mean(ratio < lower_bound),
        'ratio_clipped_total': jnp.mean(ratio > upper_bound) + jnp.mean(ratio < lower_bound),
        'chi_po_upper_bound': upper_bound,
        'chi_po_beta': beta if chi_po_active else jnp.float32(0.0),
        'kl_reg_beta': beta_kl if kl_reg_active else jnp.float32(0.0),
    }

    return pg_loss, stats


# ===================== FPO / FPO++ =====================

def compute_spo_penalty(
    ratios: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_epsilon: float,
) -> jnp.ndarray:
    abs_adv = jnp.abs(advantages)
    upper_excess = jnp.maximum(0.0, ratios - (1.0 + clip_epsilon))
    lower_excess = jnp.maximum(0.0, (1.0 - clip_epsilon) - ratios)
    excess = upper_excess + lower_excess  # one is always 0
    return jnp.square(excess) * abs_adv / (4.0 * clip_epsilon)


def compute_fpo_cfm_ratios(
    old_actor_fn: Callable,
    new_actor_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    n_mc: int,
    rng: jnp.ndarray,
    per_sample: bool,
    cfm_loss_clamp: float,
    cfm_diff_clamp: float,
    use_huber: bool = False,
    huber_delta: float = 1.0,
) -> Tuple[jnp.ndarray, dict]:
    batch_size = observations.shape[0]
    act_dim = actions.shape[-1]

    # 1. Draw shared MC samples
    rng, tau_key, eps_key = jax.random.split(rng, 3)
    taus = jax.random.uniform(tau_key, (n_mc,))                      # [n_mc]
    eps = jax.random.normal(eps_key, (n_mc, batch_size, act_dim))    # [n_mc, batch, act_dim]

    # 2. Compute noised actions and velocity targets
    taus_expanded = taus[:, None, None]                              # [n_mc, 1, 1]
    x_tau = taus_expanded * actions[None] + (1.0 - taus_expanded) * eps   # [n_mc, batch, act_dim]
    vel_target = actions[None] - eps                                      # [n_mc, batch, act_dim]

    # 3. Compute CFM loss for each MC sample
    def single_mc_cfm_loss(x_tau_i, vel_target_i, tau_i, actor_fn):
        t = jnp.full((batch_size, 1), tau_i)
        v_pred = actor_fn(observations, x_tau_i, t)
        diff = v_pred - vel_target_i
        if use_huber:
            # Huber loss (matches reference FPO: MSE when |err|<=delta, linear beyond)
            abs_diff = jnp.abs(diff)
            per_dim = jnp.where(
                abs_diff <= huber_delta,
                jnp.square(diff),
                2.0 * huber_delta * abs_diff - huber_delta ** 2,
            )
        else:
            per_dim = jnp.square(diff)
        return jnp.sum(per_dim, axis=-1)  # [batch]

    vmapped_cfm = jax.vmap(single_mc_cfm_loss, in_axes=(0, 0, 0, None))

    cfm_old = vmapped_cfm(x_tau, vel_target, taus, old_actor_fn)    # [n_mc, batch]
    cfm_new = vmapped_cfm(x_tau, vel_target, taus, new_actor_fn)    # [n_mc, batch]

    # 4. Clamp individual losses
    cfm_old = jnp.clip(cfm_old, 0.0, cfm_loss_clamp)
    cfm_new = jnp.clip(cfm_new, 0.0, cfm_loss_clamp)

    # 5. Compute ratios
    if per_sample:
        diff = cfm_old - cfm_new                                    # [n_mc, batch]
        diff = jnp.clip(diff, -cfm_diff_clamp, cfm_diff_clamp)
        ratios = jnp.exp(diff)                                      # [n_mc, batch]
    else:
        diff = cfm_old.mean(axis=0) - cfm_new.mean(axis=0)          # [batch]
        diff = jnp.clip(diff, -cfm_diff_clamp, cfm_diff_clamp)
        ratios = jnp.exp(diff)                                      # [batch]

    stats = {
        'cfm_old_mean': cfm_old.mean(),
        'cfm_new_mean': cfm_new.mean(),
        'ratio_mean': ratios.mean(),
        'ratio_std': ratios.std(),
    }

    return ratios, stats


def compute_fpo_ppo_loss(
    ratios: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_epsilon: float,
    per_sample: bool,
    use_aspo: bool,
    clip_min_epsilon_multiplier: float = 1.0,
    adv_clip_min: float = None,
) -> Tuple[jnp.ndarray, dict]:
    lower_bound = 1.0 - clip_epsilon * clip_min_epsilon_multiplier
    upper_bound = 1.0 + clip_epsilon * clip_min_epsilon_multiplier

    if adv_clip_min is not None:
        advantages = jnp.clip(advantages, adv_clip_min)

    if per_sample:
        adv = advantages[None, :]                                    # [1, G*batch]
    else:
        adv = advantages

    clipped_ratios = jnp.clip(ratios, lower_bound, upper_bound)

    if use_aspo:
        # PPO branch for positive advantages
        ppo_obj = jnp.minimum(ratios * adv, clipped_ratios * adv)
        # SPO branch for negative advantages
        spo_penalty = compute_spo_penalty(ratios, adv, clip_epsilon)
        spo_obj = ratios * adv - spo_penalty
        # Select based on advantage sign
        pos_mask = adv > 0
        objective = jnp.where(pos_mask, ppo_obj, spo_obj)
    else:
        objective = jnp.minimum(ratios * adv, clipped_ratios * adv)

    pg_loss = -jnp.mean(objective)

    # Stats (compatible with compute_ppo_loss output keys)
    ratio_flat = ratios.reshape(-1)
    approx_kl = jnp.float32(0.0)  # Not meaningful for FPO (no log-prob ratio)
    stats = {
        'ratio': ratio_flat.mean(),
        'ratio_std': ratio_flat.std(),
        'ratio_min': ratio_flat.min(),
        'ratio_max': ratio_flat.max(),
        'approx_kl': approx_kl,
        'ratio_clipped_upper': jnp.mean(ratio_flat > upper_bound),
        'ratio_clipped_lower': jnp.mean(ratio_flat < lower_bound),
        'ratio_clipped_total': jnp.mean(ratio_flat > upper_bound) + jnp.mean(ratio_flat < lower_bound),
    }

    return pg_loss, stats


# ===================== AWR (Advantage-Weighted Regression) for Flow Matching =====================

def compute_awr_weights(
    advantages: jnp.ndarray,
    beta: float,
    weight_max: float,
    mode: str = 'symmetric',
) -> jnp.ndarray:
    if mode == 'asymmetric':
        weights = jnp.exp(jnp.maximum(0.0, advantages) / beta)
    elif mode == 'positive_only':
        weights = jnp.exp(advantages / beta) * (advantages > 0).astype(jnp.float32)
    elif mode == 'symmetric':
        weights = jnp.exp(advantages / beta)
    else:
        raise ValueError(f"awr_mode={mode!r}; expected symmetric | asymmetric | positive_only")
    return jnp.clip(weights, 0.0, weight_max)


def compute_awr_cfm_loss(
    actor_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    n_mc: int,
    rng: jnp.ndarray,
    awr_beta: float,
    awr_weight_max: float = 20.0,
    awr_mode: str = 'symmetric',
) -> Tuple[jnp.ndarray, dict]:
    batch_size = observations.shape[0]
    act_dim = actions.shape[-1]

    # 1. Compute AWR weights based on mode
    weights = compute_awr_weights(advantages, awr_beta, awr_weight_max, awr_mode)
    weight_sum = weights.mean() + 1e-8
    weights = weights / weight_sum  # normalize so mean weight ≈ 1

    # 2. Draw MC samples for CFM loss
    rng, tau_key, eps_key = jax.random.split(rng, 3)
    taus = jax.random.uniform(tau_key, (n_mc,))
    eps = jax.random.normal(eps_key, (n_mc, batch_size, act_dim))

    # 3. Compute noised actions and velocity targets (OT flow matching)
    taus_expanded = taus[:, None, None]
    x_tau = taus_expanded * actions[None] + (1.0 - taus_expanded) * eps
    vel_target = actions[None] - eps

    # 4. Compute CFM loss for each MC sample
    def single_mc_cfm_loss(x_tau_i, vel_target_i, tau_i):
        t = jnp.full((batch_size, 1), tau_i)
        v_pred = actor_fn(observations, x_tau_i, t)
        return jnp.sum(jnp.square(v_pred - vel_target_i), axis=-1)  # [batch]

    cfm_losses = jax.vmap(single_mc_cfm_loss)(x_tau, vel_target, taus)  # [n_mc, batch]
    cfm_loss_per_sample = cfm_losses.mean(axis=0)  # [batch]

    # 5. Weighted loss
    loss = jnp.mean(weights * cfm_loss_per_sample)

    stats = {
        'awr_weight_mean': weights.mean(),
        'awr_weight_std': weights.std(),
        'awr_weight_max': weights.max(),
        'awr_weight_min': weights.min(),
        'awr_cfm_loss_mean': cfm_loss_per_sample.mean(),
        'awr_cfm_loss_std': cfm_loss_per_sample.std(),
        'awr_frac_positive': (advantages > 0).astype(jnp.float32).mean(),
    }

    return loss, stats


def compute_cfm_chi2_ratio(
    current_actor_fn: Callable,
    ref_actor_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    n_mc: int,
    rng: jnp.ndarray,
    cfm_loss_clamp: float = 100.0,
    cfm_diff_clamp: float = 5.0,
) -> jnp.ndarray:
    batch_size = observations.shape[0]
    act_dim = actions.shape[-1]

    rng, tau_key, eps_key = jax.random.split(rng, 3)
    taus = jax.random.uniform(tau_key, (n_mc,))
    eps = jax.random.normal(eps_key, (n_mc, batch_size, act_dim))

    taus_expanded = taus[:, None, None]
    x_tau = taus_expanded * actions[None] + (1.0 - taus_expanded) * eps
    vel_target = actions[None] - eps

    def single_mc_cfm(x_tau_i, vel_target_i, tau_i, actor_fn):
        t = jnp.full((batch_size, 1), tau_i)
        v_pred = actor_fn(observations, x_tau_i, t)
        return jnp.sum(jnp.square(v_pred - vel_target_i), axis=-1)

    vmapped_cfm = jax.vmap(single_mc_cfm, in_axes=(0, 0, 0, None))

    cfm_ref = vmapped_cfm(x_tau, vel_target, taus, ref_actor_fn)
    cfm_cur = vmapped_cfm(x_tau, vel_target, taus, current_actor_fn)

    cfm_ref = jnp.clip(cfm_ref, 0.0, cfm_loss_clamp)
    cfm_cur = jnp.clip(cfm_cur, 0.0, cfm_loss_clamp)

    diff = (cfm_ref - cfm_cur).mean(axis=0)  # [batch]
    diff = jnp.clip(diff, -cfm_diff_clamp, cfm_diff_clamp)
    ratio = jnp.exp(diff)

    return jax.lax.stop_gradient(ratio)


# MIP Stuff
def sample_mip_actions_sde(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        mip_t_star: float,
        act_dim: int,
        use_constant_noise: bool = False,
        use_tapered_noise: bool = False,
        error_correct_sde_to_ode: bool = False,  # accepted for API parity; see note
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        randn_clip_value: float = 3.0,
        act_min: float = -1.0,
        act_max: float = 1.0,
        is_encoded: bool = False,
        params: Any = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del error_correct_sde_to_ode  # not applicable to denoiser-style MIP step
    batch_size = observations.shape[0]

    def _sigma_for_t(t):
        if use_tapered_noise:
            return constant_noise_std * jnp.sqrt(jnp.maximum(1.0 - t, 0.0)) * jnp.ones((batch_size, act_dim))
        elif use_constant_noise:
            return constant_noise_std * jnp.ones((batch_size, act_dim))
        else:
            if params is not None:
                s = noise_fn(observations, t, params=params)
            else:
                s = noise_fn(observations, t)
            return jnp.maximum(s, min_noise_std)

    rng, key = jax.random.split(rng)
    x_0 = jax.random.normal(key, (batch_size, act_dim))
    chains = [x_0]
    sigmas = []

    # Initial log prob: p(x_0) ~ N(0, I)
    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    logprob = init_dist.log_prob(x_0).sum(-1)

    # Step 1: t=0
    t_0 = jnp.zeros((batch_size, 1))
    if params is not None:
        mean_0 = actor_fn(observations, x_0, t_0, params=params, is_encoded=is_encoded)
    else:
        mean_0 = actor_fn(observations, x_0, t_0, is_encoded=is_encoded)

    sigma_0 = _sigma_for_t(t_0)
    sigmas.append(sigma_0)

    step_0_dist = distrax.Normal(mean_0, sigma_0)
    rng, noise_key = jax.random.split(rng)
    a_0_hat = step_0_dist.sample(seed=noise_key)

    a_0_hat = jnp.clip(
        a_0_hat,
        mean_0 - randn_clip_value * sigma_0,
        mean_0 + randn_clip_value * sigma_0
    )

    logprob = logprob + step_0_dist.log_prob(a_0_hat).sum(-1)
    chains.append(a_0_hat)

    # Step 2: t=t*
    t_star = jnp.full((batch_size, 1), mip_t_star)
    if params is not None:
        mean_final = actor_fn(observations, a_0_hat, t_star, params=params, is_encoded=is_encoded)
    else:
        mean_final = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    sigma_final = _sigma_for_t(t_star)
    sigmas.append(sigma_final)

    step_final_dist = distrax.Normal(mean_final, sigma_final)
    rng, noise_key = jax.random.split(rng)
    actions = step_final_dist.sample(seed=noise_key)

    actions = jnp.clip(
        actions,
        mean_final - randn_clip_value * sigma_final,
        mean_final + randn_clip_value * sigma_final
    )

    logprob = logprob + step_final_dist.log_prob(actions).sum(-1)

    actions = jnp.clip(actions, act_min, act_max)
    chains.append(actions)

    chains = jnp.stack(chains, axis=1)
    sigmas = jnp.stack(sigmas, axis=1)

    return actions, chains, logprob, sigmas


def sample_mip_actions_ode(
        actor_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        mip_t_star: float,
        act_dim: int,
        act_min: float = -1.0,
        act_max: float = 1.0,
        noises: jnp.ndarray = None,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
    batch_size = observations.shape[0]

    if noises is None:
        actions = jax.random.normal(rng, (batch_size, act_dim))
    else:
        actions = noises

    # Step 1: t=0
    t_0 = jnp.zeros((batch_size, 1))
    a_0_hat = actor_fn(observations, actions, t_0, is_encoded=is_encoded)

    # Step 2: t=t*
    t_star = jnp.full((batch_size, 1), mip_t_star)
    actions = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    actions = jnp.clip(actions, act_min, act_max)
    return actions


def compute_mip_log_prob(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        chains: jnp.ndarray,
        mip_t_star: float,
        use_constant_noise: bool = False,
        use_tapered_noise: bool = False,
        error_correct_sde_to_ode: bool = False,  # accepted for API parity; unused
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        normalize_horizon: bool = False,
        normalize_dim: bool = False,
        is_encoded: bool = False,
        params: Any = None,
        get_entropy: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict]:
    del error_correct_sde_to_ode
    batch_size = chains.shape[0]
    act_dim = chains.shape[-1]

    def _sigma_for_t(t):
        if use_tapered_noise:
            return constant_noise_std * jnp.sqrt(jnp.maximum(1.0 - t, 0.0)) * jnp.ones((batch_size, act_dim))
        elif use_constant_noise:
            return constant_noise_std * jnp.ones((batch_size, act_dim))
        else:
            if params is not None:
                s = noise_fn(observations, t, params=params)
            else:
                s = noise_fn(observations, t)
            return jnp.maximum(s, min_noise_std)

    x_0 = chains[:, 0, :]
    a_0_hat = chains[:, 1, :]
    a_final = chains[:, 2, :]

    logprob = 0.0
    joint_entropy = 0.0
    logprob_steps = 0

    # Initial probability: p(x_0) ~ N(0, I)
    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    logprob_init = init_dist.log_prob(x_0).sum(-1)
    logprob = logprob + logprob_init
    logprob_steps += 1

    if get_entropy:
        entropy_init = init_dist.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_init

    # Step 1: Transition from x_0 to a_0_hat at t=0
    t_0 = jnp.zeros((batch_size, 1))
    if params is not None:
        mean_0 = actor_fn(observations, x_0, t_0, params=params, is_encoded=is_encoded)
    else:
        mean_0 = actor_fn(observations, x_0, t_0, is_encoded=is_encoded)

    sigma_0 = _sigma_for_t(t_0)

    dist_0 = distrax.Normal(mean_0, sigma_0)
    logprob_0 = dist_0.log_prob(a_0_hat).sum(-1)
    logprob = logprob + logprob_0
    logprob_steps += 1

    if get_entropy:
        entropy_0 = dist_0.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_0

    # Step 2: Transition from a_0_hat to a_final at t=t*
    t_star = jnp.full((batch_size, 1), mip_t_star)
    if params is not None:
        mean_final = actor_fn(observations, a_0_hat, t_star, params=params, is_encoded=is_encoded)
    else:
        mean_final = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    sigma_final = _sigma_for_t(t_star)

    dist_final = distrax.Normal(mean_final, sigma_final)
    logprob_final = dist_final.log_prob(a_final).sum(-1)
    logprob = logprob + logprob_final
    logprob_steps += 1

    if get_entropy:
        entropy_final = dist_final.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_final
        entropy_rate = joint_entropy / logprob_steps / act_dim
    else:
        entropy_rate = None

    # Normalization
    if normalize_horizon:
        logprob = logprob / logprob_steps

    if normalize_dim:
        logprob = logprob / act_dim

    info = {
        'mean_0': mean_0.mean(),
        'mean_final': mean_final.mean(),
        'sigma_0': sigma_0.mean(),
        'sigma_final': sigma_final.mean(),
        'logprob_mean': logprob.mean(),
        'logprob_steps': logprob_steps,
    }

    if get_entropy:
        info['joint_entropy'] = joint_entropy.mean()
        info['entropy_rate'] = entropy_rate.mean()

    return logprob, entropy_rate, info

