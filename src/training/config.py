import dataclasses
import difflib
import tyro
from openpi.training.config import (
    _CONFIGS,
    TrainConfig,
    DataConfig,
    pi0_config,
    SimpleDataConfig,
    AssetsConfig,
    ModelType,
    LeRobotLiberoDataConfig,
)
from typing import Literal, Sequence
import re

import openpi.training.optimizer as _optimizer
import openpi.policies.droid_policy as _droid_policy
import openpi.transforms as _openpi_transforms
import optax
import openpi.training.weight_loaders as weight_loaders
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class NormalizerState:
    bias: jax.Array
    scale: jax.Array
    ema_weight: float


class Normalizer:
    def __init__(self, ema_weight: float = 0.99):
        self._ema_weight = ema_weight

    def init(self) -> NormalizerState:
        return NormalizerState(
            bias=jnp.array(0.0),
            scale=jnp.array(1.0),
            ema_weight=self._ema_weight,
        )

    @staticmethod
    @jax.jit
    def update(normalizer_state: NormalizerState, bias: jax.Array, scale: jax.Array) -> NormalizerState:
        prev_bias = normalizer_state.bias
        prev_scale = normalizer_state.scale
        ema_weight = normalizer_state.ema_weight
        new_bias = (1.0 - ema_weight) * bias + ema_weight * prev_bias
        new_scale = (1.0 - ema_weight) * scale + ema_weight * prev_scale
        return normalizer_state.replace(bias=new_bias, scale=new_scale)


@dataclasses.dataclass(frozen=True)
class ConstantSchedule(_optimizer.LRScheduleConfig):
    """Constant learning rate schedule."""

    value: float = 5e-5

    def create(self) -> optax.Schedule:
        return optax.constant_schedule(self.value)


@dataclasses.dataclass(frozen=True)
class LinearSchedule(_optimizer.LRScheduleConfig):
    """Linear schedule that starts at init_value and ends at end_value"""

    init_value: float = 0.0
    end_value: float = 1.0
    transition_steps: int = 1000

    def create(self) -> optax.Schedule:
        return optax.linear_schedule(
            self.init_value, self.end_value, self.transition_steps
        )


@dataclasses.dataclass(frozen=True)
class StepSchedule(_optimizer.LRScheduleConfig):
    """Step schedule: returns init_value for steps < switch_step, then end_value."""

    init_value: float = 0.0
    end_value: float = 1.0
    switch_step: int = 1000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            schedules=[
                optax.constant_schedule(self.init_value),
                optax.constant_schedule(self.end_value),
            ],
            boundaries=[self.switch_step],
        )


@dataclasses.dataclass(frozen=True)
class NormalizerConfig:
    q_up: float = 0.95
    q_low: float = 0.05
    method: str = 'quantile'
    min_scale: float = 1.0
    ema_weight: float = 0.99


@dataclasses.dataclass(frozen=True)
class RLAlgorithmConfig:
    discount: float = 0.99
    buffer_capacity: int = 1024
    # Shared by both policy and critic sampling: fraction of each batch drawn from
    # the online replay buffer (rest comes from the offline SFT dataset).
    online_ratio: float = 0.5


@dataclasses.dataclass(frozen=True)
class PolicyTrainingConfig:
    update_interval: int = 1
    training_start_step: int = 0


@dataclasses.dataclass(frozen=True)
class CriticTrainingConfig:
    update_interval: int = 1
    training_start_step: int = 0
    use_ema: bool = True
    ema_decay: float = 0.995
    reduction: str = "min"
    encoder_hidden_dims: Sequence[int] = (512, 512)
    decoder_hidden_dims: Sequence[int] = (256, 256)
    num_qs: int = 2
    num_vs: int = 2
    num_updates_per_batch: int = 1
    td_weight_schedule: StepSchedule = StepSchedule(init_value=0.0, end_value=1.0, switch_step=1_000)
    pre_training_steps: int = 1_000
    num_value_bins: int = 1  # 1: Gaussian (MSE-equivalent), >1 = Categorical over bins
    value_lower_bound: float | None = None  # If None: auto-computed from reward type and discount
    value_upper_bound: float | None = None
    value_target_type: str = "two_hot"  # "one_hot" | "two_hot"
    use_distributional_critic: bool = False
    distributional_target_reduction: str = "min"
    inference_start_step: int = 100
    # Critics are lightweight (MLP-only); a larger batch than the policy often
    # stabilises TD learning without a meaningful memory cost.
    batch_size: int | None = None  # If None: use the global config.batch_size
    # Class-level attributes (not dataclass fields) so subclasses can override the default.
    lr_schedule = ConstantSchedule(value=1e-4)
    optimizer = _optimizer.AdamW(clip_gradient_norm=1.0)
    # BRONet critic (alternative to the MLP backbone)
    use_bronet: bool = False
    bronet_hidden_dim: int = 512
    bronet_depth: int = 2
    # --- conservative twin-critic AWR knobs (defaults reproduce current behavior) ---
    q_bootstrap_reduction: str | None = None  # V-ensemble reduction for Q target; None -> `reduction`
    per_critic_value_target: bool = False  # True: V_i regresses to Q_i (needs num_vs == num_qs)
    # Per-task critics (multi-task): T disjoint Q ensembles and T disjoint V
    # ensembles with NO shared parameters, one per distinct collect.tasks entry,
    # routed per sample by a `task_index` transition field in the replay buffer
    # (src/rl/networks/per_task_critic.py, src/rl/task_registry.py). A sample of
    # task t is only ever scored by task t's critics, in the TD update and in
    # the OGPO advantage alike. Slots are keyed on the task ID (libero_90_79),
    # not the prompt string. Also switches the critic loss to a per-task mean
    # averaged over the tasks present in the batch, and the critic grad clip to
    # one clip_by_global_norm per task subtree (a single global clip would let
    # task A's gradient magnitude throttle task B's update). Must equal the
    # number of distinct collect.tasks ids (asserted at learner construction;
    # the registry must be full after the first collection round).
    # Cost: T x the critic forward/backward and T x critic params+Adam+EMA.
    # Analytic, NOT measured: the mt4 recipe forces --rl.critic.use_bronet
    # --rl.critic.bronet_hidden_dim 1024 (2 heads), ~0.42 GB today -> ~1.7 GB at
    # T=4; the 10-head `_ref` path is ~9 GB at T=4. Measure before an
    # unfrozen-backbone arm. New-run only:
    # the critic param tree changes shape, so a single-critic checkpoint or
    # buffer shard does not resume into it. OGPO only. None = today's shared
    # critics, bit-identical. docs/changes/2026-08-21-per-task-critics/.
    num_tasks: int | None = None


# Define hyperparameter structures for your algorithms
@dataclasses.dataclass(frozen=True)
class FilteredSFTLearnerConfig(RLAlgorithmConfig):
    policy: PolicyTrainingConfig = PolicyTrainingConfig()


@dataclasses.dataclass(frozen=True)
class AdvantageWeightedSFTLearnerConfig(FilteredSFTLearnerConfig):
    critic: CriticTrainingConfig = CriticTrainingConfig()
    beta: float = 0.05
    weight_clip: float = 20.0
    advantage_scale: float = 10.0
    normalizer_config: NormalizerConfig = NormalizerConfig()
    use_mc_returns: bool = False
    store_success_episodes_only: bool = False
    normalize_advantages: bool = False
    # n_samples > 1 enables best-of-N collection: the agent samples N candidate
    # action sequences and selects the one with the highest Q-value.
    n_samples: int = 1
    # Weight for an auxiliary BC loss on successful transitions only.
    # Combined loss = AWR loss + filtered_sft_weight * mean(is_success * BC loss).
    filtered_sft_weight: float = 0.0
    awr_loss_weight: float = 1.0
    # --- conservative twin-critic AWR knobs (defaults reproduce current behavior) ---
    advantage_weight_type: str = "exp"  # "exp": exp(adv/beta) weights; "relu": max(adv, 0) weights
    advantage_combination: str = "reduced"  # "reduced": reduce(Q)-reduce(V); "conservative": per-critic A_i combine; "grpo_conservative" (OGPO G>1 only): per-head Q_i - mean_G(Q_i), sign-unanimous combine — V(s) never enters


@dataclasses.dataclass(frozen=True)
class BestofNLearnerConfig(FilteredSFTLearnerConfig):
    online_ratio: float = 1.0
    critic: CriticTrainingConfig = CriticTrainingConfig()
    n_samples: int = 32
    discount: float = 0.995
    train_on_policy_value_function: bool = False


@dataclasses.dataclass(frozen=True)
class MPOWeightedSFTLearnerConfig(AdvantageWeightedSFTLearnerConfig):
    store_buffer_actions_in_batch: bool = False


@dataclasses.dataclass(frozen=True)
class FlowGRPOSFTLearnerConfig(MPOWeightedSFTLearnerConfig):
    group_size: int = 8
    num_steps: int = 10
    noise_level: float = 0.3
    normalize_adv: bool = True
    use_mpo_advantage_weight: bool = True


@dataclasses.dataclass(frozen=True)
class OGPOSFTLearnerConfig(AdvantageWeightedSFTLearnerConfig):
    # v1 is on-policy only — no LeRobot offline data, no success buffer.
    online_ratio: float = 1.0
    # PPO / IS-ratio
    group_num_samples: int = 8
    clip_epsilon: float = 0.01
    entropy_coeff: float = 0.0
    # Stochastic flow sampling
    num_sde_steps: int = 10
    noise_level: float = 0.3
    # Advantage shaping ('vanilla' = group-mean baseline, 'max' = group-max
    # baseline, 'subtract_v' = q - v with no group baseline).
    adv_strategy: str = "vanilla"
    adv_clip_min: float | None = None
    # BC regularization on the on-policy actions in the same batch.
    bc_coeff: float = 1.0
    use_bc_regularization: bool = True
    # When True, successful episodes are additionally stored in a dedicated
    # success buffer and the BC anchor draws its (obs, actions) batch from it
    # (falling back to the online batch until the buffer holds a full batch).
    use_success_buffer: bool = False
    success_buffer_capacity: int = 50_000
    # Performance-only: compute the PaliGemma prefix ONCE per state and tile
    # its KV cache across the G group members in the sampler and rescorer,
    # instead of running G redundant prefix forwards on identical observations.
    # Semantically equivalent (allclose, certified by tests/ogpo/
    # test_group_dedup.py); big win for group_num_samples > 1.
    # MEMORY-LOAD-BEARING when backbone_lora is on: with adapters trainable the
    # rescorer's prefix pass is a live backward, so dedup-off pays G backbone
    # backwards per update instead of one (docs/changes/2026-08-29-backbone-lora/).
    dedup_group_prefix: bool = False
    # Multitask: normalize advantages by the per-task std within the batch
    # (tasks identified by tokenized prompt). Equalizes advantage scale across
    # tasks so cross-task value differences can't dominate the PPO gradient.
    normalize_advantage_per_task: bool = False
    # Multitask: sample the success-buffer BC batch balanced across tasks
    # (equal share per task with recorded successes) instead of uniformly over
    # transitions, so abundant-success tasks don't monopolize the BC anchor.
    balance_success_buffer_tasks: bool = False
    # Treat the EMA train state as the "old" policy (PPO denominator).
    # When False, the current params are used (stop-gradient'd).
    use_ema_as_old_policy: bool = True
    # Log-prob normalization for the PPO ratio. Mirrors the official OGPO
    # ``normalize_denoising_horizon`` / ``normalize_act_space_dimension``
    # knobs (see ``ogpo/configs/algos/ogpo.yaml``). With both on, the
    # log-ratio is per-(sde_step, horizon_pos, action_dim) — making
    # ``clip_epsilon`` a meaningful per-dim bound.
    normalize_denoising_horizon: bool = True
    normalize_act_space_dimension: bool = True
    # EMA-quantile advantage normalizer (the AWR normalizer applied to the
    # group-centered PPO advantage): divide by an EMA of the per-update
    # (q95 - q05) spread, floored at ``normalizer_config.min_scale``. Pins the
    # advantage scale (= effective policy LR) against critic-spread drift.
    # Reuses the inherited ``normalizer_config`` (q_up/q_low/min_scale/
    # ema_weight). The running scale PERSISTS across a resume: it rides in the
    # resume manifest's ``extra`` block (OGPOAgentLearner.save_extra_resume_state,
    # even when the success buffer is off) and is restored in __init__. Only a
    # resume from a manifest written before 2026-08-27 has no ``extra``; that
    # path warns and re-warms from min_scale within ~1/(1-ema_weight) policy
    # updates, as every resume used to.
    normalize_group_advantage: bool = False
    # Symmetric per-sample advantage clip, applied AFTER normalization. Bounds
    # the worst single-sample gradient contribution (the PPO analog of AWR's
    # weight_clip); None disables.
    adv_clip_sym: float | None = None
    # Micro-batches accumulated per policy optimizer step. Each micro-batch is
    # a fresh ``batch_size``-state sample (fresh success-BC batch too); grads
    # and logged aux are averaged. Raises state diversity per update at
    # unchanged peak memory, costing proportional wall-clock.
    policy_grad_accum: int = 1
    # Critic-only "digestion burst" after each collection round (skipped at
    # step 0, where policy.training_start_step already provides the same
    # actor-free critic head start): run this many extra critic updates on
    # the refreshed buffer BEFORE the next policy update, so the actor never
    # ranks fresh actions with a critic that hasn't fit the newly collected
    # distribution. 0 disables.
    post_collection_critic_steps: int = 0
    # Train the BURST's critic updates on pure MC returns (td_weight forced
    # to 0 inside the burst only; the regular loop keeps its schedule). MC
    # targets need no backward bootstrapping through the episode, so a fresh
    # high-value region calibrates in ~one pass instead of many TD sweeps.
    burst_use_mc_targets: bool = False
    # Critic updates per trainer step (fresh buffer batch EACH update, unlike
    # critic.num_updates_per_batch which reuses one batch). 1 = today's
    # behavior. Chronic critic-priority: the standing-ratio alternative to
    # the post-collection burst.
    critic_utd: int = 1
    # Success oversampling for the critic: one EXTRA critic update per trainer
    # step on a success-only batch, on top of the all-data batches. Mirrors the
    # reference's `use_success_buffer_q` / `critic_update_sb` (ogpo.py:1581-1585),
    # on in 9 of its 15 recipes including all three PaliGemma ones. Needs
    # use_success_buffer=True; silently inert until that buffer holds a full
    # critic batch. False reproduces today's behavior exactly.
    critic_success_oversample: bool = False
    # Policy warmstart: before this step the PG term is zeroed (advantage
    # multiplied by 0 — sampling/rescoring still run, keeping rng streams and
    # jit structure identical), so the actor trains on the BC anchor alone
    # while the critic calibrates on the improving data. 0 = PPO from start.
    pg_start_step: int = 0
    # Ralf-style filtered SFT for the BC anchor: instead of sampling the BC
    # batch from the success buffer, use the ONLINE batch weighted per-sample
    # by is_success (failures contribute zero BC loss). Mutually exclusive
    # with use_success_buffer in spirit; if both set, this wins.
    bc_filtered_sft: bool = False
    # PG ramp after the warmstart handoff: advantages are scaled by
    # min(1, (step - pg_start_step)/pg_ramp_steps), linearly rebuilding the
    # effective-LR curriculum that from-scratch runs get for free (a newborn
    # critic's tiny Q-spreads). Without it, a mature critic's full-size
    # advantages hit the BC-tuned policy in ONE step (measured: grad_norm
    # 0.05 -> 1.8 at the WS 20k handoff). 0 disables (hard switch).
    pg_ramp_steps: int = 0


@dataclasses.dataclass(frozen=True)
class DSRLLearnerConfig(RLAlgorithmConfig):
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    # Network architecture (kept explicit for parity across scripts/experiments).
    critic_decoder_hidden_dims: tuple[int, ...] = (128, 128, 128)
    policy_decoder_hidden_dims: tuple[int, ...] = (128, 128, 128)
    critic_num_qs: int = 10
    critic_reduction: str = "mean"
    backup_entropy: bool = False
    critic_update_frequency: int = 1
    actor_update_frequency: int = 1
    critic_ema_decay: float | None = 0.995
    encoder_type: str = "small"
    encoder_norm: str = "group"
    use_spatial_softmax: bool = True
    softmax_temperature: float = 1.0
    image_latent_dim: int = 50
    use_image_bottleneck: bool = True
    use_state_branch: bool = True
    autotune_alpha: bool = True
    init_alpha: float = 1.0
    target_entropy: str | float = "auto"
    policy_distribution: str = "tanh_normal"
    sac_image_size: int = 64
    random_crop_padding: int = 4
    warmup_gaussian_noise: bool = True


@dataclasses.dataclass(frozen=True)
class CollectionConfig:
    collect_interval: int = 200
    env_num: int = 4
    env_resolution: int = 256
    resize_image_h: int = 224
    resize_image_w: int = 224
    num_rollouts: int = 50
    num_initial_rollouts: int | None = None
    domain: Literal["libero", "molmo"] = "libero"
    tasks: list[str] | str = dataclasses.field(
        default_factory=lambda: ["libero_90_59"],
        metadata={
            "help": (
                "Task(s) to collect. Can be a single string (e.g., 'libero_90_59') or a list. "
                "Supports ranges (e.g., 'libero_90_22-56') and optional multipliers (e.g., 'libero_90_59x4' "
                "or 'libero_90_22-56x4')."
            )
        },
    )
    eval_tasks: list[str] | str = dataclasses.field(
        default_factory=lambda: ["libero_90_59"],
        metadata={
            "help": (
                "Task(s) to evaluate. Can be a single string (e.g., 'libero_90_59') or a list. "
                "Supports ranges (e.g., 'libero_90_22-56') and optional multipliers (e.g., 'libero_90_59x4' "
                "or 'libero_90_22-56x4')."
            )
        },
    )
    replan_steps: int = 5
    num_steps_wait: int = 10
    use_time_to_success_as_reward: bool = True
    # Reward paid on the terminating (success) step, on top of the -1/step time
    # penalty. 0.0 reproduces the original behavior exactly (success = the mere
    # absence of the penalty), so every existing config and recipe is unchanged.
    # The reference OGPO pays +5.0 on each success step (envs/robomimic_utils.py:457
    # and :467 in OGPO_public -- robomimic's raw success reward is 1.0, the -1.0 shift
    # lands it at 0.0, then += 5.0), sustained over post_success_steps=8 extra steps,
    # so up to +45 against its -100 failure floor = 45%. The same ratio against our
    # -200 floor is 90.0. Only read when use_time_to_success_as_reward is True.
    success_reward_bonus: float = 0.0
    fix_mc_returns: bool = True
    store_prefix_rep: bool = False
    eval_env_num: int = 4
    eval_interval: int = 300
    num_eval_rollouts: int = 32
    max_episode_steps: int = 400  # used to auto-compute value bounds
    # Multiplies the domain's TimeLimit truncation (the libero suite map at
    # envs/libero.py:126-138, molmo's 450) AND the T in the discount>=1
    # value-bound fallback, so the two cannot drift. 1 == today's behavior;
    # bounds are T-independent for discount < 1, so this only lengthens episodes.
    episode_steps_multiplier: int = 1

    def expand_tasks(self, tasks: str) -> list[str]:
        # Expand task ranges and handle multipliers
        expanded_tasks = []
        for task in tasks:
            # 1. Extract optional multiplier (e.g., "x4")
            multiplier = 1
            base_task = task
            mult_match = re.search(r"x(\d+)$", task)
            if mult_match:
                multiplier = int(mult_match.group(1))
                base_task = task[:mult_match.start()]

            # 2. Check if the base task is a range
            range_match = re.match(r"(.+)_(\d+)-(\d+)$", base_task)
            sub_tasks = []
            if range_match:
                prefix = range_match.group(1)
                start = int(range_match.group(2))
                end = int(range_match.group(3))
                for i in range(start, end + 1):
                    sub_tasks.append(f"{prefix}_{i}")
            else:
                sub_tasks.append(base_task)

            # 3. Add to expanded list, repeating by the multiplier
            for sub_task in sub_tasks:
                expanded_tasks.extend([sub_task] * multiplier)
        return expanded_tasks

    def __post_init__(self):
        tasks = [self.tasks] if isinstance(self.tasks, str) else self.tasks
        expanded_tasks = self.expand_tasks(tasks)
        object.__setattr__(self, 'tasks', expanded_tasks)

        eval_tasks = [self.eval_tasks] if isinstance(self.eval_tasks, str) else self.eval_tasks
        expanded_eval_tasks = self.expand_tasks(eval_tasks)
        object.__setattr__(self, 'eval_tasks', expanded_eval_tasks)


@dataclasses.dataclass(frozen=True)
class OnlineDataConfig(DataConfig):
    # additional LeRobot repo paths to include (keeps repos separate but concatenates them for training)
    additional_repo_paths: Sequence[str] = ()


@dataclasses.dataclass(frozen=True)
class OnlineTrainConfig(TrainConfig):
    # additional configs for online training
    group_name: str = "online_training"
    collect: CollectionConfig = CollectionConfig()
    rl: RLAlgorithmConfig = FilteredSFTLearnerConfig()
    max_runtime: int = 60 * 60 * 24
    requeue_before_eval: bool = False
    free_buffer_before_eval: bool = False
    default_prompt: str | None = None
    # Backbone LoRA (docs/changes/2026-08-29-backbone-lora/). A REWRITE flag,
    # not a passive field: __post_init__ restates BOTH model.paligemma_variant
    # ("gemma_2b_lora") and freeze_filter (adapter-aware) so the two can never
    # disagree — disagreement is the CLI trap the guard below rejects. Cost:
    # +27.87M trainable params (+6.5%), ~+0.5 GB steady state, and a newly-live
    # backward through the 2B backbone in jit-2a/2b (activation cost is the real
    # unknown; measure with scripts/exp_ogpo_unfrozen_backbone_memdiag.py).
    # Requires an OGPOSFTLearnerConfig; see the raise in __post_init__.
    backbone_lora: bool = False
    
    def __post_init__(self):
        super().__post_init__()

        if self.collect.episode_steps_multiplier < 1:
            # gymnasium's TimeLimit truncates on elapsed >= max_episode_steps, so
            # 0 or negative truncates every episode at step 0 -- a run that trains
            # and means nothing. A typo'd EP_MULT=0 must die here, not at eval.
            raise ValueError(
                f"collect.episode_steps_multiplier must be >= 1, got "
                f"{self.collect.episode_steps_multiplier}. Set "
                "--collect.episode_steps_multiplier 1 (default) or a positive "
                "multiplier (EP_MULT in scripts/ogpo_multitask_4task.sh)."
            )

        if isinstance(self.rl, (BestofNLearnerConfig, AdvantageWeightedSFTLearnerConfig)):
            if self.rl.critic.use_distributional_critic:
                # The C51 distributional backup is only implemented for best-of-N.
                # Other critic-based algos (AWR and its subclasses MPO/FlowGRPO/OGPO)
                # share CriticTrainingConfig, so the flag is settable there but would
                # be silently ignored — reject it explicitly to avoid that footgun.
                assert isinstance(self.rl, BestofNLearnerConfig), (
                    "use_distributional_critic=True is only supported for "
                    "BestofNLearnerConfig; it is not implemented for "
                    f"{type(self.rl).__name__}."
                )
                assert self.rl.critic.num_value_bins > 1, (
                    "use_distributional_critic=True requires num_value_bins > 1; "
                    "set rl.critic.num_value_bins (e.g. 51) in the config."
                )
            if self.rl.critic.num_tasks is not None:
                # Mirror image of the footgun above: per-task critics live in the
                # AWR critic module (src/rl/advantage_weighted_sft/update_critic.py);
                # the best-of-N copy is untouched and would silently ignore the flag.
                # The learner-level check (which subclasses actually thread
                # task_index) is in AdvantageWeightedSFTLearner.__init__.
                if not isinstance(self.rl, OGPOSFTLearnerConfig):
                    raise ValueError(
                        f"rl.critic.num_tasks={self.rl.critic.num_tasks} is only "
                        "implemented for OGPOSFTLearnerConfig (per-task critics, "
                        "docs/changes/2026-08-21-per-task-critics/); "
                        f"{type(self.rl).__name__} would silently ignore it. "
                        "Set rl.critic.num_tasks=None."
                    )
                if self.rl.critic.num_tasks < 1:
                    raise ValueError(
                        f"rl.critic.num_tasks must be >= 1 or None, got {self.rl.critic.num_tasks}."
                    )
            # Value bounds are resolved lazily in get_value_bounds(), not cached
            # here: caching into value_lower/upper_bound would suppress
            # re-resolution after tyro/YAML overrides.

        # --- Backbone LoRA (docs/changes/2026-08-29-backbone-lora/) ----------
        if self.backbone_lora:
            if not isinstance(self.rl, OGPOSFTLearnerConfig):
                # Same footgun-rejection shape as the critic.num_tasks branch
                # above: the flag rewrites freeze_filter to the OGPO
                # backbone-frozen filter, and on any other online config the
                # filter is nnx.Nothing (everything trainable), so the rewrite
                # would FREEZE ~430M params that train today and look normal.
                raise ValueError(
                    "backbone_lora=True is only defined for OGPOSFTLearnerConfig: "
                    "it rewrites freeze_filter to the OGPO backbone-frozen filter, "
                    f"which would FREEZE parameters {type(self.rl).__name__} trains "
                    "today. Drop --backbone_lora, or use one of the OGPO configs "
                    "(pi05_libero_online_ogpo_sft / _sft_pertask / _ref)."
                )
            if not isinstance(self.model, pi0_config.Pi0Config):
                raise ValueError(
                    "backbone_lora=True sets model.paligemma_variant='gemma_2b_lora', "
                    f"which only exists on Pi0Config; got {type(self.model).__name__}. "
                    "Drop --backbone_lora."
                )
            # dataclasses.replace, NOT a restated Pi0Config: replace preserves
            # pi05=True / action_horizon / discrete_state_input, which
            # make_base_libero_config sets wholesale and a restated model would
            # silently drop. Idempotent by construction (tyro re-instantiates
            # the dataclass, re-running this rewrite on its own output).
            object.__setattr__(
                self, "model",
                dataclasses.replace(self.model, paligemma_variant="gemma_2b_lora"),
            )
            object.__setattr__(
                self, "freeze_filter", _make_ogpo_freeze_filter(allow_lora=True)
            )

        # Guard, evaluated AFTER the rewrite so it also certifies it. Catches
        # the trap that is otherwise live: `--model.paligemma_variant
        # gemma_2b_lora` without the flag leaves the 10 adapter leaves matched
        # by the freeze filter -> randomly initialized (openpi lora.py inits
        # BOTH factors normal(0.01)), never loaded from the checkpoint
        # (weight_loaders back-fills them as ShapeDtypeStructs), never trained.
        # KL, clipfrac, alive-fraction and every grad norm look normal.
        if (
            isinstance(self.model, pi0_config.Pi0Config)
            and "lora" in self.model.paligemma_variant
        ):
            import flax.nnx as nnx

            freezes = nnx.filterlib.to_predicate(self.freeze_filter)
            # All 10 adapter leaves of the gemma_2b_lora tree, not one probe: a
            # filter freezing only a SUBSET (e.g. just the FFN adapters, or
            # just the lora_b factors) would slip a single-path probe and
            # recreate the trap for that subset (verifier finding F2,
            # docs/changes/2026-08-29-backbone-lora/VERIFICATION.md).
            probe_paths = (
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"),
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_b"),
                ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_a"),
                ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_b"),
                ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "lora_a"),
                ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "lora_b"),
                ("PaliGemma", "llm", "layers", "mlp", "gating_einsum_lora_a"),
                ("PaliGemma", "llm", "layers", "mlp", "gating_einsum_lora_b"),
                ("PaliGemma", "llm", "layers", "mlp", "linear_lora_a"),
                ("PaliGemma", "llm", "layers", "mlp", "linear_lora_b"),
            )
            probe_leaf = nnx.VariableState(nnx.Param, 0.0)
            if any(freezes(p, probe_leaf) for p in probe_paths):
                raise ValueError(
                    f"model.paligemma_variant={self.model.paligemma_variant!r} "
                    "creates LoRA adapters, but freeze_filter freezes them — a "
                    "randomly perturbed, permanently untrainable backbone with "
                    "normal-looking metrics. Pass --backbone_lora instead of "
                    "setting --model.paligemma_variant by hand: it rewrites the "
                    "variant AND the freeze filter together."
                )


def make_base_libero_config(
    name: str, rl_config: RLAlgorithmConfig, **kwargs
) -> OnlineTrainConfig:
    """
    Factory function to generate a base OnlineTrainConfig.
    Injects the specific RL algorithm config to keep the _CONFIGS list DRY.

    Extra kwargs are forwarded to OnlineTrainConfig (e.g. freeze_filter,
    ema_decay, num_train_steps overrides).
    """
    defaults = dict(
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                # load norm_stats from pretrained checkpoint
                assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
            ),
            base_config=OnlineDataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=ConstantSchedule(value=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10_000,
        num_workers=4,
    )
    defaults.update(kwargs)
    return OnlineTrainConfig(name=name, rl=rl_config, **defaults)


def make_base_molmo_config(
        name: str, rl_config: RLAlgorithmConfig
) -> OnlineTrainConfig:
    """
    Factory function to generate a base OnlineTrainConfig for Molmo.
    Injects the specific RL algorithm config to keep the _CONFIGS list DRY.
    """
    return OnlineTrainConfig(
        name=name,
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=15, discrete_state_input=False
        ),
        data=SimpleDataConfig(
            repo_id=None,
            assets=AssetsConfig(
                # load norm_stats from pretrained checkpoint
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
            data_transforms=lambda model: _openpi_transforms.Group(
                inputs=[_droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[
                    _openpi_transforms.AbsoluteActions(
                        _openpi_transforms.make_bool_mask(7, -1)
                    ),
                    _droid_policy.DroidOutputs(),
                ],
            ),
            base_config=OnlineDataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=ConstantSchedule(value=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params"
        ),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10_000,
        num_workers=4,  # override default num_workers
        rl=rl_config,
        collect=CollectionConfig(
            domain="molmo",
            max_episode_steps=450,
            resize_image_h=224,
            resize_image_w=224,
        )
    )


def _make_ogpo_freeze_filter(*, allow_lora: bool = False):
    """Freeze PaliGemma LLM (except the action expert) and the SigLIP vision
    tower. The action expert lives at LLM stack index 1 — matched by the
    ``.*llm.*_1.*`` path regex, mirroring ``Pi0Config.get_freeze_filter``.

    Trainable params after this filter:
      * action expert transformer blocks (LLM stack index 1)
      * the small action heads: state_proj, action_in_proj,
        action_time_mlp_in/out, action_out_proj
    Everything else (PaliGemma LLM stack index 0, SigLIP image tower) is
    frozen.

    ``allow_lora=True`` punches one hole: the ``.*lora.*`` leaves on LLM stack 0
    stay TRAINABLE (rank-16 adapters on q/kv/attn_vec einsums + both FFN
    matmuls, all 18 layers — 10 leaves, 27.87M params). Every other polarity is
    unchanged, so trainable goes 19 -> 29 leaves / 430.10M -> 457.97M params;
    SigLIP and the 9 stack-0 base leaves stay frozen. Set only by
    ``OnlineTrainConfig.__post_init__`` when ``backbone_lora`` is on
    (docs/changes/2026-08-29-backbone-lora/).

    Deliberately NOT built from ``Pi0Config.get_freeze_filter()``: that filter
    returns ``nnx.Nothing`` the moment the variant flips back to ``gemma_2b``,
    which would silently unfreeze the WHOLE backbone. This form is
    inert-but-safe under the same flip — a tree with no lora leaves gets exactly
    the ``allow_lora=False`` behavior (tests/ogpo/test_backbone_lora_config.py).
    """
    import flax.nnx as nnx
    import openpi.shared.nnx_utils as nnx_utils

    gemma_params  = nnx_utils.PathRegex(".*llm.*")
    action_expert = nnx_utils.PathRegex(".*llm.*_1.*")
    img_params    = nnx_utils.PathRegex(".*PaliGemma/img.*")
    llm_backbone = [gemma_params, nnx.Not(action_expert)]
    if allow_lora:
        llm_backbone.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
    return nnx.Any(
        nnx.All(*llm_backbone),
        img_params,
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS.extend(
    [
        #
        # Online training configs.
        #
        # These train configs define the hyperparameters for online data collection and fine-tuning.
        # 1. Filtered SFT
        make_base_libero_config(
            name="pi05_libero_online_filtered_sft",
            rl_config=FilteredSFTLearnerConfig(),
        ),
        make_base_molmo_config(
            name="pi05_molmo_online_filtered_sft",
            rl_config=FilteredSFTLearnerConfig(online_ratio=1.0),
        ),
        # 2. Advantage Weighted SFT (AWSFT)
        make_base_libero_config(
            name="pi05_libero_online_aw_sft",
            rl_config=AdvantageWeightedSFTLearnerConfig(
                policy=PolicyTrainingConfig(update_interval=20, training_start_step=100),
            ),
        ),
        # 3. MPO Weighted SFT
        make_base_libero_config(
            name="pi05_libero_online_mpo_sft",
            rl_config=MPOWeightedSFTLearnerConfig(
                store_buffer_actions_in_batch=False,
                policy=PolicyTrainingConfig(update_interval=20, training_start_step=100),
            ),
        ),
        make_base_libero_config(
            name="pi05_libero_online_flow_grpo_sft",
            rl_config=FlowGRPOSFTLearnerConfig(
                store_buffer_actions_in_batch=True,
                policy=PolicyTrainingConfig(update_interval=20, training_start_step=100),
            ),
        ),
        # 4. Best of N
        make_base_libero_config(
            name="pi05_libero_online_best_of_n",
            rl_config=BestofNLearnerConfig(),
        ),
        make_base_molmo_config(
            name="pi05_molmo_online_best_of_n",
            rl_config=BestofNLearnerConfig(online_ratio=1.0),
        ),
        make_base_libero_config(
            name="pi05_libero_online_dsrl",
            rl_config=DSRLLearnerConfig(),
        ),
        # OGPO: PPO on flow policies with on-policy SDE log-probs and a BC
        # anchor on the same on-policy batch. v1 freezes the PaliGemma
        # backbone + SigLIP tower; only the action expert and the small
        # action heads are trainable. See docs/ogpo_agent_plan.md.
        make_base_libero_config(
            name="pi05_libero_online_ogpo_sft",
            rl_config=OGPOSFTLearnerConfig(
                policy=PolicyTrainingConfig(update_interval=20, training_start_step=100),
                group_num_samples=8,
            ),
            freeze_filter=_make_ogpo_freeze_filter(),
        ),
        # OGPO with PER-TASK critics for the 4-task LIBERO arm
        # (scripts/ogpo_multitask_4task.sh): identical to `pi05_libero_online_ogpo_sft`
        # except rl.critic.num_tasks=4 -- one disjoint Q/V pair per task, routed by
        # the buffer's task_index (see CriticTrainingConfig.num_tasks) -- and the
        # 4-task collect/eval set baked in, because num_tasks must equal the number
        # of distinct collect.tasks (asserted at learner construction) and the
        # single-task CollectionConfig default would make the name unrunnable on
        # its own. Same dataclass, so the isinstance dispatch in scripts/exp.py:74-85
        # is untouched. Rationale: docs/changes/2026-08-21-per-task-critics/.
        make_base_libero_config(
            name="pi05_libero_online_ogpo_sft_pertask",
            rl_config=OGPOSFTLearnerConfig(
                policy=PolicyTrainingConfig(update_interval=20, training_start_step=100),
                group_num_samples=8,
                critic=CriticTrainingConfig(num_tasks=4),
            ),
            collect=CollectionConfig(
                tasks=["libero_90_79", "libero_90_31", "libero_90_82", "libero_90_38"],
                eval_tasks=["libero_90_79", "libero_90_31", "libero_90_82", "libero_90_38"],
            ),
            freeze_filter=_make_ogpo_freeze_filter(),
        ),
        # OGPO aligned with the reference implementation (OGPO_public, recipe
        # scripts/ogpo/square_image_paligemma.sh). Same learner and same dataclass
        # as `pi05_libero_online_ogpo_sft` -- only defaults differ, so the
        # isinstance dispatch in scripts/exp.py:74-85 is untouched. The two
        # configs exist side by side so the pre-alignment stack stays runnable and
        # the ten completed multitask arms remain valid comparators.
        # Rationale for each value: docs/changes/2026-08-20-ogpo-reference-alignment/.
        make_base_libero_config(
            name="pi05_libero_online_ogpo_ref",
            rl_config=OGPOSFTLearnerConfig(
                policy=PolicyTrainingConfig(update_interval=20, training_start_step=100),
                group_num_samples=8,  # NOT the reference's 32: 4x the actor fwd+bwd on a 3B expert
                clip_epsilon=0.1,  # NOT 0.01: measured ratio_max <= 0.9888 < 0.99 would clip every sample
                discount=0.995,  # NOT 0.99: 0.995 at our ~295-step successes == 0.99 at the reference's ~150
                advantage_combination="grpo_conservative",
                normalize_group_advantage=False,
                normalize_advantage_per_task=False,
                adv_clip_sym=None,
                critic_success_oversample=True,
                use_success_buffer=True,
                n_samples=8,  # best-of-N collection, inherited from AdvantageWeightedSFTLearner
                critic=CriticTrainingConfig(
                    num_qs=10,
                    num_vs=10,
                    reduction="mean",
                    batch_size=1024,
                    pre_training_steps=0,
                    use_bronet=True,
                    bronet_hidden_dim=1024,
                    # 95% TD / 5% MC: the only measured lever that moved q_value_mean
                    # off the -1/(1-gamma) fixed point (-189.5 vs -198.6).
                    td_weight_schedule=StepSchedule(
                        init_value=0.95, end_value=0.95, switch_step=999_999
                    ),
                ),
            ),
            freeze_filter=_make_ogpo_freeze_filter(),
        ),
    ]
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
