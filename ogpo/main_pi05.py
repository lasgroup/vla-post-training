"""OGPO with the pi05 action expert on LIBERO (`libero_pi05-*` envs) or MolmoSpaces (`molmo_pi05-*` envs).

Config comes from the standard ogpo.yaml + CLI overrides (same flags as
ogpo/main.py), plus:
  --agent.pi05_checkpoint   pi05 checkpoint dir (with params/ and assets/): pi05_libero | pi05_droid_jointpos
  --agent.pi05_robot        libero | droid; must match the env family (libero_pi05 -> libero, molmo_pi05 -> droid)
  --agent.pi05_tokenizer    paligemma_tokenizer.model
  --agent.pi05_action_horizon, --agent.pi05_lr, --agent.pi05_clip_grad_norm, --agent.pi05_weight_decay
  --buffer.bc_horizon       BC window length (defaults to pi05_action_horizon)
  --success_buffer.bc_batch_size  BC anchor batch (defaults to success_buffer.batch_size, which also sets the
                            success-buffer critic batch)
  --training.collection_mode=rounds  with --training.{num_rounds, rollouts_per_round, updates_per_round}:
                            vla-post-training's round schedule instead of one update per env step
  --env.name=a,b,c          several tasks of one family (comma list) = multi-task, as vla-post-training's
                            collect.tasks: one prompt row per task, rollouts_per_round episodes per task
                            per round, eval_episodes per task (needs collection_mode=rounds)
  --env.eval_name=...       eval tasks (comma list); default: the training tasks
  --training.task_balance=none|bc|bc_ppo|all  multi-task batch stratification (see pi05_online_runner._update_once)
  --env.parallel_envs=m     rounds only: m envs per task, collected / evaluated in lockstep with batched inference
  --env.vector_envs=true    rounds only: the T x m envs run in worker processes (AsyncVectorEnv) and every task's
                            waves run at once, one batched inference pass over all slots per replan
  --env.inference_batch=c   vector_envs only: rows per encode / policy call (BoN memory grows linearly in it);
                            with gpu_mode=multi the rows are sharded over the devices (each computes c / n_dev)
  --env.worker_egl_devices=auto|0,1,..  vector_envs only: GPUs the worker processes render on, round robin
                            (auto = CUDA_VISIBLE_DEVICES); default "" = every worker on the trainer's EGL GPU
  --buffer.storage_dir=<dir>  existing dir (node-local disk, e.g. /scratch): the replay buffer's images / next_images are
                            memmapped there instead of RAM (~0.6 of the ~0.62 MB per transition)
  --training.actor_update_interval=N  full actor+critic update every N updates, critic-only otherwise
  --training.zero_actor_state=true  multi-GPU only (ZeRO-1): shard the actor optimiser state + init copy across devices
  --training.restore_checkpoint=<checkpoint_latest.pkl> --training.eval_only=true  restore a saved agent and run one ODE + one
                            BoN eval on --env.eval_name (default: --env.name) at the checkpoint's step, no training
  --training.freeze_actor=true  critic-only updates forever (BoN-only baseline: pair with --agent.pi05_bon_sampler=ode
                            --agent.pi05_bon_agg=mean for vla-post-training-style BoN acting)
  --training.first_round_policy=ode  rounds only: round 0 collects with one unclipped ODE sample instead of BoN
  --agent.pi05_actor_objective=awr  rounds only: paper AWR (Peng et al. 2019) instead of OGPO; needs
                            --agent.use_value_fn=true --agent.v_training_mode=lambda_return; each round fits V for
                            --training.awr_value_steps steps, then runs updates_per_round AWR policy steps (ODE eval only)
  --agent.pi05_actor_objective=group_awr  OGPO with the PPO term replaced by min(exp(A/awr_beta), awr_weight_max) * CFM
                            on the same SDE samples and group advantages (--agent.awr_n_mc (t, x0) draws per sample)
  --agent.grad_accum_steps=K  split each update into K sequential micro-batches
  --agent.donate_agent      reuse the old agent's buffers on update (default true; 48 GB GPUs)

molmo_pi05 needs --env.vector_envs=true (the sim must stay out of the trainer process: it turns on jax x64) and
its prompts are read from the workers once they have loaded (`vector_prompts`); see envs/molmo_pi05_utils.py.
"""

import os
import random

import jax
import numpy as np

from envs import libero_pi05_utils, molmo_pi05_utils
from envs.libero_pi05_utils import get_prompt, is_libero_pi05_env, make_env
from envs.molmo_pi05_utils import is_molmo_pi05_env
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent
from ogpo.runners.pi05_critic_diag import run_bon_select_diag
from ogpo.runners.pi05_online_runner import (VectorTasks, load_checkpoint, run_pi05_eval_only, run_pi05_online_rl,
                                            run_pi05_round_rl, vector_prompts)
from ogpo.utils.resume import ResumeStore
from ogpo.utils.config_loader import config_to_agent_dict, load_config, parse_cli_args
from ogpo.utils.main_helper import make_vectorized_env, setup_experiment_logging
from ogpo.utils.sharding import create_mesh, shard_pi05_agent


def worker_egl_devices(spec):
    """`env.worker_egl_devices` -> GPU ids the vector-env workers render on (round robin), or None.

    "" (default): workers inherit the trainer's MUJOCO_EGL_DEVICE_ID, so every MuJoCo EGL context sits on
    one GPU. "auto": this run's GPUs, from CUDA_VISIBLE_DEVICES (the cluster scripts export it as the
    --device list). "0,2": an explicit comma list of GPU ids (EGL device index = nvidia-smi index).
    """
    spec = str(spec).strip()
    if not spec:
        return None
    if spec == "auto":
        spec = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not spec:
            raise ValueError("env.worker_egl_devices=auto needs CUDA_VISIBLE_DEVICES (this run's GPU list)")
    try:
        devices = [int(d) for d in spec.split(",") if d.strip()]
    except ValueError:
        raise ValueError(f"env.worker_egl_devices={spec!r}; expected auto | a comma list of GPU ids") from None
    if not devices:
        raise ValueError(f"env.worker_egl_devices={spec!r}; expected auto | a comma list of GPU ids")
    return devices


def main():
    algo_name, cli_overrides = parse_cli_args()
    assert algo_name == 'ogpo', "main_pi05 only runs OGPO"
    config = load_config(algo_name, cli_overrides)
    exp, env_cfg, train, buf, ev, intervals, sb = (
        config[k] for k in ("experiment", "env", "training", "buffer", "eval", "intervals", "success_buffer"))
    agent_config = config_to_agent_dict(config)
    for k in ('value_hidden_dims', 'actor_hidden_dims'):
        agent_config[k] = tuple(agent_config[k])

    env_name = env_cfg["name"]
    env_names = [n.strip() for n in str(env_name).split(",") if n.strip()]
    eval_names = [n.strip() for n in str(env_cfg.get("eval_name") or env_name).split(",") if n.strip()]
    for n in env_names + eval_names:
        assert is_libero_pi05_env(n) or is_molmo_pi05_env(n), f"expected a libero_pi05-* or molmo_pi05-* env, got {n}"
    molmo = [is_molmo_pi05_env(n) for n in env_names + eval_names]
    if any(molmo) and not all(molmo):
        raise ValueError(f"env.name / env.eval_name mix libero_pi05 and molmo_pi05 tasks: {env_names} | {eval_names}")
    molmo = all(molmo)
    robot = (molmo_pi05_utils if molmo else libero_pi05_utils).ROBOT
    if agent_config["pi05_robot"] != robot:
        raise ValueError(f"agent.pi05_robot={agent_config['pi05_robot']!r} but {'molmo' if molmo else 'libero'}_pi05 "
                         f"envs need pi05_robot={robot!r} (and its checkpoint)")
    multitask = len(env_names) > 1 or eval_names != env_names
    rounds = train.get("collection_mode", "online") == "rounds"
    parallel_envs = env_cfg["parallel_envs"]
    if parallel_envs < 1:
        raise ValueError(f"env.parallel_envs={parallel_envs}; must be >= 1")
    if parallel_envs > 1 and not rounds:
        raise ValueError(f"env.parallel_envs={parallel_envs} needs --training.collection_mode=rounds "
                         f"(the per-step schedule runs one env)")
    vector_envs, inference_batch = env_cfg["vector_envs"], env_cfg["inference_batch"]
    if vector_envs and not rounds:
        raise ValueError("env.vector_envs=true needs --training.collection_mode=rounds (the per-step schedule runs one env)")
    if molmo and not vector_envs:
        raise ValueError("molmo_pi05 envs need --env.vector_envs=true (in-process, the sim turns on jax x64 in the trainer)")
    if inference_batch < 1:
        raise ValueError(f"env.inference_batch={inference_batch}; must be >= 1")
    egl_devices = worker_egl_devices(env_cfg.get("worker_egl_devices", ""))
    if egl_devices and not vector_envs:
        raise ValueError("env.worker_egl_devices is only used with env.vector_envs=true (worker processes)")
    if train["actor_update_interval"] < 1:
        raise ValueError(f"training.actor_update_interval={train['actor_update_interval']}; must be >= 1")
    if train["zero_actor_state"] and train.get("gpu_mode", "single") == "single":
        # Fail before building envs / loading pi05 (the mesh-is-None check below also covers 1 visible GPU).
        raise ValueError("training.zero_actor_state=true needs a multi-device mesh (training.gpu_mode=multi | reserve:N)")
    if train["first_round_policy"] not in ("bon", "ode"):
        raise ValueError(f"training.first_round_policy={train['first_round_policy']!r}; expected bon | ode")
    if train["first_round_policy"] == "ode" and not rounds:
        raise ValueError("training.first_round_policy=ode needs --training.collection_mode=rounds")
    if train["freeze_actor"]:
        print("[pi05] freeze_actor: critic-only updates, training.actor_update_interval ignored")
    awr = agent_config["pi05_actor_objective"] == "awr"
    if awr:
        # AWR owns the round schedule (V fit, then policy steps) and always collects with the online actor.
        if not rounds:
            raise ValueError("agent.pi05_actor_objective=awr needs --training.collection_mode=rounds")
        if train["first_round_policy"] != "bon":
            raise ValueError("training.first_round_policy is not used under awr (every round samples the online actor); "
                             "leave it at bon")
        if train["freeze_actor"] or train["actor_update_interval"] != 1:
            raise ValueError("agent.pi05_actor_objective=awr needs training.freeze_actor=false and "
                             "training.actor_update_interval=1")
        if train["awr_value_steps"] < 0:
            raise ValueError(f"training.awr_value_steps={train['awr_value_steps']}; must be >= 0")
        if train["awr_value_refreshes"] < 1:
            raise ValueError(f"training.awr_value_refreshes={train['awr_value_refreshes']}; must be >= 1")
    seed = exp["seed"]
    random.seed(seed)
    np.random.seed(seed)

    discount = agent_config["discount"]
    agent_config["q_min"], agent_config["q_max"] = -1.0 / (1.0 - discount), 0.0
    agent_config.setdefault("adv_clip_min", None)

    post_success_steps = env_cfg.get("post_success_steps", 0)
    # parallel_envs > 1: each task's env slot is a list of m envs (same make_env seed; resets are seeded per episode).
    def _envs(n, env_seed):
        es = [make_env(n, seed=env_seed, post_success_steps=post_success_steps) for _ in range(parallel_envs)]
        return es if parallel_envs > 1 else es[0]

    def _prompt(e):
        return (e[0] if isinstance(e, list) else e).prompt

    # Eval tasks share the prompt table: train tasks first (same rows), then any extra eval-only tasks.
    prompt_rows = list(env_names) + [n for n in eval_names if n not in env_names]
    vector_tasks = []
    if vector_envs:
        # Built before the agent and mesh; workers are spawned (JAX threads / EGL state are not fork-safe).
        # libero: prompts come from the task suite, so the trainer builds no sim (gymnasium's slot-0 dummy aside).
        # molmo: the prompt needs the loaded task, so it is read from the workers once the vector envs exist.
        prompts = {} if molmo else {n: get_prompt(n) for n in prompt_rows}

        def _vector(names, env_seed):
            vec = make_vectorized_env([n for n in names for _ in range(parallel_envs)], len(names) * parallel_envs,
                                      seed=env_seed, post_success_steps=post_success_steps, context="spawn",
                                      env_kwargs={'vector': True}, egl_devices=egl_devices)
            vector_tasks.append(VectorTasks(vec, list(names), [prompt_rows.index(n) for n in names],
                                            parallel_envs, inference_batch))
            return vector_tasks[-1]
    else:
        train_envs = [(n, _envs(n, i)) for i, n in enumerate(env_names)]
        eval_envs = [(n, _envs(n, 42 + prompt_rows.index(n))) for n in eval_names]
        prompts = {n: _prompt(e) for n, e in train_envs + eval_envs}

    accum = agent_config.setdefault("grad_accum_steps", 1)
    if accum < 1:
        raise ValueError(f"agent.grad_accum_steps={accum}; must be >= 1")
    agent_config.setdefault("donate_agent", True)  # update steps reuse the old agent's buffers (48 GB GPUs)
    if vector_envs:
        # Construction seeds are inert: every reset passes a seed.
        env = _vector(env_names, 0)
        eval_env = env if eval_names == env_names else _vector(eval_names, 42)
        if molmo:
            for vt in vector_tasks:  # train and eval vector envs may share tasks: their prompts must agree
                for n, p in vector_prompts(vt).items():
                    if prompts.setdefault(n, p) != p:
                        raise RuntimeError(f"{n}: train and eval workers disagree on the prompt ({prompts[n]!r} vs {p!r})")
        agent = OGPOPi05Agent.create(seed, [prompts[n] for n in prompt_rows], agent_config)
        print(f"[pi05] vector envs: train {env_names} | eval {eval_names} | {parallel_envs} slot(s) per task, "
              f"{sum(vt.num_slots for vt in vector_tasks)} worker(s), inference_batch={inference_batch}, "
              f"worker EGL GPUs={egl_devices or 'inherited'}")
    elif multitask or parallel_envs > 1:
        if not rounds:
            raise ValueError("multi-task needs --training.collection_mode=rounds")
        agent = OGPOPi05Agent.create(seed, [prompts[n] for n in prompt_rows], agent_config)
        env = [(n, e, prompt_rows.index(n)) for n, e in train_envs]        # (name, env(s), prompt-table row)
        eval_env = [(n, e, prompt_rows.index(n)) for n, e in eval_envs]
        print(f"[pi05] train {env_names} | eval {eval_names} | {parallel_envs} env(s) per task")
    else:
        env, eval_env = train_envs[0][1], eval_envs[0][1]
        agent = OGPOPi05Agent.create(seed, env.prompt, agent_config)
    ckpt_step = 0
    if train.get("restore_checkpoint"):  # before sharding: the agent is still the freshly built one
        agent, ckpt_step = load_checkpoint(agent, train["restore_checkpoint"], keep_base_actor=train.get("eval_base_actor", False))
    elif train.get("eval_only"):
        raise ValueError("training.eval_only=true needs training.restore_checkpoint")
    # Data parallel: agent replicated on every device, update batches split on axis 0.
    mesh = create_mesh(train.get("gpu_mode", "single"))
    n_dev = len(mesh.devices) if mesh is not None else 1
    # Every micro-batch must split evenly over the devices.
    for k, v in (("agent.batch_size", agent_config["batch_size"]),
                 ("agent.ppo_batch_size", agent_config["ppo_batch_size"]),
                 ("success_buffer.batch_size", sb["batch_size"]),
                 ("success_buffer.bc_batch_size", sb.get("bc_batch_size") or sb["batch_size"])):
        if v % (n_dev * accum):
            raise ValueError(f"{k}={v} must be divisible by {n_dev} devices x {accum} grad_accum_steps")
    if train["zero_actor_state"] and mesh is None:
        raise ValueError("training.zero_actor_state=true needs a multi-device mesh (training.gpu_mode=multi | reserve:N)")
    if mesh is not None:
        agent = shard_pi05_agent(agent, mesh, zero_actor_state=train["zero_actor_state"])
        per_dev = sum(x.addressable_shards[0].data.nbytes for x in jax.tree.leaves(agent)) / 2 ** 30
        print(f"[pi05] data parallel over {n_dev} devices ({train['gpu_mode']}); "
              f"zero_actor_state={train['zero_actor_state']}, agent state {per_dev:.2f} GiB per device")

    snapshot_dir = train.get("resume_snapshot_dir") or ""
    if snapshot_dir:
        if not rounds:
            raise ValueError("training.resume_snapshot_dir needs --training.collection_mode=rounds")
        if train.get("restore_checkpoint") or train.get("eval_only") or train.get("resume_s3") or os.environ.get("OGPO_RESUME_S3"):
            raise ValueError("training.resume_snapshot_dir is exclusive with restore_checkpoint, eval_only and resume_s3")
    elif buf.get("reuse_storage"):
        raise ValueError("buffer.reuse_storage needs training.resume_snapshot_dir")
    resume = None
    resume_s3 = train.get("resume_s3") or os.environ.get("OGPO_RESUME_S3")
    if resume_s3:
        if not rounds:
            raise ValueError("training.resume_s3 needs --training.collection_mode=rounds")
        resume = ResumeStore(train.get("resume_dir") or os.environ.get("OGPO_RESUME_DIR") or "/tmp/ogpo_resume", resume_s3,
                             train["updates_per_round"], train.get("resume_ckpt_interval", 250),
                             rebase_rounds=train.get("resume_rebase_rounds", False))
    save_dir, logger, _ = setup_experiment_logging(
        seed=seed, project=exp["project"], run_group=exp["run_group"], wandb_name=exp["wandb_name"],
        save_dir=exp["save_dir"], env_name=env_name, log_enabled=exp["log"],
        prefixes=["eval", "eval_sde", "env", "online_agent"] + (["awr"] if awr else []), config=config,
        append_csv=(resume is not None and resume.manifest() is not None) or bool(snapshot_dir),
    )

    finished = False
    if train.get("eval_only"):
        try:
            modes = tuple(m.strip() for m in str(train.get("eval_modes", "ode,bon")).split(","))
            run_pi05_eval_only(agent, eval_env, logger, step=ckpt_step, eval_episodes=ev["eval_episodes"], mesh=mesh,
                               modes=tuple(m for m in modes if not m.startswith("sel_")))
            for m in (m for m in modes if m.startswith("sel_")):  # critic diagnostic: BoN with the pick swapped
                if not isinstance(eval_env, VectorTasks) or save_dir is None:
                    raise ValueError(f"eval mode {m} needs env.vector_envs=true and experiment.log=true")
                rule = m[len("sel_"):]
                run_bon_select_diag(agent, eval_env, ckpt_step, ev["eval_episodes"], rule,
                                    os.path.join(save_dir, f"diag_{rule}.pkl"), mesh=mesh)
            finished = True
        finally:
            for vt in vector_tasks:
                vt.vec.close(terminate=not finished)
        logger.close()
        return

    common = dict(
        batch_size=agent_config["batch_size"],
        success_buffer_batch_size=sb["batch_size"],
        bc_batch_size=sb.get("bc_batch_size") or sb["batch_size"],
        horizon_length=agent_config["horizon_length"],
        bc_horizon=buf.get("bc_horizon") or agent_config["pi05_action_horizon"],
        discount=discount,
        buffer_size=buf["size"],
        buffer_storage_dir=buf.get("storage_dir") or None,
        eval_interval=intervals["eval_interval"],
        eval_episodes=ev["eval_episodes"],
        log_interval=intervals["log_interval"],
        seed=seed,
        save_dir=save_dir if exp["log"] else None,
        mesh=mesh,
        actor_update_interval=train["actor_update_interval"],
        freeze_actor=train["freeze_actor"],
    )
    try:
        if train.get("collection_mode", "online") == "rounds":
            # vla-post-training schedule: collect full episodes with a frozen policy, then a block of updates.
            run_pi05_round_rl(
                agent, env, eval_env, agent_config, logger,
                num_rounds=train["num_rounds"],
                rollouts_per_round=train["rollouts_per_round"],
                updates_per_round=train["updates_per_round"],
                task_balance=train.get("task_balance", "none"),
                first_round_policy=train["first_round_policy"],
                resume=resume,
                resume_snapshot_dir=snapshot_dir or None,
                buffer_reuse_storage=bool(buf.get("reuse_storage")),
                awr_value_steps=train["awr_value_steps"],
                awr_value_refreshes=train["awr_value_refreshes"],
                **common,
            )
        else:
            run_pi05_online_rl(
                agent, env, eval_env, agent_config, logger,
                online_steps=train["online_steps"],
                start_training=buf["start_training"],
                **common,
            )
        finished = True
    finally:
        # A plain close() joins workers without a timeout; after an error (possibly a hung worker) kill them.
        for vt in vector_tasks:
            vt.vec.close(terminate=not finished)
    logger.close()


if __name__ == '__main__':
    main()
