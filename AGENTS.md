# AGENTS.md — vla-post-training: setup, replication, and operations guide

*Written 2026-08-21 for anyone (human or agent) picking up this repo — specifically
the `shashwat/stability-study` branch and its experimental record. Everything below
was executed repeatedly during the study; the gotchas are all real ones we hit.*

## 1. What this repo is

Online RL post-training of flow-matching VLA policies (π0.5 = PaliGemma-2B +
300M flow action expert) on LIBERO manipulation tasks. Several algorithm
families coexist (see §7); the `shashwat/stability-study` branch contains
**OGPO** (PPO over the SDE denoising chain) plus the stability machinery that
makes it work: critic digestion burst, BC warmstart, PG ramp, advantage
normalizer/clip, conservative-GRPO advantage.

**Headline result**: the "locked recipe" (§5) reaches 100% final eval on
libero_90_44 across 6/6 seeds with zero terminal collapses, vs. routine
100→0 crashes for unstabilized PPO. Full story: `reports/stability_study_summary.md`,
`reports/SWISS_VLA_DUMP.md` (all experiments), `reports/NON_OGPO_EXPERIMENTS_DUMP.md`
(other families), `reports/stability_study_hypotheses.md` (pre-registered H1–H14).

## 2. Repo layout (what matters)

```
src/rl/ogpo/            # OGPO learner + actor update (the study's code)
  ogpo_learner.py       #   burst, warmstart gate, ramp, UTD, bc_mask plumbing
  update_actor.py       #   advantage construction (grpo_conservative etc.), PPO surrogate
src/rl/advantage_weighted_sft/  # AWR base classes the OGPO learner inherits from
src/training/config.py  # ALL config dataclasses — every rl.* flag documented inline
scripts/exp.py          # the training entrypoint (train loop, collection, eval)
scripts/stability_study.sh      # THE experiment launcher (env-var driven, §4)
scripts/ws_bcbb_pipeline.sh     # two-stage unfrozen-BC-backbone pipeline (arms iii/iv)
tests/ogpo/             # equivalence suite — run after ANY learner change
reports/                # experiment record: summaries, dumps, plots, metrics_archive/
openpi/                 # submodule: lasgroup/openpi @ 11f08089
molmospaces/            # submodule: lasgroup/molmospaces @ 2a828d54 (unused for LIBERO)
```

## 3. Setup from scratch (fresh Ubuntu 22.04 GPU node, ~40 min)

```bash
# 1. System deps (cmake for egl-probe build; GL/OSMesa for MuJoCo headless render)
sudo apt-get install -y cmake libglvnd0 libegl1 libgl1 libopengl0 libgles2 \
                        libglib2.0-0 libosmesa6 libosmesa6-dev

# 2. uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Repo + pinned submodules (private; needs a GitHub PAT with lasgroup access)
git clone https://github.com/lasgroup/vla-post-training.git ~/vla-post-training
cd ~/vla-post-training && git checkout shashwat/stability-study
git clone https://github.com/lasgroup/openpi.git openpi
(cd openpi && git checkout 11f08089af6e507c4e76e193ba42a3ded217113e)
git clone https://github.com/lasgroup/molmospaces.git molmospaces   # optional for LIBERO
(cd molmospaces && git checkout 2a828d549307f5d2e1d5384dc0662b3a1666954a)

# 4. Python env
uv sync        # fails without cmake (egl-probe); rerun after apt step if needed

# 5. LIBERO first-run config — REQUIRED, else libero import blocks on an
#    interactive Y/N prompt and headless runs crash
mkdir -p run_store/libero
V=$PWD/.venv/lib/python3.12/site-packages/libero/libero
cat > run_store/libero/config.yaml <<EOF
benchmark_root: $V
bddl_files: $V/./bddl_files
init_states: $V/./init_files
datasets: $V/../datasets
assets: $V/./assets
EOF

# 6. LIBERO scene assets (~408MB). Auto-download on first env creation — but
#    HuggingFace RATE-LIMITS datacenter IPs. If downloads fail, copy the cache
#    from any working machine: tar ~/.cache/libero/assets across, then:
#    ln -sfn ~/.cache/libero/assets $V/assets

# 7. Model weights: gs://openpi-assets/checkpoints/pi05_libero (auto-downloaded
#    into run_store/cache/openpi on first run; needs GCS egress).

# 8. Sanity check (CPU, ~6 min) — must be 15/15:
JAX_PLATFORMS=cpu uv run pytest tests/ogpo/ -x -q
```

**Rendering**: default is EGL (GPU). If the node lacks `libEGL_nvidia.so`
(check `ldconfig -p | grep EGL_nvidia`), launch with
`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` (CPU render, ~2× slower collection,
identical training math). The launcher handles EGL vendor-JSON discovery.

## 4. Running experiments — scripts/stability_study.sh

One arm per invocation, config via env vars. All flags default OFF = the
original (viii)_r2 recipe on libero_90_44.

```bash
cd ~/vla-post-training
env GPU=0 ARM=myrun SEED=0 \
    PG_START=20000 PG_RAMP=5000 BURST=1000 CONS=1 NORM=1 CLIP_SYM=4.0 INIT_ROLLOUTS=40 \
    nohup bash scripts/stability_study.sh > ~/stab_myrun.log 2>&1 &
```

| Var | Effect (flag it sets) |
|---|---|
| GPU / ARM / SEED | device index / run name (exp_name=stab_$ARM) / seed |
| TASK | collect+eval task (default libero_90_44) |
| PG_START | `rl.pg_start_step` — BC-only warmstart until this step |
| PG_RAMP | `rl.pg_ramp_steps` — linear PG fade-in after handoff |
| BURST | `rl.post_collection_critic_steps` — critic-only digestion after each collection |
| CONS | `rl.advantage_combination grpo_conservative` (per-head Q−mean_G(Q), sign-gated; V never enters) |
| NORM | `rl.normalize_group_advantage` (EMA-quantile scale, floor 1) |
| CLIP_SYM | `rl.adv_clip_sym` (symmetric clip post-norm) |
| INIT_ROLLOUTS | `collect.num_initial_rollouts` — extra episodes at step 0 (spark insurance) |
| BC_POST | `rl.bc_coeff_post_warmstart` — anchor strength once PG is live (0.0 = pure PPO) |
| NOISE | `rl.noise_level` — actor group-sampling SDE temperature (collection/eval are deterministic-ODE) |
| HORIZON | `collect.max_episode_steps` (default 400) |
| UTD / BURST_MC / QS / FSFT / EMA / ACCUM / FC_INT+FC_ROLLOUTS | critic UTD, MC-target burst, ensemble size, masked-BC anchor, actor EMA, grad accum, streaming collection |

Monitoring: metrics land in
`run_store/checkpoints/stability_study/pi05_libero_online_ogpo_sft/stab_$ARM/metrics.jsonl`
(one JSON per log step; eval rows have `eval/success_rate`). Progress:
`tr '\r' '\n' < ~/stab_$ARM.log | grep -a "Progress on" | tail -1`.

## 5. THE LOCKED RECIPE (replicate this first)

```bash
env GPU=0 ARM=lock_s0 SEED=0 TASK=libero_90_44 \
    PG_START=20000 PG_RAMP=5000 BURST=1000 CONS=1 NORM=1 CLIP_SYM=4.0 INIT_ROLLOUTS=40 \
    nohup bash scripts/stability_study.sh > ~/stab_lock_s0.log 2>&1 &
```
Expected (validated on 6 seeds): warmstart climbs to 60–100% by 10–20k;
PG ramps in 20–25k; possibly ONE transient deep eval mid-run (recovers within
one eval); final eval (100k) = 100%. Runtime ~24h on one H200 (GPU render),
~30h on B200 with osmesa. Watch for: `adv_scale` floored at 1.0 early then
lifting; end-of-burst `burst/q_loss` low (≤~30) after each collection;
`alive_fraction`≈0.7 and `clipfrac_upper`≡0 once PG is live.

**What each piece is for** (evidence in reports/): burst = recalibrate the
critic on each collection's novel data BEFORE the actor consumes its rankings
(the boom-bust root cause); warmstart = move the fragile climb out of PPO;
ramp = a mature critic's full-size advantages would hit the BC-tuned policy in
one step (measured grad 0.05→1.8) — fade them in; norm+clip = pin advantage
scale ≈ effective LR against critic-spread drift; cons-GRPO = discard samples
the Q-heads disagree on (helps only with a calibrated critic — harmful cold);
init-rollouts = P(zero successes in 20 eps @5% base SR) ≈ 36% — the "slow
seed" lottery; 40 extra episodes kills it.

**Known boundary**: the recipe transforms low-base-SR tasks (2%→100%) but
does NOT lift mid-base-SR tasks (libero_90_38, base ~45%: 14 arms, 8
interventions, all plateau at ≈baseline). Diagnosis: one-bit terminal reward
carries no action-attributable signal at the margin; critic ranking quality
(q_mc_corr) pins at ~0.55. Ruled out: anchor strength/removal, cons on/off,
ensemble 2→5, init rollouts, horizon 400→600, noise 0.02→0.1→0.3 (higher noise
is HARMFUL — flat log-prob landscape shrinks gradients and drifts). Next lever
if needed: dense/subtask rewards.

## 6. Cluster operations (FAR/SkyPilot specifics)

- Nodes: `sky launch -c <name> --infra kubernetes/<ll-sna|ll-sea|...> --gpus H200:8
  --image-id docker:241533154612.dkr.ecr.us-east-1.amazonaws.com/skypilot-gpu:latest -y`
  (admin policy requires --infra and an approved ECR image).
- Auth decays: `mwinit -o` (interactive), then `cp ~/.midway/cookie ~/.sky/cookies.txt`
  and `sky api login -e https://skypilot-api.gamma.compute.far.ftr.amazon.dev`.
- **Pods are ephemeral**: kubelet evicts on ephemeral-storage pressure and the
  disk is WIPED (lost 2 pods this way). Prune finished runs' `run_store` dirs
  (~50GB/run) after harvesting metrics.jsonl; final checkpoints ~12GB each.
  A dead pod blocks relaunch with "pod already exists" → `sky down <name>` then launch.
- Runs checkpoint at every collection boundary (10k); a killed run loses ≤10k steps.
- Kill runs by PID (`pgrep -af "stab_<ARM>"`), never broad pkill.

## 7. Other algorithm families in this repo (context)

fSFT (success-filtered BC — the universal baseline; Ralf's current recipe is
BoN+fSFT), AWR (exp-advantage-weighted SFT; Marco's line — source of the
normalizer we ported), Best-of-N (Q ranks N=32 proposals at inference; critic
never touches gradients), MPO/flow-GRPO/DSRL/residual-RL/PA-RL (abandoned;
useful ideas absorbed — see reports/NON_OGPO_EXPERIMENTS_DUMP.md). The critic
everywhere: 2Q/2V BroNet MLPs on mean-pooled frozen-PaliGemma prefix (2048d)
+ state (+ flattened 10×32 action chunk for Q), TD on r=−1/step, γ=0.995.

## 8. Replicating the full study / analyzing the record

`reports/metrics_archive/extracted/<node>/stab_*/metrics.jsonl` holds the raw
per-step telemetry of 36 runs (every eval, collection round, actor/critic
health metric discussed in the reports). Row types: collection SR rows,
`eval/*` rows (32-ep EMA evals every 10k), actor rows (every 25 steps:
approx_kl, alive_fraction, advantage_*, adv_scale, cons_zero_frac, ...),
critic rows (q_loss, q_grad_norm, q_mc_corr, buffer sizes), `burst/*` rows
(end-of-burst critic state). Plots in reports/*.png were generated from these.
The arm-name → config mapping is in reports/SWISS_VLA_DUMP.md §§C–F tables.
```
