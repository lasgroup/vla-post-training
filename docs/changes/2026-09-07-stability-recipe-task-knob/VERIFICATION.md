# Verification

Independent agent: Opus, xhigh reasoning effort, fresh context (change spec +
diff only). It also received, mid-run, the probe's multi-pass addition and the
LIBERO seeding block. Report verbatim, then outcomes.

---

## Verifier report (verbatim)

## Execution disclaimer

Login node: no GPU, no `/data` mount, no LIBERO/MuJoCo runtime. Everything below is either (a) a shell render with `uv` replaced by an argv-dumping shim or `DRY=1`, (b) the real `src/training/config.py:cli()` fed the rendered argv on CPU, (c) CPU pytest, or (d) `sacct` / existing job logs. **No job was submitted, no recipe was run without `DRY=1`/a shim, no file under `scripts/` or `src/` was edited.** I appended tests to my own verifier file `tests/ogpo/test_candidate_q_spread_verifier.py` (explicitly authorised by the coordinator) and nothing else.

Both target files moved under me mid-verification (wrapper at 21:03 and 21:13, recipe at 21:13). Everything reported below is against the state at **21:13** (`stability_study.sh` md5-stable since; wrapper 90 lines, `--time=05:00:00`, `QSPREAD_PASSES=4`).

## Verified correct

| Claim | Evidence |
|---|---|
| `bash -n` clean | both files, re-run after the 21:13 seeding-block edit |
| Diff is exactly what `DIFF.md` describes | `diff` of `stability_study.before.sh` vs current: 6 hunks — header comment, `TASK=` (`:52`), LIBERO seeding (`:150-152`), banner `task=$TASK` (`:169`), `RUN=`/`DRY` triple (`:174-177`), `--collect.tasks/--collect.eval_tasks "$TASK"` (`:193-194`), `"$@"` (`:228`). Nothing else. |
| Every existing caller renders **byte-identical argv** | 10 env combos, before-copy vs current file, both through the same `uv` shim: all IDENTICAL (details below). Only stdout diff is the banner gaining `task=libero_90_44`. |
| Empty `"$@"` adds no argument | `set --; f(){ printf "argc=%d\n" $#; }; f a b "$@"` → `argc=2`. Also visible in the renders: 95 tokens before and after. |
| `[ ... ] && RUN=(...)` survives `set -euo pipefail` in both branches | executed both ways, exit 0 both times (bash exempts a non-final `&&` member) |
| tyro takes the **last** occurrence | real `_config.cli()` on the rendered argv: `--rl.n_samples 8 … --rl.n_samples 16` → `8`→`16`; `--collect.eval_tasks` repeated → **replaces** (`['libero_90_31']`), does not append; `--no-backbone_lora … --backbone_lora` → `gemma_2b_lora`. openpi's `cli()` is `tyro.extras.overridable_config_cli` (`openpi/src/openpi/training/config.py:980-981`); this repo's is `src/training/config.py:881`. |
| Wrapper delivers the intended config | real `cli()` on the wrapper's own rendered argv: `OGPOSFTLearnerConfig`, `n_samples=8`, `critic.reduction='min'`, `num_qs/num_vs=2`, `num_tasks=None`, `inference_start_step=1`, `collect.tasks==eval_tasks==['libero_90_38']`, `env_num=8`, `episode_steps_multiplier=1`, `resume=True/overwrite=False`, `fsdp_devices=1`, `exp_name='stab_NCB_libero_90_38'`, `checkpoint_dir=<STORE_ROOT>/checkpoints/stability_study/pi05_libero_online_ogpo_sft/stab_NCB_libero_90_38`, `paligemma_variant='gemma_2b'`. `--resume` ×1, `--overwrite` ×0, `--rl.n_samples` ×1 (last two tokens of the line). |
| **What the probe will actually score**: M=8 per call × 4 passes = **32** candidates, scored by **min over 2 Q heads** | `AdvantageWeightedSFTLearnerConfig.n_samples: int = 1` (`src/training/config.py:197`), `CriticTrainingConfig.reduction="min"`, `num_qs=num_vs=2` (`:128,:131-132`); `pi05_libero_online_ogpo_sft` overrides none of them (`:805-812`). Without the new passthrough the probe would raise (`probe_candidate_q_spread.py:349-355`). |
| 82 chunks / 400 env steps | `get_max_steps_libero("libero_90")=400` (`src/envs/libero.py:126-133`), `replan_steps=5` (`src/training/config.py:404`), `400//5+2=82` (`probe_candidate_q_spread.py:467-468`) |
| Config tree matches the run that saved the checkpoint | The sibling recipe that produced it (`vla_single_task/scripts/stability_study.sh`, commit `c3dbcac`) differs from this one only in: `--collect.max_episode_steps 400` (this repo's default is also `400`, `src/training/config.py:421`, and it feeds value-bound computation, not the env TimeLimit, which comes from `get_max_steps_libero`), `--rl.noise_level "$NOISE"` with `NOISE=0.02` (identical), `RESUME_FLAG` vs `CKPT_MODE_FLAG` (same rendered flag), and the absent `--no-backbone_lora` (defaults False either way — `cli()` confirms `gemma_2b`). No tree-shaping difference. |
| Restoring a pre-hardening checkpoint of this campaign through **this** code works | `/home/pchellap/logs/lang_grounding/lang_ground_s3_10274728.out`: `Resume resolved to step 80000 (orbax steps=[80000], per-step manifests=[])` (`filtered_sft_learner.py:456`), `Restored replay buffer … (transitions=59920)`, `Restored training checkpoint … at committed step 80000` (`filtered_sft_learner.py:325`), then the expected degrade: `Resume manifest at step 80000 has no 'extra' block … starting the success buffer empty and _adv_scale at min_scale` (`ogpo_learner.py:235`). Ran on **1 GPU (L40S) with `--fsdp_devices 1`** — so the FSDP-mismatch trap in the mt4 wrapper's header does not apply here. Code paths: `runtime_state.py:98-109` (no manifests → `resume_state.json` pointer, else loud `FileNotFoundError`), `ogpo_learner.py:225-237`. |
| Nothing in `src/` has changed since that restore | newest `src/**/*.py` mtime is `2026-08-30 00:52`; the stage-3 restore was `2026-08-31 16:13`. The two post-08-31 records (`2026-09-05-rollout-gif-probe`, `2026-09-07-candidate-q-spread-rollouts`) both state "no edits to existing code" and the mtimes agree. |
| LIBERO seeding block is a byte-identical copy | md5 of the 3 code lines matches `ogpo_multitask_4task.sh:253-255` exactly. |
| Seeding block is a no-op for every pre-existing caller | `run_store/libero/config.yaml` exists (Aug 11, 616 B) and `ws_bcbb_pipeline.sh`, `stability_wave1_bc0.sh`, `stage4_trained_instructions.sbatch` all use the default local `STORE_ROOT`; the 10-combo render (with `config.yaml` present) is still IDENTICAL after the block was added. |
| The seeded config is scientifically the same one the run trained with | `answer="n"` → `get_default_path_dict()` (`libero/libero/__init__.py:14-37, 98-128`): all five paths package-relative. `~/.libero/config.yaml`, `run_store/libero/config.yaml` and the sibling checkout's are byte-identical and all point at **this** repo's venv — the sibling's `run_store` is a **symlink to this repo's** (`vla_single_task/run_store -> ../vla-post-training/run_store`). Both venvs pin `hf-libero 0.1.3`, same sdist hash. So `bddl_files`/`init_files`/`assets` are the same files the training run used. |
| `\|\| true` is a sanctioned copy, not a new deviation | identical to the pre-existing `ogpo_multitask_4task.sh:254` (and the sibling's `:140-141`); failure mode is that the real import then dies on the same `EOFError` a few seconds later. |
| Tests | `pytest tests/ogpo/test_candidate_q_spread.py tests/ogpo/test_candidate_q_spread_verifier.py -q` → **42 passed** (16 + 26, incl. 5 new multi-pass tests). |

## Findings

**1. [CONFIRMED] — `--mem=200G` is justified by a misattributed and censored measurement. Severity: low (the conclusion is right, the citation is wrong).**
Header (`probe_candidate_q_spread_stab.sbatch:34-36`) says "the stage-3 probe … peaked at 128-134 GB (jobs 10274728/10274386)".
```
$ sacct -j 10274728,10274386,10210760 --format=JobID,JobName,MaxRSS,ReqMem,Elapsed -n
10274728 lang_ground_s3   MaxRSS 128.0 GiB (=137.4 GB)  ReqMem 128G   100.0% of request
10274386 lg_s2            MaxRSS  43.1 GiB (= 46.3 GB)  ReqMem  64G    67.3% of request
10210760 stab_ncb_…_38    MaxRSS 138.6 GiB (=148.9 GB)  ReqMem 150G    92.4% of request  23:26:45
```
`10274386` is **`lg_s2`** (stage 2), not a stage-3 probe, and used 43 GiB. "128-134 GB" is one number in two units. And `10274728`'s MaxRSS is *exactly* its 128 G cap — a censored measurement, not headroom. The defensible number is the third one the header already cites: the checkpoint's **own training run** peaked at 138.6 GiB while holding the same 100k-step buffer plus optimizer state, so 200 G leaves ~44%. Fix the citation, not the request.

**2. [CONFIRMED] — `DRY=1` is no longer side-effect-free. Severity: low.**
The seeding block (`:150-152`) sits *before* the `RUN=`/`DRY` swap (`:174-175`). Demonstrated with a store lacking `libero/config.yaml`: `DRY=1` printed the command **and** invoked `uv run python -c "import libero.libero"` (shim-captured). Against a real `uv` that resolves the venv, imports LIBERO/torch/MuJoCo (tens of seconds) and **writes `$LIBERO_CONFIG_PATH/config.yaml`**. The header line "DRY - 1 => print the final command instead of running it" (`:38`) is therefore not strictly true. `ogpo_multitask_4task.sh` has the same ordering, so this is consistent with precedent — but the header should say "prints instead of running the *training* command; still seeds LIBERO's config if missing".

**3. [CONFIRMED] — the change record is stale relative to the files. Severity: low (docs).**
`DIFF.md` still says "preempt, 1 GPU, 200G, **3 h**" (file: `--time=05:00:00`), "the **three** `QSPREAD_*` knobs" (now four: `QSPREAD_PASSES`), and `QSPREAD_EPISODES` default 8 (now 16). `BLAST-RADIUS.md` still says "Verified by a before/after render: the pre-edit copy run with `ENTRY=/bin/echo`" — that render was never run that way here (see the sound-comparison note below). Neither document states anywhere that **M is 32, not 8** — the single most important number for reading the probe's output.

**4. [CONFIRMED] — the probe now writes into another run's store root. Severity: low, but the header should say it.**
For the wrapper, `LIBERO_CONFIG_PATH=$STORE_ROOT/libero = /data/…/vla_single_task/libero`, which is empty (proved below). The seeding block will create `config.yaml` there. The header's "READ-ONLY wrt the checkpoint" (`:20`) is literally true and the write is harmless, but "read-only" now has an asterisk.

**5. [CONFIRMED] — why job 10274728 did not prompt: the file never existed, so stage-3 must have inherited `LIBERO_CONFIG_PATH`. Severity: informational; nothing about stage 3 is invalidated.**
`/home/pchellap/logs/lang_grounding/d5_10274729.out` (2026-08-31, listing the sibling store root):
```
drwxrws---+ 3 … Aug 24 15:11 checkpoints
drwxrws---+ 2 … Aug 31 16:02 libero
=== …/vla_single_task/libero ===
total 66
drwxrws---+ 2 … Aug 31 16:02 .        (empty — no config.yaml)
```
Everything except `checkpoints/` was created on Aug 31 16:02 — by stage-3's own `mkdir -p`. So the training run never used this root for anything but checkpoints (it ran with the local `STORE_ROOT` and only `CKPT_BASE_DIR` redirected — the sibling recipe allows that, `${CKPT_BASE_DIR:-…}`, which this repo's copy removed). `libero.libero` reads exactly `$LIBERO_CONFIG_PATH/config.yaml` and nothing else (`__init__.py:4-8, 98`), so with an empty directory the *only* way stage-3 avoided the prompt is that `LIBERO_CONFIG_PATH` was already exported in its submitting shell and carried by `sbatch --export=ALL` (the recipe's `${LIBERO_CONFIG_PATH:-…}` defers to it). Same `--export=ALL` reconstruction gotcha already in the maintainer's notes. Because all three candidate config files are byte-identical, stage-3 read the same bddl/init files either way. The seeding block makes this deterministic and is the right fix.

**6. [CONFIRMED] — a `TASK` exported in the submitting shell now silently retargets a training arm. Severity: low-medium.**
`TASK` is a generic name and `ws_bcbb_pipeline.sh` reaches the recipe through `env …` without `-i`, i.e. it inherits exports — exactly the mechanism the file's own header documents for `LORA=1` (`ws_bcbb_pipeline.sh:20-24`). The new `probe_candidate_q_spread_stab.sbatch` `export TASK=libero_90_38`. Mitigation already present: the banner prints `task=$TASK` (`:169`). Consider the same one-line warning the `LORA` trap got.

**7. [CONFIRMED] — `TASK` is single-task only. Severity: low (documentation).**
`--collect.tasks "$TASK"` is one quoted word. `TASK="libero_90_31 libero_90_38"` renders as a **single** argv token (shim-verified), giving `tasks=['libero_90_31 libero_90_38']` and a nonsense suite name at `"_".join(task.split("_")[:-1])`. The header says "task id" (singular), so this is consistent; the multi-task route is the new passthrough (`… --collect.tasks a b --collect.eval_tasks a b`, which *replaces* the recipe's — verified).

**8. [PLAUSIBLE] — the 5 h walltime may be tight. Severity: medium (a preempt job that hits the wall loses the unfinished waves; finished waves survive — `flush()` runs per wave, `probe_candidate_q_spread.py:516`).**
The header extrapolates from the mt4 smoke (28 min per 8-episode wave, 160 chunks, M=8) as `× 82/160 × 4 ≈ 1 h/wave`, but that measurement was on **2 GPUs / FSDP=2** and this job is 1 GPU. It also charges 4 passes as "4× M" when the per-call fixed work (`nnx.merge` + `compose_full_params` + `zero_lora_params` over the 3B tree, plus the critic-prefix VLM forward on 8 rows, `advantage_weighted_sft_learner.py:472-489, :565-571`) is *also* multiplied by 4 while the batch stays at 64 rows. 2 waves + ~25 min startup could land near 4-5 h.

**9. [CONFIRMED, informational] — `pytest tests/ogpo` cannot complete on the login node, and 3 tests fail for unrelated reasons.** See the test section below.

## Before/after renders — commands and results

Harness (`$S` = scratchpad): the pre-edit copy and the current file were run **through the same mechanism** — an exported bash function `uv() { printf 'uv\n'; printf '%s\n' "$@" > "$RENDER_OUT"; }`, which shadows the binary by name resolution and therefore survives the script's own `export PATH="$HOME/.local/bin:$PATH"`. Each case: `env -u <all recipe vars> PROJECT_DIR=<repo> STORE_ROOT=$S/render/store <case env> RENDER_OUT=… timeout 120 bash <file>`, then `diff` of the two argv dumps.

**On the soundness of the comparison**: `ENTRY=/bin/echo` through `uv run` would *not* have been a sound before/after comparison — it changes `$ENTRY`, which is itself part of the rendered command, and it actually executes `uv` (venv resolution, network, and — since 21:13 — it would also be the thing the seeding block runs). The function shim is applied identically to both files, changes no variable the scripts read, and dumps argv one token per line so an empty argument would show as an empty line. `DRY=1` was then separately shown to print exactly that argv (`diff` of the space-joined shim output vs the `DRY` line: identical apart from a trailing newline).

```
A_baseline      ARM=N GPU=3 NORM=1                                  IDENTICAL argv (96 tokens)
B_wave1_ENCG    ARM=ENCG GPU=6 EMA=0.999 NORM=1 CLIP_SYM=4.0 ACCUM=2   IDENTICAL (100)
C_wsbcbb_stageA GPU=0 ARM=iii_bcA ENTRY=…unfrozen_backbone.py CONFIG_NAME=…_unfrozen_backbone
                N_STEPS=20000 SAVE_INT=20000 PG_START=99999 BURST=1000  IDENTICAL (99)
D_wsbcbb_stageB GPU=0 ARM=iii WEIGHT_LOADER=… N_STEPS=80000 PG_START=0
                PG_RAMP=5000 BURST=1000 CONS=1 NORM=1 CLIP_SYM=4.0      IDENTICAL (108)
E_stage3_80k    ENTRY=…/stage3_rollouts.py GPU=0 ARM=NCB_libero_90_31   IDENTICAL (95)
F_stage4        ENTRY=…/stage4_trained_instructions.py GPU=0 ARM=…      IDENTICAL (95)
G_kitchen_sink  LORA=1 FSFT=1 INIT_ROLLOUTS=100 FC_INT=1000 FC_ROLLOUTS=2
                QS=10 UTD=2 BURST_MC=1 CONS=1 NORM=1 CLIP_SYM=4.0
                ACCUM=2 PG_START=0 PG_RAMP=5000 BURST=1000 WEIGHT_LOADER=… IDENTICAL (120)
H_ckptmode      ARM=H GPU=0 CKPT_MODE_FLAG=--resume                     IDENTICAL (95)
I_fresh         ARM=I GPU=0 FRESH=1                                     IDENTICAL (95)
J_lora0         ARM=J GPU=0 LORA=0                                      IDENTICAL (95)
```
Exit code 0 on both sides in every case. Re-run unchanged after the 21:13 seeding-block edit. Only stdout difference anywhere: `[stability-study] … arm=N` → `… arm=N task=libero_90_44 …`.

`scripts/stability_wave1_bc0.sh:25` was read, not rendered: its `"$@"` are `SEED=1` / `EMA=0.999` style tokens placed **before** `nohup`, so `env` consumes them as assignments — they never become script arguments. Case B reproduces that env.

**Wrapper render** — the wrapper's lines 47-end executed verbatim with only `STORE_ROOT` redirected to a fake `…/vla_single_task` tree, `OUT_DIR` to scratch, and `exec bash` → `exec env DRY=1 bash`:
```
uv run scripts/probe_candidate_q_spread.py pi05_libero_online_ogpo_sft
  --project_name ogpo_stability --group_name stability_study
  --exp_name stab_NCB_libero_90_38
  --checkpoint_base_dir <STORE_ROOT>/checkpoints/stability_study
  --seed 0 --fsdp_devices 1 --resume --log_interval 25
  --save_interval 100000 --keep_period 100000 --num_train_steps 100000
  --ema_decay 0.99 --lr_schedule.value 2.5e-5 --max_runtime 169200
  --collect.tasks libero_90_38 --collect.eval_tasks libero_90_38
  … --rl.critic.inference_start_step 1 …
  --batch_size 32 --rl.normalize_group_advantage --rl.adv_clip_sym 4.0
  --rl.post_collection_critic_steps 1000 --no-backbone_lora --rl.n_samples 8
```
Env handed to the probe: `QSPREAD_BON=1 QSPREAD_PASSES=4 QSPREAD_EPISODES=16 QSPREAD_SEED=0 QSPREAD_OUT_DIR=… BON_N=8`.

## Multi-pass addition

**(a) Exactness.** The claim holds, in the distributional sense, and the two properties it rests on are real:
- **Fresh noise per call.** `sample_actions` does `rng, self._rng = jax.random.split(self._rng)` on entry (`advantage_weighted_sft_learner.py:423`) and passes that half to `_sample_action`, which draws `jax.random.normal(rng, (group_env_num * n_samples, H, A))` (`filtered_sft_learner.py:623-625`). iid across the batch axis, and `self._rng` advances, so 4 calls of 8 are 32 iid draws exactly as 1 call of 32 is. New test `test_real_sample_actions_draws_a_fresh_rng_each_call` asserts this on the real body (fresh key each call, `self._rng` advanced).
- **Deterministic, row-independent scoring.** `q_model.eval()` (`:462`); the critic is `BroNet`, whose only normalisation is `nnx.LayerNorm` (`src/rl/networks/bronet_critic.py:17,19,59`) — per-example, no batch statistics, no dropout on this path. So candidate *j*'s score does not depend on which other rows were in the batch, and the union argmax over 32 is the same selection a 32-wide call would make on those candidates.

Two caveats, neither fatal. (i) The equality is *distributional*, not sample-wise — 4 keys ≠ 1 key, so a 4-pass run is not bit-reproducible against a 1-pass run at the same seed; the docstring's "the union argmax **is** the single-pass selection" reads stronger than what holds. (ii) Different XLA batch shapes (64 vs 256 rows) can differ in the last bits, which could flip a near-tie in `argmax`. Immaterial for a variance statistic.

**(b) Concatenation / `executed_idx`.** Correct. `cands`/`scores` are concatenated on axis 1 in pass order, `exec_idx = scores.argmax(axis=1)` indexes the same axis, and `act = cands[arange(G), exec_idx]` (`probe_candidate_q_spread.py:245-260`), so union index *j* means pass `j // n_samples_per_pass`, candidate `j % n_samples_per_pass`. The JSON records `n_samples` (total), `n_samples_per_pass`, `passes` (`:441-443`) so it decodes. New test `test_multipass_union_index_resolves_to_the_right_pass_and_candidate` places the two envs' winners in *different* passes (first and last) so a reversed/rotated part order changes both answers — the implementer's test puts the winner in the last candidate of the last pass, which several wrong orders still satisfy. `test_multipass_bon_off_executes_pass_zero_candidate_zero` pins that the control arm still executes pass-0 candidate 0 with 5 better-scoring candidates drawn after it, while still recording an M-wide score row (so both arms measure spread at the same M). `test_split_over_passes_executes_the_same_candidate_as_one_call` drives an identical 6-candidate pool as 1×6 and 3×2 and asserts bit-identical executed action.

**(c) Per-call state.** Nothing else varies between passes. `_process_obs_for_pi0`, `_policy._input_transform`, `_state_normalize`, `_action_normalize` are stateless; the prefix model is rebuilt deterministically from the same train state each call (`:472-489`); there is no obs-keyed cache. `_task_registry` is `None` for this config (`num_tasks=None`, confirmed by `cli()`), and even when on, `_task_slot` is first-seen-then-stable, so pass 2 reuses pass 1's slot. `_sample_action`'s shapes are identical across passes (`n_samples` fixed at 8), so no jit retrace.

**(d) Failure scenarios.**
- **The `passes==1` cross-check never runs in the shipped arm.** `exec_idx == best_parts[0]` — the check that the recorded scores are the ones production selected on — is guarded by `if passes == 1` (`:252`), and the wrapper defaults to `passes=4`. The per-pass check (`cands_p[arange, best_idx_p] == best`, `:239`) still runs every pass and does pin env alignment, so the gap is narrow; my `test_passes_one_cross_check_fires_but_is_off_for_passes_gt_one` records it. Cheap mitigation: one `QSPREAD_PASSES=1` job first, or as the smoke.
- **NaN scores are not checked.** `best` (the action chunk) is checked for finiteness (`:231-236`), `scores` is not; `argmax` over NaNs silently returns index 0. Same in production (`:619`), so not a regression.
- Host memory for the union is negligible (8×32×10×7 float32 ≈ 71 KB); GPU peak is still set by the per-call `n_samples=8` tiling, which is the point of the knob.
- The BoN=0 arm still runs all 4 passes. That is *correct*, not waste — it keeps the variance statistic at the same M as the BoN=1 arm.

## Tests

```
$ pytest tests/ogpo/test_candidate_q_spread.py tests/ogpo/test_candidate_q_spread_verifier.py -q
42 passed, 3 warnings in 11.75s      (16 implementer + 26 verifier, incl. 5 new multi-pass tests)
```
`pytest tests/ogpo` as a single process **dumps core on the login node** (documented in CLAUDE.md; needs `sbatch scripts/run_ogpo_tests.sbatch`). Run file-by-file instead:
```
test_backbone_lora_config.py           10 passed          test_per_task_critics_verifier2.py  30 passed
test_backbone_lora_grads.py            core dump          test_resume_hardening.py            16 passed
test_backbone_lora_init.py             core dump          test_resume_hardening_verifier.py   58 passed
test_backbone_lora_verifier.py         core dump          test_reward_and_value_bounds.py     10 passed
test_candidate_q_spread.py             16 passed          test_sampling.py                    core dump
test_candidate_q_spread_verifier.py    26 passed          test_split_equivalence.py           core dump
test_ema_utils.py                       1 passed          test_grad_norm_decomposition.py     core dump
test_episode_steps_multiplier.py       21 passed, 3 skip  test_group_dedup.py                 core dump
test_per_task_critics.py               core dump          test_verifier_alignment.py    3 FAILED, 114 passed, 3 skipped
```
The core dumps are the PaliGemma-backed legs the login node kills — not failures.

The **3 failures are pre-existing and unrelated**:
```
FAILED test_head_value_distribution_differential_over_every_registered_config
       E  assert -99.99999999999991 < -99.99999999999991
FAILED test_head_value_distribution_differential_is_a_real_change_at_201_bins
       E  assert (-200.49999999999983, 0.49999999999999956) != (same tuple)
FAILED test_head_wrapper_differential_over_a_randomized_flag_script
       E  Failed: DID NOT RAISE <class 'TypeError'>
```
These fetch the implementation from `git HEAD` by `subprocess` and assert the working tree **differs**; commit `5b94510` ("OGPO reference alignment…") already contains that change, so the differential's premise is stale. Subject is `src/rl/value_distribution.get_value_bounds` / `TimeToSuccessAsRewardWrapper` (`tests/ogpo/test_verifier_alignment.py:30-33`); the file references neither `stability_study.sh` nor `probe_candidate_q_spread.py`. Nothing in this change can reach them.

## What remains unverified

- **The GPU run itself.** No GPU, no `/data`. Whether `stab_NCB_libero_90_38`'s `resume_state.json` names step 100000 and whether orbax holds that step cannot be checked from here. Failure mode is loud (`runtime_state.py:101-109` raises, and `probe_candidate_q_spread.py:398-404` raises if openpi downgrades `--resume` to a fresh start) — no silent-wrong path.
- **Peak RSS of this specific job.** The only comparable measurement (`10274728`) is censored at its cap; the argument that 200 G suffices rests on the checkpoint's own training run at 138.6 GiB, which I checked but could not re-measure.
- **Wall clock** (finding 8): no way to measure the 1-GPU, 4-pass, 82-chunk rate without running it.
- **That the sibling `c3dbcac` checkout's `src/` matches this one.** I diffed the *recipes* and confirmed no tree-shaping flag differs, and the 2026-08-31 restore of a sibling checkpoint through this code proves compatibility empirically for the 80k/`_31` run — not, strictly, for the 100k/`_38` one.
- **`pytest tests/` at large** still errors at collection (OQ-4); untouched, as the change does not go near `tests/nnx_networks/`.

---

## Outcomes (implementing session)

| # | Outcome |
|---|---|
| 1 | **Fixed.** Wrapper header now cites the training run's 138.6 GiB (job 10210760) and calls the stage-3 number censored; the stage-2 job is no longer cited. Request unchanged at 200G. |
| 2 | **Fixed (docs).** Recipe header: `DRY` "prints instead of running the entry point; still seeds `$LIBERO_CONFIG_PATH/config.yaml` if missing". Ordering left as in the mt4 recipe. |
| 3 | **Fixed (docs).** DIFF.md and BLAST-RADIUS.md refreshed: 8 h, four `QSPREAD_*` knobs, 16 episodes, **M = 32** stated, both render methods described. |
| 4 | **Fixed (docs).** Wrapper header states the one write into the store root (the 616 B default config). |
| 5 | **Recorded.** BLAST-RADIUS.md's "found the hard way" section now carries the verifier's resolution: the file never existed on that root; stage 3 inherited `LIBERO_CONFIG_PATH` via `--export=ALL`. |
| 6 | **Fixed (docs).** Recipe header carries the `TASK` inheritance trap next to the `LORA` one; `docs/code/scripts.md` gotchas gained the entry. |
| 7 | **Fixed (docs).** Recipe header says ONE task id and points multi-task at the passthrough. |
| 8 | **Acted on.** Job 10352252 was cancelled while still pending and resubmitted as **10352338** with `--time=08:00:00` (partition max is 31 d); wrapper header re-budgets at up to ~2 h per wave on one GPU and notes the per-wave JSON flush. |
| 9 | **Noted.** Same three pre-existing `test_verifier_alignment.py` failures as in the probe record. |
| (a) | **Accepted.** Docstring claim is distributional; a `QSPREAD_PASSES=1` job would be the bit-level comparison — not run. |
| (d) | **Accepted.** The `passes==1` cross-check is off at `passes=4`; the per-pass alignment check still runs. NaN scores unchecked as in production. |

Tests after the outcomes: unchanged code paths, `42 passed`.

## GPU run

Job 10352129: FAILED in 17 s at LIBERO's prompt (before the seeding block).
Job 10352252: cancelled while pending (wall-time raise).
Job 10352338: FAILED after 7 min. The interesting half worked — `Resume
resolved to step 100000 (orbax steps=[100000], per-step manifests=[])`, the
250k-capacity replay buffer restored in ~4 min, `Restored training
checkpoint ... at committed step 100000` on one GPU — and then
`create_trained_policy` (the lora-less policy twin that supplies the
transforms, `filtered_sft_learner.py:408`) died with
`FileNotFoundError: Metadata file (named _METADATA) does not exist at
<STORE_ROOT>/cache/openpi/openpi-assets/checkpoints/pi05_libero/params`:
redirecting `STORE_ROOT` moved `OPENPI_DATA_HOME` to a root that holds no
openpi asset cache. Verifier finding 5 had already established that the
campaign's runs kept the local store and redirected only `CKPT_BASE_DIR`
(the stage-3 probe redirected `STORE_ROOT` and survived only because its
submitting shell exported the cache paths). Fix: the recipe regains the
sibling's `${CKPT_BASE_DIR:-...}` override (default unchanged; equivalence
re-rendered), the wrapper sets `CKPT_BASE_DIR` alone. MaxRSS at the point of
death 87 GB. Resubmitted as **job 10352521**, outputs
`/home/pchellap/logs/q_spread_stab_10352521_bon1/`; results appended when it finishes.

### Job 10352521 — COMPLETED

24:24 wall, MaxRSS 108 GB (the 8 h / 200G envelope was generous: ~11 min
startup incl. the 250k-buffer restore, then **~8 min per 8-episode wave** at
M = 4 × 8 on one RTX PRO 6000 — the verifier's finding-8 worry did not
materialise; the per-call fixed work is small next to the 82-chunk episodes).

```
[qspread] agent restored at step 100000 ...; M=32 (4 pass(es) x 8) bon=True reduction=min
[qspread] libero_90_38 wave=0 recorded=8/8 wave_SR=0.375 cum_SR=0.375 steps=[382, 400, 400, 400, 400, 400, 390, 400] q_var[min/median/max]=0.003233/0.06095/6097
[qspread] libero_90_38 wave=1 recorded=8/8 wave_SR=0.625 cum_SR=0.500 steps=[388, 365, 370, 400, 359, 400, 400, 398] q_var[min/median/max]=0.005391/0.07076/1371
[qspread] libero_90_38     SR=0.500 n=16
```

Outputs `/home/pchellap/logs/q_spread_stab_10352521_bon1/libero_90_38/`
(16 episode PNGs, `mean_trace.png`) and `q_spread_results.json`. The
executed candidate's pass index is close to uniform over the four passes in
every episode (e.g. `[23, 17, 21, 16]`), so the union argmax shows no
pass bias. Plots inspected; render as designed.
