# Diff

## `scripts/stability_study.sh` (edited)

- Header comment: documents `TASK`, `DRY`, and the trailing-argument
  passthrough; adds the `DRY=1` usage line.
- `TASK="${TASK:-libero_90_44}"` after `EMA=`.
- Banner line gains `task=$TASK`.
- `uv run "$ENTRY" \` replaced by the `RUN=(uv run "$ENTRY")` /
  `[ "${DRY:-0}" = "1" ] && RUN=(echo uv run "$ENTRY")` / `"${RUN[@]}" \`
  triple, copied from `ogpo_multitask_4task.sh:255-258`.
- `--collect.tasks libero_90_44` / `--collect.eval_tasks libero_90_44` →
  `"$TASK"` (sibling checkout's `:171-172`).
- `"${EXTRA_FLAGS[@]}"` → `"${EXTRA_FLAGS[@]}" \` + `"$@"`
  (`ogpo_multitask_4task.sh:311`).
- **`CKPT_BASE_DIR` override** (added after job 10352338 restored the
  checkpoint fine and then died in `create_trained_policy` on a missing
  `_METADATA` under `<STORE_ROOT>/cache/openpi/...`): `CKPT_BASE_DIR` is now
  `${CKPT_BASE_DIR:-$STORE_ROOT/checkpoints/stability_study}`, the sibling
  recipe's form (`:46`). Default unchanged. The wrapper now redirects only
  this, leaving every cache in the local `run_store` — which is how the
  campaign's own runs were launched (verifier finding 5).
- **LIBERO config seeding** (added after job 10352129 died in 17 s on
  `EOFError` at LIBERO's first-import prompt): the
  `if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then printf 'n\n' | uv run
  python -c "import libero.libero" ...` block, copied from
  `ogpo_multitask_4task.sh:253-254`, which the sibling checkout's recipe also
  has (`:140-141`) and this copy never gained. Inert for every existing
  caller whose store already holds `libero/config.yaml` (the local
  `run_store` does); any caller without one was already dying at the prompt.

Nothing else in the file changed. Rendered command for every existing
invocation is identical (see VERIFICATION.md).

## `scripts/probe_candidate_q_spread_stab.sbatch` (new)

preempt, 1 GPU, 200G, 8 h (raised from 3 h → 5 h → 8 h on the verifier's
wall-clock finding). Exports `CKPT_BASE_DIR=/data/.../vla_single_task/checkpoints/stability_study`
(NOT `STORE_ROOT` — first draft did, see the `CKPT_BASE_DIR` bullet above),
`ARM=NCB_libero_90_38`,
`TASK=libero_90_38`, `SEED=0`,
`ENTRY=scripts/probe_candidate_q_spread.py`, `CKPT_MODE_FLAG=--resume`,
`GPU` from `CUDA_VISIBLE_DEVICES`, the NCB arm's `NORM=1 CLIP_SYM=4.0
BURST=1000`, the four `QSPREAD_*` knobs (`OUT_DIR`, `EPISODES` default
**16**, `SEED`, `PASSES` default **4**); mount-hang guard on the run dir;
`exec bash scripts/stability_study.sh --rl.n_samples "$BON_N"` (default 8).
**M = 4 × 8 = 32 candidates per state**, scored by min over the config's 2
Q heads.
Default output dir `/home/pchellap/logs/q_spread_stab_<job>_bon<0|1>/`.

## Divergence from PLAN

No `PLAN.md` — Tier 1, one pass. Matches `BLAST-RADIUS.md`.
