# VERIFICATION — episode-steps multiplier

Step-3 verifier: fresh-context Opus agent, spec (BLAST-RADIUS.md + DIFF.md) +
working tree, adversarial. Findings below are its report **verbatim in
substance** (trimmed only of formatting); resolutions by the implementing
session follow each finding.

## PASS items (verifier)
1. **Multiplier reaches TimeLimit in both domains.** libero executed on CPU via
   closure inspection: `libero_90_79` → 400/800/1200 at mult 1/2/3, `libero_10_0`
   → 520/1040/1560; molmo read-verified only (`molmo_spaces` not importable on
   CPU) — `config` is in scope, value inline in the constructor. No missed cap
   site (grep over TimeLimit / max_episode_steps / all suite constants).
2. **Behavior preservation at default 1.** All 11 registered online configs
   resolve the field to 1; no getattr-default reads; `--collect.episode_steps_multiplier 1`
   yields a collect block identical to no flag.
3. **New tests** `tests/ogpo/test_episode_steps_multiplier.py`: 20 passed,
   3 skipped, 11s (skips = the three configs with no `.critic` block, same as
   test_verifier_alignment). Exact `==` tolerances — both sides run the identical
   float expression, so any difference is a real behavior change.
4. **Existing suite** (`-k "value or bounds or config"`): 2 failed, 59 passed —
   both failures proven pre-existing by surgically reverting ONLY the in-scope
   value_distribution.py hunk (they persist; they are copy-vs-current identity
   pins a ×1 multiplier cannot perturb). Full `pytest tests/ogpo` aborts on the
   login node in test_grad_norm_decomposition.py (pre-existing, reproduced with
   the new file excluded) — the sbatch suite was not run.
5. **Recipe.** `bash -n` clean; DRY: EP_MULT=2 → one
   `--collect.episode_steps_multiplier 2` + `--overwrite`; EP_MULT unset → the
   flag at 1 (harmless no-op), `--resume`.

## Failures / gotchas (verifier) → resolutions (implementer)
- **F-A. Blast-radius miss:** three ad-hoc `SimpleNamespace` stub configs call
  `make_env_libero` with only `env_resolution` + `num_steps_wait`
  (`scripts/repro_egl_drain.py:75-78`, `scripts/egl_safe_probe.py:259-264`,
  `:466-471`) and now raise `AttributeError` (reproduced). Not the training
  path (EGL debug probes only).
  → **Fixed:** all three stubs now pass `episode_steps_multiplier=1`; stale
  "only reads env_resolution + num_steps_wait" comment updated. The verifier's
  contract pin (`test_make_env_libero_now_requires_the_field_on_its_config`)
  kept, docstring updated to record the resolution.
- **F-B. No fail-fast on the field:** `EP_MULT=0` / `-1` were accepted;
  gymnasium's TimeLimit truncates on `elapsed >= max_episode_steps`, so both
  truncate every episode at step 0 — a run that trains and means nothing.
  → **Fixed:** `OnlineTrainConfig.__post_init__` raises `ValueError` (with the
  fix in the message) for multiplier < 1; pinned by the new
  `test_multiplier_below_one_is_rejected_at_config_construction`.
- **F-C. Recipe header row misplaced** (EP_MULT line split BATCH's two-line
  entry). → **Fixed:** rows reordered.
- **F-D. Step-4 docs not yet done** at verification time. → **Done after:**
  `docs/code/envs.md` (libero + molmo TimeLimit bullets), `docs/code/scripts.md`
  (EP_MULT knob row). `training.md:122`'s `max_episode_steps=450` cite refers to
  the value-bounds field, still accurate — untouched.

Post-fix re-run: `pytest tests/ogpo/test_episode_steps_multiplier.py` →
**21 passed, 3 skipped**.

## Could not be verified (honest list)
- Real truncation at 800 env steps: needs GPU + EGL LIBERO rollout. Only the
  integer's arrival at the `TimeLimit` constructor is verified.
- `src/envs/molmo.py` wiring: read-verified only (`molmo_spaces` not installed).
- Full `pytest tests/ogpo` via sbatch: not run (no-launch rule).
- Wall-time / discount-shift consequences of EP_MULT=2: run-level.
