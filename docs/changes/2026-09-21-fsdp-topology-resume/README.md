# Cross-topology resume: restore the policy train_state onto the current mesh

**Tier 2** (sharding annotation + `TrainState`/checkpoint restore contract, reached
by all six learners through `FilteredSFTLearner.__init__`). Discovery pass only —
this directory holds `README.md` and `BLAST-RADIUS.md`; no source was touched.

## What the change is

Make an OGPO run checkpointed under one `fsdp_devices` resumable under another, by
passing the current run's sharding tree into the orbax restore of the main policy
`train_state` instead of letting orbax reuse the checkpoint's saved device layout.

Today `init_train_state` computes the right target sharding and then throws it away
on the resume path: it returns the bare `jax.eval_shape` tree (every leaf
`.sharding is None`) as the restore target
(`src/rl/filtered_sft_agent/filtered_sft_learner.py:198-202`), and
`filtered_sft_learner.py:315-324` hands orbax only that tree — never
`self._train_state_sharding`. openpi's `restore_state` then calls
`checkpoint_manager.restore(step, items=…)` with no `restore_args`
(`openpi/src/openpi/training/checkpoints.py:100-106`), so orbax falls back to the
checkpoint's own `_sharding` metadata file and tries to map its recorded **device
ids** onto the current allocation.

The critic/value/normalizer half of the resume already does this correctly —
`AdvantageWeightedSFTLearner` passes a *concrete* target built from the current mesh
into `ocp.StandardCheckpointer.restore` (`advantage_weighted_sft_learner.py:153`,
`:254-259`), and that handler derives per-leaf `ArrayRestoreArgs(sharding=…)` from
the target. So on a topology change the critics resharded and the policy did not.
This change brings the policy path to parity.

## Why

**The maintainer wants to cycle one multitask OGPO run between 1- and 2-GPU
allocations depending on what the cluster has free.** That is already the documented
intent of the recipe stack: `scripts/ogpo_multitask_4task_maxlab.sbatch:5` defaults to
`--gres=gpu:1` with `FSDP=1` (`:42`) and its own header (`:39-40`) says

> Batch 32 is global under FSDP, so more GPUs never changes the science —
> to shard for speed, submit with: `sbatch --export=ALL,FSDP=2 --gres=gpu:2 ...`

Since the recipes pass `--resume` by default (2026-08-27), re-submitting that line
against an existing FSDP=1 run is the obvious move, and it is exactly what does not
work. The science is unaffected by the switch (batch size is global under FSDP), so
the only thing standing between the maintainer and using whatever GPUs are free is
this restore path.

It has already cost a job. `docs/changes/2026-09-07-candidate-q-spread-rollouts/`
records an FSDP=2 checkpoint restored on a 1-GPU probe dying at restore (job
10324589, 2026-09-05) with

```
ValueError: sharding passed to deserialization should be specified, concrete
    and an instance of `jax.sharding.Sharding`. Got None
```

and explicitly defers the fix: *"The proper fix — annotating the restore target with
`state_sharding` — touches the sharding contract of every learner's resume and is
Tier 2; not done here"* (`BLAST-RADIUS.md:97-99`). `docs/code/scripts.md:429-442`
carries the same note. The four rollout/analysis probes
(`probe_rollout_gifs.py`, `probe_counterfactual_rollouts.py`,
`probe_candidate_q_spread.py`, `probe_value_next_spread.py`) all construct a learner
with `--resume` and are therefore all pinned to the checkpoint's original GPU count;
`probe_candidate_q_spread.sbatch:83` hardcodes `FSDP=2` with the comment
`# = the checkpoint's fsdp_devices`, which is the workaround this change removes the
need for.

Secondary benefit: orbax emits a `UserWarning` on **every** resume today —
*"Sharding info not provided when restoring. Populating sharding info from sharding
file. … this option is unsafe when restoring on a different topology than the
checkpoint was saved with"* — which the fix silences by making the sharding explicit.

## Outcome (added after implementation + verification)

Implemented in `filtered_sft_learner.py` only (`DIFF.md`); independently verified on
CPU and on real π0.5 weights at 1→2, 2→2 and 2→1 GPUs (`VERIFICATION.md`). No-op at
an unchanged topology. One deliberate behaviour change: restore now raises on a
target-vs-stored leaf *shape* mismatch where it used to silently return the stored
shape (`DIFF.md`, "Behaviour change recorded").

## What is not in scope

Cross-topology restore of the **replay buffer, OGPO's extra resume state, and the
resume manifest**: all three are already topology-independent (plain numpy/JSON, no
device arrays, verified — see `BLAST-RADIUS.md` §C). `src/rl/dsrl/`'s
`init_train_state` clone is **not** a second site for this fix — it has no `resume`
parameter and so never takes the buggy branch (`src/rl/dsrl/dsrl_env.py:70-74`).
Auto-deriving `fsdp_devices` from the allocation is a separate question and is left
alone.
