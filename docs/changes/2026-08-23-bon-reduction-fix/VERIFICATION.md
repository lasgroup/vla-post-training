# Verification

No independent verifier agent and no new tests: the maintainer's standing
instruction for this line of work is to make the change without a test pass.
Stated plainly so the gap is on the record.

## Done
- Duplication sweep: `grep -rn "scores.min(axis=0)" src/rl/` returns nothing after
  the fix; both clone copies patched.
- Import check under the recipe's `PYTHONPATH`: all four affected learner modules
  (`advantage_weighted_sft`, `best_of_n`, `ogpo`, `mpo_weighted_sft`) import cleanly.
- Confirmed no local (`rl_config`, `_lower`, `_upper`, `q_dist`) was reused after
  either rewritten block before deleting it.
- Confirmed `summarize_critic_values` calls `make_value_distribution` with the same
  four arguments the deleted code did, so the distribution construction is unchanged.

## `pytest tests/ogpo` — RUN, job 10201675 (general, CPU-only, 15m41s)

```
4 failed, 221 passed, 3 skipped, 159 warnings in 941.00s (0:15:41)
FAILED tests/ogpo/test_verifier_alignment.py::test_best_of_n_collection_scoring_ignores_critic_reduction
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_over_every_registered_config
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_is_a_real_change_at_201_bins
FAILED tests/ogpo/test_verifier_alignment.py::test_head_wrapper_differential_over_a_randomized_flag_script
```

**Failure 1 is caused by this change and is the intended outcome.** That test was a
deliberate pin on the defect — its own docstring says "Pinned, not asserted-away" —
asserting `"scores.min(axis=0)" in body`. Rewritten as
`test_best_of_n_collection_scoring_honours_critic_reduction`, which now asserts the
opposite across BOTH clone copies. Passes.

**Failures 2-4 are pre-existing and unrelated.** Established, not assumed:
- this diff touches only the two learner files; `git status src/rl/value_distribution.py`
  is clean;
- `_head_get_value_bounds()` (`test_verifier_alignment.py:1106-1120`) lifts
  `get_value_bounds` out of **`git show HEAD:src/rl/value_distribution.py`**;
- commit `5b94510` (already at HEAD) contains the `get_value_bounds` change.

So the "before" blob and the working-tree "after" are the same text, and the
differential asserts `before != after`. These tests were written to verify an
*uncommitted* change and went tautologically red when it was committed. The fix is to
re-point them at `5b94510^` instead of `HEAD`. **Not done — outside this change's blast
radius, and `tests/ogpo/test_verifier_alignment.py` is someone else's in-flight
uncommitted work.** Raised with the maintainer.

## NOT done
- No runtime execution of the patched path. `sample_actions` needs real pi0.5
  weights, a GPU and a simulator.
- The claim that the min/mean reduction explains the -19 Q-vs-TD gap is
  **unconfirmed**: direction and rough magnitude fit, but the across-head spread at
  those states was not measured.
- Whether mean-reduced ranking scores better against realised return than
  min-reduced is **unmeasured**. Tier C ran entirely on the min path. A paired
  answer needs per-head scores in `_bon_record` and one re-run.
