# Implementation plan

1. `src/rl/ogpo/update_actor.py` — jit-2a (`loss_and_grad_pg`):
   compute `optax.global_norm(grads_pg)` after `nnx.value_and_grad` returns,
   and return `pg_aux | {"grad_norm_pg": ...}`. The norm cannot live inside
   `loss_fn` (the grads do not exist yet there), so it is folded into the aux
   at the return statement. Update the `20 PPO keys` / `33-key schema` comment.
2. `src/rl/ogpo/update_actor.py` — jit-2b (`bc_grad_accumulate`):
   in the `use_bc_regularization` branch, take `optax.global_norm(grads_bc)`
   immediately before the accumulating `jax.tree.map`; in the `else` branch set
   it to `jnp.float32(0.0)`. Add to `loss_aux`. Update the `22-key` comment.
3. `tests/ogpo/test_split_equivalence.py` — add `_SPLIT_ONLY_INFO_KEYS`
   and teach `_assert_info_close` to require the reference keys to match
   exactly *and* the extra keys to be exactly that set.
4. `tests/ogpo/test_grad_norm_decomposition.py` — new file, four asserts:
   keys present and finite; `advantage = 0` ⇒ `grad_norm_pg == 0` and
   `grad_norm == grad_norm_bc`; `bc_coeff = 0` ⇒ `grad_norm_bc == 0` and
   `grad_norm == grad_norm_pg`; triangle inequality in the general case.
5. `pytest tests/ogpo`.
6. Docs: `docs/code/rl-ogpo.md` (`:169`, `:381`), `docs/code/tests.md` (`:244`).
