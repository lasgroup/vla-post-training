# ruff: noqa: F722
"""Adversarial verification of cross-topology (`fsdp_devices`) policy resume.

Covers `src/rl/filtered_sft_agent/filtered_sft_learner.py`'s
``_params_item_sharding`` / ``_restore_state_sharded`` (2026-09-21 Tier-2
change), which replace openpi's ``restore_state`` on the single inherited
``FilteredSFTLearner.__init__`` restore call.

Required properties, in order of importance:

1. **M == N is a no-op.** Restoring through the new helper on the SAME mesh the
   checkpoint was written under must produce the same treedef, bit-identical
   values and the SAME per-leaf placement as the untouched
   ``openpi.training.checkpoints.restore_state``. The untouched submodule
   function is the verbatim pre-change implementation, so the differential is
   run in-process against it rather than against a copy of the new code.
2. **M != N reshards.** Save under a 1-device mesh, restore under a 2-device
   mesh (and the reverse); values bit-identical, placement = the CURRENT mesh's
   request.
3. The ``params``/EMA item is restored with the MIXED layout ``save_checkpoint``
   writes (trainable leaves replicated, frozen leaves FSDP-sharded), not a
   uniform FSDP layout.

This module needs >= 2 JAX devices and there is no precedent for that in this
suite, so it forces two host platform devices BEFORE importing jax -- but ONLY
when jax has not already been imported by an earlier module in the same pytest
process. In a plain ``pytest tests/ogpo`` run jax is already imported at
collection time, the flag is not set, ``jax.device_count() == 1`` and every test
here SKIPS, leaving the rest of the suite's ``sharding.make_mesh(1)`` meshes
untouched. Run it standalone to actually exercise it:

    uv run pytest tests/ogpo/test_cross_topology_resume_verifier.py

``sharding.make_mesh(n)`` always spans ALL devices, so the two meshes are built
directly over device subsets instead.
"""
import os
import sys

_JAX_PREIMPORTED = "jax" in sys.modules
if not _JAX_PREIMPORTED:  # standalone process: force 2 CPU devices
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _xla_flags:
        os.environ["XLA_FLAGS"] = (
            _xla_flags + " --xla_force_host_platform_device_count=2"
        ).strip()

import dataclasses  # noqa: E402
import warnings  # noqa: E402

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
import pytest  # noqa: E402
from orbax.checkpoint.checkpoint_utils import construct_restore_args  # noqa: E402

import openpi.shared.array_typing as at  # noqa: E402
import openpi.shared.nnx_utils as nnx_utils  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.training.sharding as sharding  # noqa: E402
import openpi.training.utils as training_utils  # noqa: E402
from openpi.models import pi0_config  # noqa: E402

from src.rl.ema_utils import compose_full_params  # noqa: E402
from src.rl.filtered_sft_agent.filtered_sft_learner import (  # noqa: E402
    _params_item_sharding,
    _restore_state_sharded,
)
from src.training.config import get_config  # noqa: E402

_N_DEVICES = jax.device_count()

pytestmark = pytest.mark.skipif(
    _N_DEVICES < 2,
    reason=(
        f"needs >= 2 JAX devices, have {_N_DEVICES}; jax was already imported when this "
        "module loaded (full-suite run), so XLA_FLAGS could not be set. Run standalone: "
        "uv run pytest tests/ogpo/test_cross_topology_resume_verifier.py"
    ),
)

_P = jax.sharding.PartitionSpec
_SAVE_STEP = 7


# --------------------------------------------------------------------------- #
# Builders (dummy pi0.5 variants; no real weights, no GPU)
# --------------------------------------------------------------------------- #
def _build_config(*, action_dim: int = 4):
    base = get_config("pi05_libero_online_ogpo_sft")
    dummy_model = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=action_dim,
        action_horizon=2,
        max_token_len=8,
        pi05=True,
    )
    # Production freeze_filter kept on purpose (_make_ogpo_freeze_filter): the
    # trainable/frozen split is what _params_item_sharding keys off.
    return dataclasses.replace(base, model=dummy_model, batch_size=2)


def _make_params(config, model):
    params = nnx.state(model)
    return nnx_utils.state_map(
        params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
    )


def _build_train_state(config, *, with_ema: bool, seed: int = 0):
    """A realistic TrainState: real nnx.State / VariableState / optax opt_state /
    nnx.GraphDef, mirroring `init_train_state`'s construction."""
    model = config.model.create(jax.random.key(seed))
    params = _make_params(config, model)
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )
    return training_utils.TrainState(
        step=jnp.asarray(_SAVE_STEP, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(nnx.filter_state(params, config.trainable_filter)),
        ema_decay=config.ema_decay if with_ema else None,
        ema_params=params if with_ema else None,
    )


def _shape_tree(state):
    """Shape-only mirror of `state`. jax.tree.map preserves the treedef, so the
    static (`pytree_node=False`) fields `tx` / `ema_decay` and the GraphDef stay
    the SAME OBJECTS -- which is what lets the shape tree and the sharding tree
    be compared/zipped later (see B.1)."""
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)), state
    )


def _mesh(n_devices: int) -> jax.sharding.Mesh:
    devs = np.array(jax.devices()[:n_devices]).reshape(1, n_devices)
    return jax.sharding.Mesh(devs, (sharding.BATCH_AXIS, sharding.FSDP_AXIS))


class _DummyDataConfig:
    norm_stats = None
    asset_id = None


class _DummyLoader:
    """`save_state` only calls `data_config()` to decide whether to write norm
    stats; None/None makes that a no-op, keeping the real save path."""

    def data_config(self):
        return _DummyDataConfig()


class _Ckpt:
    """A real, production-shaped checkpoint: `initialize_checkpoint_dir` +
    `save_state`, i.e. the same CheckpointManager (`item_handlers` with
    PyTreeCheckpointHandler for train_state/params) and the same two-item split
    the learner uses."""

    def __init__(self, directory, config, mesh, *, with_ema=True, min_size_mbytes=4, seed=0):
        self.config = config
        self.mesh = mesh
        self.with_ema = with_ema
        self.replicated = jax.sharding.NamedSharding(mesh, _P())
        state = _build_train_state(config, with_ema=with_ema, seed=seed)
        self.shape_tree = _shape_tree(state)
        self.state_sharding = sharding.fsdp_sharding(
            self.shape_tree, mesh, min_size_mbytes=min_size_mbytes
        )
        # Put the state on the save mesh exactly as a live run would hold it.
        state = jax.device_put(state, self.state_sharding)
        self.state_to_save = self._compose_as_save_checkpoint(state)
        self.manager, resuming = _checkpoints.initialize_checkpoint_dir(
            directory, keep_period=None, overwrite=True, resume=False
        )
        assert resuming is False
        _checkpoints.save_state(self.manager, self.state_to_save, _DummyLoader(), _SAVE_STEP)
        self.manager.wait_until_finished()

    def _compose_as_save_checkpoint(self, state):
        """Byte-for-byte the body of `FilteredSFTLearner.save_checkpoint`."""
        if not self.with_ema:
            return state
        ema_trainable = nnx.filter_state(state.ema_params, self.config.trainable_filter)
        ema_rep = jax.device_put(ema_trainable, self.replicated)
        full_ema = compose_full_params(
            state.params, ema_rep, self.config.trainable_filter
        )
        return dataclasses.replace(
            state, ema_params=full_ema, ema_decay=self.config.ema_decay
        )

    def target_for(self, mesh, *, min_size_mbytes=4):
        """(shape_tree, state_sharding, replicated) for restoring onto `mesh`."""
        return (
            self.shape_tree,
            sharding.fsdp_sharding(self.shape_tree, mesh, min_size_mbytes=min_size_mbytes),
            jax.sharding.NamedSharding(mesh, _P()),
        )


# --------------------------------------------------------------------------- #
# Tree comparison helpers
# --------------------------------------------------------------------------- #
def _paths_and_leaves(tree):
    return jax.tree_util.tree_flatten_with_path(tree)[0]


def _assert_values_identical(a, b, label):
    la, lb = _paths_and_leaves(a), _paths_and_leaves(b)
    assert len(la) == len(lb), f"{label}: leaf count {len(la)} != {len(lb)}"
    for (pa, va), (pb, vb) in zip(la, lb):
        assert pa == pb, f"{label}: path {pa} != {pb}"
        na = np.asarray(jax.device_get(va))
        nb = np.asarray(jax.device_get(vb))
        assert na.dtype == nb.dtype, f"{label}{jax.tree_util.keystr(pa)}: {na.dtype} != {nb.dtype}"
        assert np.array_equal(na, nb), f"{label}{jax.tree_util.keystr(pa)}: values differ"


def _sharding_diffs(a, b, *, equivalent_only: bool):
    """Paths where a and b disagree on placement. `equivalent_only` compares
    semantics (same devices, same shard layout); otherwise strict `==`."""
    out = []
    for (pa, va), (_, vb) in zip(_paths_and_leaves(a), _paths_and_leaves(b)):
        sa, sb = va.sharding, vb.sharding
        if equivalent_only:
            same = sa.is_equivalent_to(sb, np.ndim(va))
        else:
            same = sa == sb
        if not same:
            out.append((jax.tree_util.keystr(pa), str(sa), str(sb)))
    return out


def _specs(tree):
    return [str(v.sharding.spec) for _, v in _paths_and_leaves(tree)]


def _n_genuinely_sharded(tree):
    n = 0
    for _, v in _paths_and_leaves(tree):
        spec = getattr(v.sharding, "spec", None)
        if spec is not None and any(s is not None for s in spec):
            n += 1
    return n


def _expected_placement(ckpt, state_sharding, replicated):
    """The FULL placement the restore should produce: `state_sharding` for the
    train_state item (step/params/opt_state) and the MIXED params-item layout
    for the EMA item. `dataclasses.replace` on a tree of shardings re-runs
    TrainState's beartype-checked __init__ (`step: at.Int[...]` vs a
    NamedSharding), so it needs the same guard the production helper uses --
    which is itself an independent confirmation that the guard is required."""
    with at.disable_typechecking():
        return dataclasses.replace(
            state_sharding,
            ema_params=_params_item_sharding(
                state_sharding, ckpt.config.trainable_filter, replicated
            ),
        )


def _restore_old(ckpt, shape_tree, step=_SAVE_STEP):
    """The verbatim pre-change path: openpi's UNTOUCHED restore_state."""
    return _checkpoints.restore_state(ckpt.manager, shape_tree, None, step=step)


def _restore_new(ckpt, shape_tree, state_sharding, replicated, step=_SAVE_STEP, filt=None):
    return _restore_state_sharded(
        ckpt.manager,
        shape_tree,
        state_sharding,
        trainable_filter=ckpt.config.trainable_filter if filt is None else filt,
        replicated_sharding=replicated,
        step=step,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cfg():
    return _build_config()


@pytest.fixture(scope="module")
def mesh1():
    return _mesh(1)


@pytest.fixture(scope="module")
def mesh2():
    return _mesh(2)


@pytest.fixture(scope="module")
def ckpt1(tmp_path_factory, cfg, mesh1):
    """Saved under a 1-device mesh (everything replicated) -- the FSDP=1 run."""
    return _Ckpt(tmp_path_factory.mktemp("ckpt_m1"), cfg, mesh1)


@pytest.fixture(scope="module")
def ckpt2(tmp_path_factory, cfg, mesh2):
    """Saved under a 2-device mesh -- the FSDP=2 run."""
    return _Ckpt(tmp_path_factory.mktemp("ckpt_m2"), cfg, mesh2)


# --------------------------------------------------------------------------- #
# 0. Environment
# --------------------------------------------------------------------------- #
def test_two_devices_are_available_and_meshes_are_distinct(mesh1, mesh2):
    assert jax.device_count() >= 2
    assert mesh1.shape[sharding.FSDP_AXIS] == 1
    assert mesh2.shape[sharding.FSDP_AXIS] == 2
    assert len(mesh1.devices.flatten()) == 1 and len(mesh2.devices.flatten()) == 2


def test_the_fixture_has_BOTH_frozen_and_trainable_leaves(cfg, ckpt1):
    """Anti-vacuity for every params-item test: `_params_item_sharding` only
    differs from a uniform layout if the production `freeze_filter` actually
    splits this dummy tree. Reports the real counts/dtypes if it does not."""
    params = ckpt1.shape_tree.params
    n_train = len(jax.tree.leaves(nnx.filter_state(params, cfg.trainable_filter)))
    n_all = len(jax.tree.leaves(params))
    dtypes = sorted({str(v.dtype) for v in jax.tree.leaves(params)})
    assert 0 < n_train < n_all, (
        f"degenerate split: trainable={n_train} of {n_all} leaves, dtypes={dtypes}. "
        "The params-item layout tests would be vacuous."
    )
    # 19 trainable of 51 here -- the same split the REAL pi0.5 tree shows on GPU,
    # so the mixed params-item layout is genuinely exercised. Note the dtypes are
    # all float32: `init_train_state`'s "convert frozen params to bfloat16" step
    # (openpi `nnx_utils.state_map`) is a no-op under the pinned flax, which is an
    # openpi-side observation, NOT something this change touches. The split that
    # `_params_item_sharding` keys off is the FILTER, not the dtype.
    assert n_train == 19 and n_all == 51, f"fixture drifted: {n_train}/{n_all}"


def test_the_fixture_checkpoint_really_shards_a_big_leaf(ckpt2):
    """Guards against a vacuous suite: `fsdp_sharding` replicates anything under
    min_size_mbytes=4 or below 2-D, so without a >= 4 MiB >= 2-D leaf every
    'resharded' assertion would be trivially satisfied by replication."""
    shard_specs = [
        (jax.tree_util.keystr(p), np.prod(sd.shape) * np.dtype(sd.dtype).itemsize, sh)
        for (p, sd), (_, sh) in zip(
            _paths_and_leaves(ckpt2.shape_tree), _paths_and_leaves(ckpt2.state_sharding)
        )
    ]
    big_2d = [
        (path, nbytes, sh)
        for path, nbytes, sh in shard_specs
        if nbytes >= 4 * 2**20
    ]
    assert big_2d, "no >= 4 MiB leaf in the dummy TrainState; the suite would be vacuous"
    sharded = [t for t in big_2d if any(s is not None for s in t[2].spec)]
    assert sharded, f"no >= 4 MiB leaf got a non-replicated spec: {[(p, n) for p, n, _ in big_2d][:5]}"


# --------------------------------------------------------------------------- #
# 1. Restore args: every leaf carries an explicit sharding
# --------------------------------------------------------------------------- #
class _SpyManager:
    def __init__(self, inner):
        self._inner = inner
        self.captured = []

    def restore(self, step, *args, **kwargs):
        self.captured.append(kwargs.get("args"))
        return self._inner.restore(step, *args, **kwargs)


def test_no_leaf_falls_back_to_a_bare_RestoreArgs(ckpt1, mesh2):
    from orbax.checkpoint._src.serialization.type_handlers import (
        ArrayRestoreArgs,
        RestoreArgs,
    )

    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh2)
    spy = _SpyManager(ckpt1.manager)
    _restore_state_sharded(
        spy,
        shape_tree,
        state_sharding,
        trainable_filter=ckpt1.config.trainable_filter,
        replicated_sharding=replicated,
        step=_SAVE_STEP,
    )
    assert len(spy.captured) == 1
    composite = spy.captured[0]
    assert isinstance(composite, ocp.args.Composite)
    n_leaves = 0
    for item_name in ("train_state", "params"):
        item_args = composite[item_name]
        for path, ra in _paths_and_leaves(item_args.restore_args):
            n_leaves += 1
            assert isinstance(ra, RestoreArgs), f"{item_name}{jax.tree_util.keystr(path)}"
            assert isinstance(ra, ArrayRestoreArgs), (
                f"{item_name}{jax.tree_util.keystr(path)} fell back to a bare RestoreArgs "
                f"-> orbax would use the checkpoint's _sharding metadata for it"
            )
            assert ra.sharding is not None, f"{item_name}{jax.tree_util.keystr(path)}: sharding=None"
            assert ra.sharding.mesh == mesh2, (
                f"{item_name}{jax.tree_util.keystr(path)}: sharding is on the wrong mesh"
            )
    assert n_leaves > 50, f"suspiciously few restore-arg leaves: {n_leaves}"


def test_restore_args_treedef_matches_the_item_treedef(ckpt1, mesh2):
    """A treedef mismatch between item and restore_args is the failure mode a
    toy dict would hide; this uses the REAL TrainState (nnx.State /
    VariableState / optax opt_state / GraphDef, non-None ema_params)."""
    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh2)
    with at.disable_typechecking():
        ts_item, params_item = _checkpoints._split_params(shape_tree)
        ts_sharding, _ = _checkpoints._split_params(state_sharding)
        params_sharding = _params_item_sharding(
            state_sharding, ckpt1.config.trainable_filter, replicated
        )
        ts_args = construct_restore_args(ts_item, ts_sharding)
        params_args = construct_restore_args(
            {"params": params_item}, {"params": params_sharding}
        )
    assert jax.tree.structure(ts_args) == jax.tree.structure(ts_item)
    assert jax.tree.structure(params_args) == jax.tree.structure({"params": params_item})


def test_new_path_emits_no_sharding_fallback_warning(ckpt1, mesh1):
    """The secondary benefit claimed by the change record: orbax's
    'Sharding info not provided when restoring ... unsafe when restoring on a
    different topology' UserWarning (type_handlers.py:1251) disappears."""
    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _restore_new(ckpt1, shape_tree, state_sharding, replicated)
    hits = [w for w in caught if "Sharding info not provided" in str(w.message)]
    assert not hits, f"new path still warns: {[str(w.message) for w in hits]}"


def test_old_path_does_emit_the_sharding_fallback_warning(ckpt1):
    """Pins the pre-change behavior so the previous test is not vacuous."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _restore_old(ckpt1, ckpt1.shape_tree)
    hits = [w for w in caught if "Sharding info not provided" in str(w.message)]
    assert hits, "old path did not warn; the 'warning disappears' claim is unverifiable here"


# --------------------------------------------------------------------------- #
# 2. THE differential: M == N must be a no-op vs openpi's untouched restore_state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2])
def test_same_topology_restore_is_identical_to_openpi_restore_state(request, n):
    """Same process, same checkpoint, same mesh: the new helper and the verbatim
    pre-change `openpi.training.checkpoints.restore_state` must agree on
    treedef, on every value bit-for-bit, and on every leaf's placement -- for
    the train_state item AND the mixed replicated/FSDP params (EMA) item."""
    ckpt = request.getfixturevalue("ckpt1" if n == 1 else "ckpt2")
    mesh = request.getfixturevalue("mesh1" if n == 1 else "mesh2")
    shape_tree, state_sharding, replicated = ckpt.target_for(mesh)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = _restore_old(ckpt, shape_tree)
    new = _restore_new(ckpt, shape_tree, state_sharding, replicated)

    assert jax.tree.structure(old) == jax.tree.structure(new)
    _assert_values_identical(old, new, f"M==N={n} old-vs-new")
    assert old.ema_params is not None and new.ema_params is not None

    equiv = _sharding_diffs(old, new, equivalent_only=True)
    assert not equiv, (
        f"M==N={n}: placement CHANGED for {len(equiv)} leaf(s); the change is not a "
        f"no-op at equal topology. First 5: {equiv[:5]}"
    )
    strict = _sharding_diffs(old, new, equivalent_only=False)
    assert not strict, (
        f"M==N={n}: shardings are equivalent but not `==` for {len(strict)} leaf(s) "
        f"(report-worthy, not necessarily a defect). First 5: {strict[:5]}"
    )


def test_same_topology_two_device_restore_is_genuinely_sharded(ckpt2, mesh2):
    """Non-vacuity guard for the M==N=2 differential: if everything were
    replicated the placement comparison above would prove nothing."""
    shape_tree, state_sharding, replicated = ckpt2.target_for(mesh2)
    new = _restore_new(ckpt2, shape_tree, state_sharding, replicated)
    assert _n_genuinely_sharded(new) > 0, "M==N=2 restore is fully replicated"


# --------------------------------------------------------------------------- #
# 3. Cross-topology: forward (1 -> 2) and reverse (2 -> 1)
# --------------------------------------------------------------------------- #
def test_forward_reshard_one_device_checkpoint_onto_two_device_mesh(ckpt1, mesh2):
    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh2)
    restored = _restore_new(ckpt1, shape_tree, state_sharding, replicated)

    # (a) values bit-identical to what was saved
    _assert_values_identical(ckpt1.state_to_save, restored, "1->2")

    # (b) placement == the CURRENT mesh's request, for BOTH items
    want = _expected_placement(ckpt1, state_sharding, replicated)
    for (p, v), (_, s) in zip(_paths_and_leaves(restored), _paths_and_leaves(want)):
        assert v.sharding == s, f"1->2{jax.tree_util.keystr(p)}: {v.sharding} != {s}"

    # (c) the >= 4 MiB leaves really moved onto both devices
    assert _n_genuinely_sharded(restored.params) > 0
    devsets = {
        frozenset(d.id for d in v.sharding.device_set)
        for _, v in _paths_and_leaves(restored.params)
    }
    assert devsets == {frozenset({0, 1})}, f"unexpected device sets: {devsets}"


def test_reverse_reshard_two_device_checkpoint_onto_one_device_mesh(ckpt2, mesh1):
    shape_tree, state_sharding, replicated = ckpt2.target_for(mesh1)
    restored = _restore_new(ckpt2, shape_tree, state_sharding, replicated)
    _assert_values_identical(ckpt2.state_to_save, restored, "2->1")
    want = _expected_placement(ckpt2, state_sharding, replicated)
    for (p, v), (_, s) in zip(_paths_and_leaves(restored), _paths_and_leaves(want)):
        assert v.sharding == s, f"2->1{jax.tree_util.keystr(p)}: {v.sharding} != {s}"
    devsets = {
        frozenset(d.id for d in v.sharding.device_set)
        for _, v in _paths_and_leaves(restored.params)
    }
    assert devsets == {frozenset({0})}, f"2->1 left arrays on {devsets}"
    assert _n_genuinely_sharded(restored) == 0, "1-device mesh must be fully replicated"


def test_old_path_leaves_a_one_device_checkpoint_unusable_on_the_two_device_mesh(
    ckpt1, mesh2
):
    """The 1->2 failure BLAST-RADIUS.md describes: the old restore succeeds (the
    saved device id still exists) and the first jit wanting a genuinely sharded
    leaf raises. Reproduced here in-process; the new path does not raise."""
    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = _restore_old(ckpt1, shape_tree)

    # Find a leaf the CURRENT mesh wants genuinely sharded.
    target = None
    for (p, s), (_, v) in zip(
        _paths_and_leaves(state_sharding), _paths_and_leaves(old)
    ):
        if any(ax is not None for ax in s.spec):
            target = (p, s, v)
            break
    assert target is not None, "no genuinely-sharded leaf; test would be vacuous"
    _p, spec_sharding, stale_leaf = target
    assert frozenset(d.id for d in stale_leaf.sharding.device_set) == frozenset({0}), (
        "old path unexpectedly did NOT leave the leaf on the saved device subset"
    )

    f = jax.jit(lambda x: x + 0, in_shardings=spec_sharding, out_shardings=spec_sharding)
    with pytest.raises(ValueError, match="[Ss]harding"):
        jax.block_until_ready(f(stale_leaf))

    fresh = _restore_new(ckpt1, shape_tree, state_sharding, replicated)
    good_leaf = dict(_paths_and_leaves(fresh))[_p]
    jax.block_until_ready(f(good_leaf))  # must not raise


def test_fsdp_devices_one_on_a_two_device_allocation(ckpt1, cfg):
    """`fsdp_devices=1` with 2 GPUs: `make_mesh(1)` spans BOTH devices as a
    (2,1) mesh, so `fsdp_sharding` replicates everything across both. The old
    path pins the arrays to the SAVED device subset instead (correct, wasteful,
    and an extra broadcast at every jit); the new path replicates properly."""
    mesh_2x1 = jax.sharding.Mesh(
        np.array(jax.devices()[:2]).reshape(2, 1),
        (sharding.BATCH_AXIS, sharding.FSDP_AXIS),
    )
    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh_2x1)
    assert all(
        s.spec == _P() for _, s in _paths_and_leaves(state_sharding)
    ), "a (2,1) mesh must replicate everything (fsdp axis == 1)"
    restored = _restore_new(ckpt1, shape_tree, state_sharding, replicated)
    _assert_values_identical(ckpt1.state_to_save, restored, "fsdp1-on-2gpu")
    devsets = {
        frozenset(d.id for d in v.sharding.device_set)
        for _, v in _paths_and_leaves(restored)
    }
    assert devsets == {frozenset({0, 1})}, f"new path did not span both devices: {devsets}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = _restore_old(ckpt1, shape_tree)
    old_devsets = {
        frozenset(d.id for d in v.sharding.device_set) for _, v in _paths_and_leaves(old)
    }
    assert old_devsets == {frozenset({0})}, (
        f"expected the old path to pin to the saved device subset, got {old_devsets}"
    )


# --------------------------------------------------------------------------- #
# 4. `_split_params` parity and the typechecking contract
# --------------------------------------------------------------------------- #
def test_split_params_takes_the_same_branch_on_shape_and_sharding_trees(ckpt1, mesh1):
    shape_tree, state_sharding, _ = ckpt1.target_for(mesh1)
    with at.disable_typechecking():
        ts_shape, params_shape = _checkpoints._split_params(shape_tree)
        ts_sh, params_sh = _checkpoints._split_params(state_sharding)
    # ema branch on both (ema_params is not None on the resume-path eval_shape tree)
    assert shape_tree.ema_params is not None and state_sharding.ema_params is not None
    assert ts_shape.ema_params is None and ts_sh.ema_params is None
    assert ts_shape.params and ts_sh.params  # not blanked -> ema branch
    assert jax.tree.structure(ts_shape) == jax.tree.structure(ts_sh)
    assert jax.tree.structure(params_shape) == jax.tree.structure(params_sh)


def test_split_params_on_a_sharding_tree_raises_without_the_typecheck_guard(ckpt1, mesh1):
    """Pins why `_restore_state_sharded` needs `at.disable_typechecking()`:
    `dataclasses.replace` re-runs TrainState's beartype-checked __init__, and
    `step: at.Int[...]` rejects a NamedSharding."""
    import jaxtyping

    _, state_sharding, _ = ckpt1.target_for(mesh1)
    with pytest.raises((jaxtyping.TypeCheckError, TypeError, Exception)) as exc:
        _checkpoints._split_params(state_sharding)
    assert "step" in str(exc.value) or "Int" in str(exc.value), str(exc.value)[:400]
    # ... and does not raise inside the guard
    with at.disable_typechecking():
        _checkpoints._split_params(state_sharding)


# --------------------------------------------------------------------------- #
# 5. `_params_item_sharding` reproduces the save-side layout, leaf for leaf
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2])
def test_params_item_sharding_equals_what_save_checkpoint_actually_writes(request, cfg, n):
    """Compare against the real composed item: `save_checkpoint` builds it with
    `compose_full_params(train_state.params, device_put(ema, replicated),
    trainable_filter)`, so every leaf of `_params_item_sharding` must equal that
    leaf's actual `.sharding`."""
    ckpt = request.getfixturevalue("ckpt1" if n == 1 else "ckpt2")
    mesh = request.getfixturevalue("mesh1" if n == 1 else "mesh2")
    _, state_sharding, replicated = ckpt.target_for(mesh)
    predicted = _params_item_sharding(state_sharding, cfg.trainable_filter, replicated)
    actual = ckpt.state_to_save.ema_params
    lp, la = _paths_and_leaves(predicted), _paths_and_leaves(actual)
    assert len(lp) == len(la)
    for (pp, sp), (pa, va) in zip(lp, la):
        assert pp == pa
        assert sp == va.sharding, f"params item{jax.tree_util.keystr(pp)}: {sp} != {va.sharding}"


def test_params_item_sharding_is_genuinely_mixed_and_differs_from_uniform_fsdp(cfg, mesh2):
    """Non-vacuity: with the production 4 MiB threshold every TRAINABLE leaf of
    the dummy model is below it and replicates anyway, which would make
    'mirror the save' and 'uniform fsdp_sharding' indistinguishable. Forcing
    min_size_mbytes=0 separates them: frozen leaves stay FSDP-sharded, trainable
    leaves are replicated, and the two trees genuinely differ."""
    state = _build_train_state(cfg, with_ema=True)
    shape_tree = _shape_tree(state)
    uniform = sharding.fsdp_sharding(shape_tree, mesh2, min_size_mbytes=0)
    replicated = jax.sharding.NamedSharding(mesh2, _P())
    mixed = _params_item_sharding(uniform, cfg.trainable_filter, replicated)

    trainable_paths = {
        p for p, _ in _paths_and_leaves(
            nnx.filter_state(uniform.params, cfg.trainable_filter)
        )
    }
    assert trainable_paths, "no trainable leaves; test vacuous"
    n_diff = 0
    for (p, s_mixed), (_, s_uniform) in zip(
        _paths_and_leaves(mixed), _paths_and_leaves(uniform.params)
    ):
        if p in trainable_paths:
            assert s_mixed == replicated, f"trainable{jax.tree_util.keystr(p)} not replicated"
            if s_uniform != replicated:
                n_diff += 1
        else:
            assert s_mixed == s_uniform, f"frozen{jax.tree_util.keystr(p)} not FSDP-sharded"
    assert n_diff > 0, (
        "uniform fsdp and the mirrored mixed layout are identical even at "
        "min_size_mbytes=0 -- the mirror decision is untestable here"
    )


# --------------------------------------------------------------------------- #
# 6. Edge cases: no EMA, step=None, mismatched trainable_filter, bad structure
# --------------------------------------------------------------------------- #
def test_ema_params_none_branch_round_trips(tmp_path_factory, cfg, mesh1, mesh2):
    """`ema_decay=None` -> `_split_params` takes the `params` branch (train_state
    gets `params={}`). Unreachable in production (every registered config sets
    ema_decay, and FSL:447 would crash on a None ema_params), but the helper
    must still be correct -- and this is where the mirrored params-item layout
    is a deliberate MISmatch: the on-disk item is then the live params, written
    fully FSDP-sharded, while `_params_item_sharding` still asks for trainable
    leaves replicated. Values stay right; placement is what shifts."""
    no_ema = dataclasses.replace(cfg, ema_decay=None)
    ck = _Ckpt(tmp_path_factory.mktemp("ckpt_noema"), no_ema, mesh2, with_ema=False)
    assert ck.state_to_save.ema_params is None
    shape_tree, state_sharding, replicated = ck.target_for(mesh1)
    restored = _restore_new(ck, shape_tree, state_sharding, replicated)
    assert restored.ema_params is None
    _assert_values_identical(ck.state_to_save, restored, "no-ema 2->1")
    assert {frozenset(d.id for d in v.sharding.device_set)
            for _, v in _paths_and_leaves(restored)} == {frozenset({0})}


def test_ema_none_params_item_layout_is_replicated_for_trainable_leaves(cfg, mesh2):
    """Documents the asymmetry above explicitly, so a future change that makes
    the no-EMA path reachable trips here."""
    no_ema = dataclasses.replace(cfg, ema_decay=None)
    state = _build_train_state(no_ema, with_ema=False)
    uniform = sharding.fsdp_sharding(_shape_tree(state), mesh2, min_size_mbytes=0)
    replicated = jax.sharding.NamedSharding(mesh2, _P())
    mixed = _params_item_sharding(uniform, no_ema.trainable_filter, replicated)
    trainable_paths = {
        p for p, _ in _paths_and_leaves(nnx.filter_state(uniform.params, no_ema.trainable_filter))
    }
    replicated_trainable = [
        p for p, s in _paths_and_leaves(mixed) if p in trainable_paths and s == replicated
    ]
    assert len(replicated_trainable) == len(trainable_paths)


def test_step_none_restores_the_latest_step(ckpt1, mesh2):
    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh2)
    latest = _restore_new(ckpt1, shape_tree, state_sharding, replicated, step=None)
    pinned = _restore_new(ckpt1, shape_tree, state_sharding, replicated, step=_SAVE_STEP)
    _assert_values_identical(latest, pinned, "step=None vs step=7")
    assert int(latest.step) == _SAVE_STEP


def test_a_different_trainable_filter_at_restore_is_placement_only(ckpt2, cfg, mesh2):
    """Restoring with a filter that differs from the one used at save (e.g. a
    LoRA-aware filter after `backbone_lora` was toggled) changes only which
    leaves are asked for replicated -- never a value."""
    shape_tree, state_sharding, replicated = ckpt2.target_for(mesh2, min_size_mbytes=0)
    matched = _restore_new(ckpt2, shape_tree, state_sharding, replicated)
    # An adapter-aware filter; inert on a lora-less tree but a different object,
    # plus a deliberately wrong one (everything trainable).
    for alt in (nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*PaliGemma/img.*"))),
                nnx.Param):
        other = _restore_new(ckpt2, shape_tree, state_sharding, replicated, filt=alt)
        _assert_values_identical(matched, other, f"filter {alt}")


@pytest.mark.parametrize("other_action_dim", [5, 3])
def test_new_path_is_STRICTER_than_old_on_a_target_vs_stored_shape_mismatch(
    ckpt1, mesh1, other_action_dim
):
    """BEHAVIOR CHANGE, deliberate, surfaced rather than buried.

    This is the one place the change is NOT a no-op at equal topology. The old
    path left a bare `RestoreArgs` per leaf, so orbax never compared the target's
    shape to the stored one: a target asking for `action_out_proj` at `(5,)`
    restored SILENTLY as the stored `(4,)`, straight into a tree whose jits were
    compiled for `(5,)`. The new path's `construct_restore_args` emits
    `ArrayRestoreArgs(global_shape=<target shape>)`, and orbax's default
    `strict=True` rejects the mismatch.

    Judged an IMPROVEMENT: a silently stored-shape restore is exactly this
    codebase's signature failure class (a run that trains and means nothing).
    Only per-leaf SHAPE moved; leaf-set mismatches already raised on both paths
    (see the next test). `action_dim` is the knob that actually changes a param
    shape -- `action_horizon` and `max_token_len` do not, so changing those
    across a resume is unaffected.
    """
    other_cfg = _build_config(action_dim=other_action_dim)
    other_state = _build_train_state(other_cfg, with_ema=True)
    other_shape = _shape_tree(other_state)
    other_sharding = sharding.fsdp_sharding(other_shape, mesh1)
    replicated = jax.sharding.NamedSharding(mesh1, _P())

    # OLD: returns, and the leaf carries the STORED shape, not the target's.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = _restore_old(ckpt1, other_shape)
    stored = dict(_paths_and_leaves(ckpt1.state_to_save))
    got = dict(_paths_and_leaves(old))
    probe = [p for p in got if "action_out_proj" in jax.tree_util.keystr(p)]
    assert probe, "no action_out_proj leaf; test would be vacuous"
    for p in probe:
        assert np.shape(got[p]) == np.shape(stored[p]), (
            "old path no longer returns the stored shape; re-characterize"
        )
        assert np.shape(got[p]) != (other_action_dim,) and np.shape(got[p])[-1] != other_action_dim

    # NEW: raises, naming both shapes.
    with pytest.raises(ValueError, match="not compatible with the stored shape"):
        _restore_state_sharded(
            ckpt1.manager,
            other_shape,
            other_sharding,
            trainable_filter=other_cfg.trainable_filter,
            replicated_sharding=replicated,
            step=_SAVE_STEP,
        )


def test_leaf_set_mismatches_raise_identically_on_both_paths(ckpt1, mesh1):
    """Bounds the behavior change above: an EXTRA or a MISSING leaf in the
    target is a tree-structure mismatch, which orbax rejects before restore args
    matter -- so both paths raise, with the same message class. Only per-leaf
    shape strictness changed."""
    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh1)
    flat = nnx.to_flat_state(shape_tree.params)

    extra = list(flat) + [(("__bogus__", "value"), jax.ShapeDtypeStruct((4, 4), jnp.float32))]
    fewer = [kv for kv in flat if "action_out_proj" not in "/".join(str(x) for x in kv[0])]
    assert len(fewer) < len(flat), "nothing dropped; test would be vacuous"

    for label, leaves in (("extra-leaf", extra), ("missing-leaf", fewer)):
        with at.disable_typechecking():
            tgt = dataclasses.replace(
                shape_tree,
                params=nnx.from_flat_state(leaves),
                ema_params=nnx.from_flat_state(leaves),
            )
            tgt_sharding = sharding.fsdp_sharding(tgt, mesh1)
        with pytest.raises(ValueError, match="tree structures do not match"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _restore_old(ckpt1, tgt)
        with pytest.raises(ValueError, match="tree structures do not match"):
            _restore_state_sharded(
                ckpt1.manager,
                tgt,
                tgt_sharding,
                trainable_filter=ckpt1.config.trainable_filter,
                replicated_sharding=replicated,
                step=_SAVE_STEP,
            )


def test_a_target_dtype_difference_is_CAST_by_the_new_path_and_ignored_by_the_old(
    ckpt1, mesh1
):
    """Second (much smaller) behavior delta, isolated: `construct_restore_args`
    also sets `dtype=<target dtype>`, so a leaf whose target dtype differs from
    the stored one is CAST on the new path, where the old path returned the
    stored dtype. Structure is untouched here, so orbax accepts both.

    Not reachable from any config: the only thing that flips a leaf's dtype is
    `freeze_filter` (frozen -> bf16), and changing that also changes the
    trainable set and therefore the `opt_state` tree, which BOTH paths reject as
    a structure mismatch. Pinned so the asymmetry is on record."""
    shape_tree, state_sharding, replicated = ckpt1.target_for(mesh1)
    flat = nnx.to_flat_state(shape_tree.params)
    # `nnx.to_flat_state` yields (path, VariableState); the ShapeDtypeStruct is
    # the VariableState's `.value`.
    dtypes_present = sorted({str(vs.value.dtype) for _, vs in flat})
    target_path = next(
        (k for k, vs in flat if str(vs.value.dtype) == "bfloat16"), None
    )
    if target_path is None:
        pytest.skip(f"no bf16 (frozen) leaf in this tree; dtypes present = {dtypes_present}")
    recast = [
        (
            k,
            v.replace(jax.ShapeDtypeStruct(v.value.shape, jnp.float32))
            if k == target_path
            else v,
        )
        for k, v in flat
    ]
    with at.disable_typechecking():
        tgt = dataclasses.replace(shape_tree, params=nnx.from_flat_state(recast))
        tgt_sharding = sharding.fsdp_sharding(tgt, mesh1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = _restore_old(ckpt1, tgt)
    new = _restore_state_sharded(
        ckpt1.manager,
        tgt,
        tgt_sharding,
        trainable_filter=ckpt1.config.trainable_filter,
        replicated_sharding=replicated,
        step=_SAVE_STEP,
    )
    old_leaf = dict(nnx.to_flat_state(old.params))[target_path]
    new_leaf = dict(nnx.to_flat_state(new.params))[target_path]
    assert old_leaf.value.dtype == jnp.bfloat16, "old path no longer returns the stored dtype"
    assert new_leaf.value.dtype == jnp.float32, "new path no longer honors the target dtype"
    assert np.array_equal(
        np.asarray(jax.device_get(old_leaf.value)).astype(np.float32),
        np.asarray(jax.device_get(new_leaf.value)),
    ), "the cast changed values beyond the widening itself"
