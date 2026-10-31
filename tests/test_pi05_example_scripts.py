"""The pi05 example recipes: scripts/{ogpo,awr}_pi05/molmo_{box138,mt16}.sh.

Each script runs for real in a sandbox, so the shell logic (path checks, override loop, qvr suffix, optional flags) is what is tested:
a fake checkout (empty .venv-molmo/bin/activate and ogpo/main_pi05.py), fake asset trees, and a fake `python` first on PATH that
records its argv instead of training. The recorded flags then go through the real config loader, `OGPOPi05Agent.create` and
`main_pi05.main` (both stubbed at the model / env boundary, as in tests/test_pi05_awr.py).

The two scripts that set the recipes up (scripts/ogpo/setup_molmo_venv.sh, scripts/ogpo/download_molmo_assets.sbatch) are run the same way
with a fake uv / gsutil / python, so their arguments and fail-fast paths are checked without installing or downloading anything.

What this proves: the flags exist, the config resolves and is accepted, the OGPO scripts carry exactly the hyperparameters of the
most recent molmo mt144 run (seed 1) except for the scale / path flags, and each AWR script is its OGPO twin plus the AWR delta.
What it cannot prove: that a run starts or learns (GPU, MolmoSpaces assets and the pi05_droid_jointpos checkpoint are not here).

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_pi05_example_scripts.py -q
"""
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from ogpo.agents.ogpo_pi05 import OGPOPi05Agent  # noqa: E402
from ogpo.utils.config_loader import config_to_agent_dict, load_config, parse_cli_args  # noqa: E402
from tests.test_pi05_awr import _Reached, _main_harness, stop_at_load  # noqa: E402,F401  (stop_at_load is a fixture)

RECIPES = {(algo, scale): f'scripts/{algo}_pi05/molmo_{scale}.sh' for algo in ('ogpo', 'awr') for scale in ('box138', 'mt16')}
N_TASKS = {'box138': 1, 'mt16': 16}
PATH_VARS = ('SAVE_DIR', 'OPENPI_CACHE', 'MLSPACES_ASSETS_DIR', 'MLSPACES_CACHE_DIR', 'NLTK_DATA')

# Flags of paper_expts_final/molmo_pi05_mt144_ogpo_s1.sh (the most recent mt144 run: seed 1, 4 x 96 GB GPUs) as the script resolves
# them, recorded from its own execution in this sandbox with the node-local buffer off. 2026-11-01.
MT144 = {
    'agent.act_max': 'inf',
    'agent.act_min': '-inf',
    'agent.actor_tau': '0.005',
    'agent.adv_strategy': 'conservative',
    'agent.batch_size': '512',
    'agent.bc_coeff': '1.0',
    'agent.best_of_n': '32',
    'agent.clip_epsilon': '0.01',
    'agent.clip_grad_norm': '1000.0',
    'agent.clip_intermediate_actions': 'false',
    'agent.constant_noise_std': '0.1',
    'agent.critic_scheduler': 'constant',
    'agent.critic_weight_decay': '1e-6',
    'agent.discount': '0.995',
    'agent.entropy_coeff': '0.0',
    'agent.error_correct_sde_to_ode': 'true',
    'agent.flow_steps': '10',
    'agent.ft_flow_steps': '10',
    'agent.grad_accum_steps': '1',
    'agent.grad_diagnostics': 'false',
    'agent.group_num_samples': '32',
    'agent.horizon_length': '5',
    'agent.layer_norm': 'true',
    'agent.num_qs': '10',
    'agent.pi05_action_horizon': '15',
    'agent.pi05_bon_agg': 'subsample',
    'agent.pi05_bronet_depth': '2',
    'agent.pi05_bronet_hidden_dim': '1024',
    'agent.pi05_clip_grad_norm': '1.0',
    'agent.pi05_critic_arch': 'bronet_vla',
    'agent.pi05_critic_pooling': 'all',
    'agent.pi05_drop_empty_camera': 'true',
    'agent.pi05_lr': '2.5e-5',
    'agent.pi05_robot': 'droid',
    'agent.pi05_weight_decay': '1e-10',
    'agent.ppo_batch_size': '32',
    'agent.ppo_lr': '1e-4',
    'agent.q_agg': 'subsample',
    'agent.q_variance_reduction': 'true',
    'agent.q_vr_num_samples': '16',
    'agent.q_vr_reduction': 'median',
    'agent.tau': '0.05',
    'agent.use_bc_regularization': 'true',
    'agent.use_constant_noise': 'false',
    'agent.use_success_buffer_q': 'true',
    'agent.use_tapered_noise': 'true',
    'agent.value_hidden_dims': '[1024,1024,1024]',
    'algo': 'ogpo',
    'env.post_success_steps': '0',
    'env.vector_envs': 'true',
    'env.worker_egl_devices': 'auto',
    'eval.eval_episodes': '10',
    'experiment.log': 'true',
    'experiment.project': 'OGPO_molmo',
    'intervals.eval_interval': '2000',
    'intervals.log_interval': '1000',
    'success_buffer.batch_size': '512',
    'success_buffer.bc_batch_size': '512',
    'success_buffer.use_success_buffer': 'true',
    'training.collection_mode': 'rounds',
    'training.gpu_mode': 'multi',
    'training.num_rounds': '10',
    'training.rollouts_per_round': '20',
    'training.task_balance': 'none',
    'training.updates_per_round': '1000',
    'training.zero_actor_state': 'false',
}
# What may differ from it, by design: the seed, names, task set and everything that scales with it, and the paths.
SCALE_FLAGS = {'experiment.seed', 'experiment.run_group', 'experiment.run_name', 'experiment.wandb_name', 'experiment.save_dir',
               'env.name', 'env.parallel_envs', 'env.inference_batch', 'buffer.size', 'agent.pi05_checkpoint', 'agent.pi05_tokenizer'}
AWR_ONLY = {'training.awr_value_steps', 'training.awr_value_refreshes', 'agent.pi05_actor_objective', 'agent.use_value_fn',
            'agent.v_training_mode', 'agent.v_ensemble_size', 'agent.v_agg', 'agent.awr_beta', 'agent.awr_weight_max',
            'agent.awr_lambda', 'agent.awr_normalize_adv'}
# flags whose value the AWR recipe changes relative to its OGPO twin (the LIBERO AWR recipe's delta)
AWR_CHANGED = {'experiment.run_group', 'experiment.run_name', 'experiment.wandb_name', 'agent.best_of_n', 'agent.bc_coeff',
               'agent.use_bc_regularization', 'agent.use_success_buffer_q', 'agent.q_variance_reduction'}
# script-only keys main_pi05 reads with .get()/[] from the CLI (pre-existing; the YAML lacks them)
SCRIPT_ONLY = {'training.collection_mode', 'training.num_rounds', 'training.rollouts_per_round', 'training.updates_per_round',
               'buffer.bc_horizon', 'success_buffer.bc_batch_size', 'training.task_balance', 'env.eval_name',
               'agent.grad_diagnostics', 'agent.donate_agent', 'agent.use_tapered_noise'}


def _assets(base):
    """Fake pi05 checkpoint / tokenizer / MolmoSpaces trees. -> env dict of the five required paths."""
    openpi = base / 'openpi'
    ckpt = openpi / 'openpi-assets' / 'checkpoints' / 'pi05_droid_jointpos'
    (ckpt / 'params').mkdir(parents=True)
    (ckpt / 'assets' / 'droid').mkdir(parents=True)
    (ckpt / 'assets' / 'droid' / 'norm_stats.json').write_text('{}')
    (openpi / 'big_vision').mkdir()
    (openpi / 'big_vision' / 'paligemma_tokenizer.model').write_text('')
    mls = base / 'mlspaces'
    (mls / 'benchmarks' / 'molmospaces-bench-v1').mkdir(parents=True)
    (base / 'nltk').mkdir()
    (base / 'cache').mkdir()
    (base / 'save').mkdir()
    return {'SAVE_DIR': str(base / 'save'), 'OPENPI_CACHE': str(openpi), 'MLSPACES_ASSETS_DIR': str(mls),
            'MLSPACES_CACHE_DIR': str(base / 'cache'), 'NLTK_DATA': str(base / 'nltk')}


def run_recipe(base, src, rel, *, env=None, args=(), activate='', gpu=None, py_rc=0, venv=True, spool=False):
    """Run the script `src` (copied to <checkout>/`rel`) with a fake python. -> (CompletedProcess, argv list or None, checkout).
    The fake python also records its cwd and environment (see `launch_env`). activate: extra lines for .venv-molmo/bin/activate;
    gpu: what a fake `nvidia-smi` reports; py_rc: the fake python's exit code; venv=False: no .venv-molmo in the checkout;
    spool=True: the script runs from a copy outside the checkout (as sbatch does) with cwd inside it."""
    root = base / 'checkout'
    dst = (base / 'spool' / os.path.basename(rel)) if spool else (root / rel)
    dst.parent.mkdir(parents=True)
    root.mkdir(exist_ok=True)
    shutil.copy2(src, dst)
    if venv:
        (root / '.venv-molmo' / 'bin').mkdir(parents=True)
        (root / '.venv-molmo' / 'bin' / 'activate').write_text('export VENV_ACTIVE=1\n' + activate)
    (root / 'ogpo').mkdir()
    (root / 'ogpo' / 'main_pi05.py').write_text('')  # even if the fake python were bypassed, this is a no-op
    bindir = base / 'bin'
    bindir.mkdir()
    fake = bindir / 'python'
    fake.write_text('#!/bin/bash\nprintf \'%s\\n\' "$@" > "$ARGV_OUT"\n{ pwd -P; env; } > "$ARGV_OUT.env"\nexit ' + str(py_rc) + '\n')
    fake.chmod(0o755)
    if gpu:
        smi = bindir / 'nvidia-smi'
        smi.write_text(f'#!/bin/bash\necho "NVIDIA {gpu}"\n')
        smi.chmod(0o755)
    cwd = (root / 'sub') if spool else (base / 'cwd')
    cwd.mkdir()
    out = base / 'argv.txt'
    full_env = {'PATH': f'{bindir}:/usr/bin:/bin', 'HOME': str(base), 'ARGV_OUT': str(out), **(env if env is not None else _assets(base))}
    r = subprocess.run(['bash', str(dst), *args], cwd=cwd, env=full_env, capture_output=True, text=True, timeout=60)
    argv = out.read_text().splitlines() if out.exists() else None
    return r, argv, root


def launch_env(base):
    """(cwd, environment) the fake python saw in run_recipe(base, ...)."""
    lines = (base / 'argv.txt.env').read_text().splitlines()
    return lines[0], dict(l.split('=', 1) for l in lines[1:] if '=' in l)


def flags_of(argv):
    assert argv[0] == 'ogpo/main_pi05.py', argv[:2]
    d = {}
    for a in argv[1:]:
        assert a.startswith('--') and '=' in a, a
        k, v = a[2:].split('=', 1)
        assert k not in d, f'flag passed twice: {k}'
        d[k] = v
    return d


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """{(algo, scale): dict(flags, env, cwd, root, assets)} from one sandbox run each."""
    out = {}
    env = _assets(tmp_path_factory.mktemp('assets'))  # one set of paths for all four, so the path flags agree between twins
    for key, rel in RECIPES.items():
        base = tmp_path_factory.mktemp('_'.join(key))
        r, argv, root = run_recipe(base, os.path.join(ROOT, rel), rel, env=env)
        assert r.returncode == 0, (rel, r.stdout[-600:], r.stderr[-600:])
        cwd, seen = launch_env(base)
        out[key] = dict(flags=flags_of(argv), env=seen, cwd=cwd, root=str(root.resolve()), assets=env)
    return out


@pytest.fixture(scope='module')
def recipes(runs):
    """{(algo, scale): flags dict}"""
    return {k: v['flags'] for k, v in runs.items()}


# ---------------------------------------------------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('key', list(RECIPES))
def test_script_is_executable_and_valid_bash(key):
    path = os.path.join(ROOT, RECIPES[key])
    assert os.access(path, os.X_OK)
    assert subprocess.run(['bash', '-n', path], capture_output=True, text=True).returncode == 0


@pytest.mark.parametrize('key', list(RECIPES))
def test_no_personal_or_cluster_paths(key):
    text = open(os.path.join(ROOT, RECIPES[key])).read()
    for bad in ('/data/user_data', '/home/', '/scratch/', 'maxlab', '--partition=', '--qos='):
        assert bad not in text, bad


@pytest.mark.parametrize('var', PATH_VARS)
def test_fails_fast_when_a_required_path_is_unset(tmp_path, var):
    rel = RECIPES[('ogpo', 'mt16')]
    env = _assets(tmp_path)
    del env[var]
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, env=env)
    assert r.returncode != 0 and f'ERROR: {var} is not set' in r.stdout + r.stderr
    assert argv is None  # never reached python


def test_path_can_be_passed_as_an_override(tmp_path):
    rel = RECIPES[('ogpo', 'box138')]
    env = _assets(tmp_path)
    save = env.pop('SAVE_DIR')
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, env=env, args=(f'--SAVE_DIR={save}',))
    assert r.returncode == 0 and flags_of(argv)['experiment.save_dir'] == save


def test_accepts_the_override_set_a_remote_launcher_passes(tmp_path):
    """A launcher on a remote node runs `bash <script> --device --gpu_mode --seed --SAVE_DIR --pi05_checkpoint --pi05_tokenizer`
    with no OPENPI_CACHE; the MolmoSpaces paths come from its environment."""
    rel = RECIPES[('awr', 'mt16')]
    env = _assets(tmp_path)
    openpi = env.pop('OPENPI_CACHE')
    save = env.pop('SAVE_DIR')
    ckpt = f'{openpi}/openpi-assets/checkpoints/pi05_droid_jointpos'
    tok = f'{openpi}/big_vision/paligemma_tokenizer.model'
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, env=env,
                            args=('--device=4,5,6,7', '--gpu_mode=multi', '--seed=2', f'--SAVE_DIR={save}',
                                  f'--pi05_checkpoint={ckpt}', f'--pi05_tokenizer={tok}'))
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    f = flags_of(argv)
    assert (f['experiment.seed'], f['experiment.save_dir'], f['agent.pi05_checkpoint'], f['agent.pi05_tokenizer']) == ('2', save, ckpt, tok)
    assert f['training.gpu_mode'] == 'multi'


def test_fails_fast_when_the_checkpoint_is_missing(tmp_path):
    rel = RECIPES[('awr', 'box138')]
    env = _assets(tmp_path)
    os.remove(os.path.join(env['OPENPI_CACHE'], 'big_vision', 'paligemma_tokenizer.model'))
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, env=env)
    assert r.returncode != 0 and 'ERROR: missing' in r.stdout + r.stderr and argv is None


def _run_name(algo, scale):
    return f'ogpo_molmo_{scale}_pi05_bronet_vla_qvr16' if algo == 'ogpo' else f'awr_molmo_{scale}_pi05_bronet_vla'


@pytest.mark.parametrize('key', list(RECIPES))
def test_override_changes_the_flag_and_node_local_buffer_adds_storage_dir(tmp_path, key):
    rel = RECIPES[key]
    env = _assets(tmp_path)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, env=env,
                            args=('--seed=3', f'--buffer_storage_root={scratch}', '--num_rounds=2'))
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    f = flags_of(argv)
    assert f['experiment.seed'] == '3' and f['training.num_rounds'] == '2'
    assert f['buffer.storage_dir'].startswith(f'{scratch}/ogpo_rb_{_run_name(*key)}_s3_')
    assert not os.path.exists(f['buffer.storage_dir'])  # the EXIT trap removed it


# ---------------------------------------------------------------------------------------------------------------------
# placement: what the script sets up around python, the SBATCH header, the fail-fast checks
# ---------------------------------------------------------------------------------------------------------------------

def _sbatch(path):
    d = {}
    for line in open(path):
        if line.startswith('#SBATCH '):
            k, v = line[len('#SBATCH --'):].strip().split('=', 1)
            d[k] = v
    return d


@pytest.mark.parametrize('key', list(RECIPES))
def test_sbatch_header(key):
    algo, scale = key
    assert _sbatch(os.path.join(ROOT, RECIPES[key])) == {
        'job-name': f'{algo}_pi05_molmo_{scale}', 'gres': 'gpu:4', 'cpus-per-task': {'box138': '32', 'mt16': '64'}[scale],
        'mem': '400G', 'time': '48:00:00', 'output': 'logs/%x_%j.out', 'error': 'logs/%x_%j.out'}


@pytest.mark.parametrize('key', list(RECIPES))
def test_launch_environment(runs, key):
    r = runs[key]
    env = r['env']
    assert r['cwd'] == r['root']                                       # the script cd's into the checkout
    assert env['VENV_ACTIVE'] == '1' and env['PYTHONPATH'].split(':')[0] == r['root']
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3' and env['XLA_PYTHON_CLIENT_MEM_FRACTION'] == '0.70'
    assert (env['MUJOCO_GL'], env['PYOPENGL_PLATFORM'], env['PYTHONUNBUFFERED']) == ('egl', 'egl', '1')
    for v in ('MLSPACES_ASSETS_DIR', 'MLSPACES_CACHE_DIR', 'NLTK_DATA'):
        assert env[v] == r['assets'][v]
    assert 'NCCL_P2P_DISABLE' not in env


@pytest.mark.parametrize('key', list(RECIPES))
def test_paths_given_as_overrides_are_exported_to_the_trainer(tmp_path, key):
    """Passed as --VAR=..., not in the environment: the script must export them for the molmo env's workers."""
    rel = RECIPES[key]
    assets = _assets(tmp_path)
    args = tuple(f'--{v}={assets[v]}' for v in ('MLSPACES_ASSETS_DIR', 'MLSPACES_CACHE_DIR', 'NLTK_DATA', 'SAVE_DIR', 'OPENPI_CACHE'))
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, env={}, args=args)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    seen = launch_env(tmp_path)[1]
    for v in ('MLSPACES_ASSETS_DIR', 'MLSPACES_CACHE_DIR', 'NLTK_DATA'):
        assert seen.get(v) == assets[v], v


@pytest.mark.parametrize('key', list(RECIPES))
def test_run_and_group_names(recipes, key):
    algo, scale = key
    f = recipes[key]
    assert f['experiment.run_name'] == f['experiment.wandb_name'] == _run_name(algo, scale)
    assert f['experiment.run_group'] == f'molmo_{scale}_pi05_{algo}' and f['experiment.project'] == 'OGPO_molmo'


@pytest.mark.parametrize('key', list(RECIPES))
@pytest.mark.parametrize('gpu,expect', [('L40S', '1'), ('H100', None)])
def test_nccl_p2p_is_disabled_only_on_l40s(tmp_path, key, gpu, expect):
    rel = RECIPES[key]
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, gpu=gpu)
    assert r.returncode == 0
    assert launch_env(tmp_path)[1].get('NCCL_P2P_DISABLE') == expect


def test_runs_from_a_spooled_copy_outside_the_checkout(tmp_path):
    """sbatch runs a copy from /var/spool: the script walks up from the submit dir to find the checkout."""
    rel = RECIPES[('awr', 'mt16')]
    r, argv, root = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, spool=True)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    cwd, env = launch_env(tmp_path)
    assert cwd == str(root.resolve()) and env['VENV_ACTIVE'] == '1' and argv is not None


@pytest.mark.parametrize('what', ['ckpt_params', 'norm_stats', 'tokenizer', 'bench', 'nltk'])
@pytest.mark.parametrize('key', [('ogpo', 'box138'), ('awr', 'mt16')])
def test_fails_fast_when_an_asset_is_missing(tmp_path, key, what):
    rel = RECIPES[key]
    env = _assets(tmp_path)
    ckpt = os.path.join(env['OPENPI_CACHE'], 'openpi-assets', 'checkpoints', 'pi05_droid_jointpos')
    victim = {'ckpt_params': os.path.join(ckpt, 'params'), 'norm_stats': os.path.join(ckpt, 'assets', 'droid', 'norm_stats.json'),
              'tokenizer': os.path.join(env['OPENPI_CACHE'], 'big_vision', 'paligemma_tokenizer.model'),
              'bench': os.path.join(env['MLSPACES_ASSETS_DIR'], 'benchmarks', 'molmospaces-bench-v1'), 'nltk': env['NLTK_DATA']}[what]
    shutil.rmtree(victim) if os.path.isdir(victim) else os.remove(victim)
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, env=env)
    assert r.returncode != 0 and f'ERROR: missing {victim}' in r.stdout + r.stderr and argv is None


@pytest.mark.parametrize('key', list(RECIPES))
def test_worker_count_must_fit_the_cpu_allocation(tmp_path, key):
    rel = RECIPES[key]
    workers = N_TASKS[key[1]] * {'box138': 8, 'mt16': 2}[key[1]]
    (tmp_path / 'tight').mkdir()
    r, argv, _ = run_recipe(tmp_path / 'tight', os.path.join(ROOT, rel), rel,
                            env={**_assets(tmp_path / 'tight'), 'SLURM_CPUS_PER_TASK': str(workers + 1)})
    assert r.returncode != 0 and f'{workers} molmo worker processes need cpus-per-task >= {workers + 2}' in r.stdout + r.stderr
    assert argv is None
    (tmp_path / 'fits').mkdir()
    r, argv, _ = run_recipe(tmp_path / 'fits', os.path.join(ROOT, rel), rel,
                            env={**_assets(tmp_path / 'fits'), 'SLURM_CPUS_PER_TASK': str(workers + 2)})
    assert r.returncode == 0 and argv is not None


def test_a_checkpoint_override_alone_still_needs_openpi_cache_for_the_tokenizer(tmp_path):
    rel = RECIPES[('ogpo', 'mt16')]
    env = _assets(tmp_path)
    openpi = env.pop('OPENPI_CACHE')
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, env=env,
                            args=(f'--pi05_checkpoint={openpi}/openpi-assets/checkpoints/pi05_droid_jointpos',))
    assert r.returncode != 0 and 'ERROR: OPENPI_CACHE is not set' in r.stdout + r.stderr and argv is None


def test_stops_when_sourcing_the_venv_fails(tmp_path):
    rel = RECIPES[('awr', 'box138')]
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, activate='false\n')
    assert r.returncode != 0 and argv is None  # set -e


def test_stops_without_a_molmo_venv(tmp_path):
    rel = RECIPES[('ogpo', 'box138')]
    r, argv, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, venv=False)
    assert r.returncode != 0 and '.venv-molmo not found' in r.stdout + r.stderr and argv is None


def test_the_trainer_exit_code_is_the_scripts(tmp_path):
    rel = RECIPES[('ogpo', 'box138')]
    r, _, _ = run_recipe(tmp_path, os.path.join(ROOT, rel), rel, py_rc=3)
    assert r.returncode == 3


# ---------------------------------------------------------------------------------------------------------------------
# the flags
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('key', list(RECIPES))
def test_flags_exist_in_the_yaml_or_are_known_script_only_keys(recipes, key):
    cfg = load_config(*parse_cli_args(['--algo=ogpo']))
    assert recipes[key]['algo'] == 'ogpo'  # the only flag without a section
    missing = {k for k in recipes[key] if '.' in k and k.split('.', 1)[1] not in cfg.get(k.split('.', 1)[0], {})}
    assert all(m.split('.')[1].startswith('pi05_') or m in SCRIPT_ONLY for m in missing), sorted(missing)
    if key[0] == 'awr':
        assert not (AWR_ONLY & missing), sorted(AWR_ONLY & missing)  # the AWR keys are real YAML keys


@pytest.mark.parametrize('scale', ['box138', 'mt16'])
def test_ogpo_recipe_carries_the_latest_mt144_hyperparameters(recipes, scale):
    f = recipes[('ogpo', scale)]
    assert MT144, 'MT144 not filled in'
    assert SCALE_FLAGS <= set(f)
    rest = set(f) - SCALE_FLAGS
    assert rest == set(MT144), sorted(rest ^ set(MT144))  # same flags, none added, none dropped
    off = {k: (MT144[k], f[k]) for k in MT144 if f[k] != MT144[k]}
    assert not off, off
    assert f['experiment.seed'] == '0'


@pytest.mark.parametrize('scale,env_n,pe,ib,buf', [('box138', 1, '8', '8', '100000'), ('mt16', 16, '2', '32', '1200000')])
@pytest.mark.parametrize('algo', ['ogpo', 'awr'])
def test_scale_flags_per_recipe(recipes, algo, scale, env_n, pe, ib, buf):
    f = recipes[(algo, scale)]
    names = f['env.name'].split(',')
    assert len(names) == len(set(names)) == env_n and all(n.startswith('molmo_pi05-') for n in names)
    assert (f['env.parallel_envs'], f['env.inference_batch'], f['buffer.size']) == (pe, ib, buf)
    assert int(f['env.inference_batch']) == env_n * int(f['env.parallel_envs'])  # one row per worker slot
    assert f['env.vector_envs'] == 'true' and f['training.collection_mode'] == 'rounds'
    if scale == 'box138':
        assert names == ['molmo_pi05-138']
    else:
        assert names == [f'molmo_pi05-{i}' for i in (267, 366, 138, 143, 176, 373, 93, 394, 48, 185, 24, 62, 4, 260, 221, 348)]


@pytest.mark.parametrize('scale', ['box138', 'mt16'])
def test_awr_is_its_ogpo_twin_plus_the_awr_delta(recipes, scale):
    o, a = recipes[('ogpo', scale)], recipes[('awr', scale)]
    assert set(a) - set(o) == AWR_ONLY and set(o) <= set(a)
    changed = {k for k in o if o[k] != a[k]}
    assert changed == AWR_CHANGED, sorted(changed ^ AWR_CHANGED)
    assert (a['agent.best_of_n'], a['agent.bc_coeff'], a['agent.use_bc_regularization'], a['agent.use_success_buffer_q'],
            a['agent.q_variance_reduction']) == ('1', '0.0', 'false', 'false', 'false')


@pytest.mark.parametrize('algo', ['ogpo', 'awr'])
def test_single_task_and_mt16_differ_only_in_scale_flags(recipes, algo):
    s, m = recipes[(algo, 'box138')], recipes[(algo, 'mt16')]
    assert set(s) == set(m)
    diff = {k for k in s if s[k] != m[k]}
    assert diff == {'env.name', 'env.parallel_envs', 'env.inference_batch', 'buffer.size', 'experiment.run_group',
                    'experiment.run_name', 'experiment.wandb_name'}, sorted(diff)


def test_awr_config_resolves(recipes):
    for scale in ('box138', 'mt16'):
        cfg = load_config(*parse_cli_args([f'--{k}={v}' for k, v in recipes[('awr', scale)].items()]))
        ac = config_to_agent_dict(cfg)
        assert ac['pi05_actor_objective'] == 'awr' and ac['use_value_fn'] is True
        assert ac['v_training_mode'] == 'lambda_return' and ac['v_agg'] == 'mean' and ac['v_ensemble_size'] == 2
        assert ac['awr_beta'] == pytest.approx(0.05) and ac['awr_weight_max'] == pytest.approx(20.0)
        assert ac['awr_lambda'] == pytest.approx(0.95) and ac['awr_mode'] == 'symmetric'
        assert ac['use_bc_regularization'] is False and ac['q_variance_reduction'] is False
        assert cfg['training']['awr_value_steps'] == 200 and cfg['training']['awr_value_refreshes'] == 1
        assert cfg['training']['collection_mode'] == 'rounds' and cfg['training']['updates_per_round'] == 1000
        assert ac['pi05_robot'] == 'droid' and ac['discount'] == pytest.approx(0.995)


def test_ogpo_config_resolves(recipes):
    for scale in ('box138', 'mt16'):
        cfg = load_config(*parse_cli_args([f'--{k}={v}' for k, v in recipes[('ogpo', scale)].items()]))
        ac = config_to_agent_dict(cfg)
        assert ac['pi05_actor_objective'] == 'ppo' and ac['use_value_fn'] is False
        assert ac['pi05_critic_arch'] == 'bronet_vla' and ac['q_variance_reduction'] is True and ac['q_vr_num_samples'] == 16
        assert ac['best_of_n'] == 32 and ac['num_qs'] == 10 and ac['adv_strategy'] == 'conservative'
        assert ac['pi05_robot'] == 'droid' and ac['pi05_drop_empty_camera'] is True and ac['grad_accum_steps'] == 1
        assert cfg['training']['collection_mode'] == 'rounds' and cfg['training']['updates_per_round'] == 1000
        assert cfg['training']['num_rounds'] == 10 and cfg['training']['rollouts_per_round'] == 20


# ---------------------------------------------------------------------------------------------------------------------
# accepted by the agent and by main_pi05
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('key', list(RECIPES))
def test_config_passes_create_validation(recipes, stop_at_load, key):
    ac = dict(config_to_agent_dict(load_config(*parse_cli_args([f'--{k}={v}' for k, v in recipes[key].items()]))))
    with pytest.raises(_Reached):
        OGPOPi05Agent.create(0, ['a'] * N_TASKS[key[1]], ac)


@pytest.mark.parametrize('key', list(RECIPES))
def test_main_pi05_accepts_the_recipe(monkeypatch, recipes, key):
    args = [f'--{k}={v}' for k, v in recipes[key].items() if k != 'training.gpu_mode']  # one device on CPU
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')  # what the script exports from `device` (worker_egl_devices=auto reads it)
    M, built, Built = _main_harness(monkeypatch, *args)
    # molmo envs are vector envs: stub the worker pool, the wrapper and the prompt read-back (everything else in main is real)
    monkeypatch.setattr(M, 'make_vectorized_env', lambda names, n, **kw: SimpleNamespace(names=list(names), n=n, kw=kw,
                                                                                         close=lambda terminate=False: None))
    monkeypatch.setattr(M, 'VectorTasks', lambda vec, names, rows, pe, ib: SimpleNamespace(vec=vec, names=names, rows=rows, pe=pe, ib=ib,
                                                                                      num_slots=len(names) * pe))
    monkeypatch.setattr(M, 'vector_prompts', lambda vt: {n: f'prompt:{n}' for n in vt.names})
    with pytest.raises(Built):
        M.main()
    assert built['runner'] == 'rounds'
    vt = built['env']
    assert vt.names == recipes[key]['env.name'].split(',') and vt.pe == int(recipes[key]['env.parallel_envs'])
    assert vt.ib == int(recipes[key]['env.inference_batch']) and vt.vec.n == len(vt.names) * vt.pe
    assert len(built['prompt']) == N_TASKS[key[1]] if isinstance(built['prompt'], list) else N_TASKS[key[1]] == 1
    assert ('awr' in built['prefixes']) == (key[0] == 'awr')
    assert built['agent_config']['pi05_actor_objective'] == ('awr' if key[0] == 'awr' else 'ppo')
    assert built['agent_config']['pi05_robot'] == 'droid'


# ---------------------------------------------------------------------------------------------------------------------
# the setup scripts the recipes point at
# ---------------------------------------------------------------------------------------------------------------------

SETUP = os.path.join(ROOT, 'scripts', 'ogpo', 'setup_molmo_venv.sh')
DOWNLOAD = os.path.join(ROOT, 'scripts', 'ogpo', 'download_molmo_assets.sbatch')
MT16_TASKS = '267,366,138,143,176,373,93,394,48,185,24,62,4,260,221,348'


FAKE_UV = r"""#!/bin/bash
echo "uv $*" >> "$UV_LOG"
if [ "$1 $2" = "pip install" ]; then
  [ -n "${UV_FAIL:-}" ] && exit 1
  prev=""
  for a in "$@"; do
    if [ "$prev" = "-r" ] && [ -f "$a" ]; then echo "FILE $(basename "$a"): $(tr '\n' ' ' < "$a")" >> "$UV_LOG"; fi
    prev="$a"
  done
fi
if [ "$1" = venv ] && [ -z "${UV_NO_CREATE:-}" ]; then
  mkdir -p "$3/bin"
  printf '#!/bin/bash\necho "venvpython $*" >> "%s"\nexit 0\n' "$UV_LOG" > "$3/bin/python"
  chmod +x "$3/bin/python"
fi
[ "$1" = export ] && printf 'numpy==1.0\nmujoco==3.9\n'
exit 0
"""


def _run_setup(tmp_path, *args, uv=True, extra_env=None):
    """The script runs with a fake `uv` first on PATH (it logs its calls and the -r files it is given); nothing is installed."""
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    log = tmp_path / 'uv.log'
    if uv:
        fake = bindir / 'uv'
        fake.write_text(FAKE_UV)
        fake.chmod(0o755)
    r = subprocess.run(['bash', SETUP, *args], cwd=tmp_path, capture_output=True, text=True, timeout=60,
                       env={'PATH': f'{bindir}:/usr/bin:/bin', 'HOME': str(tmp_path), 'UV_LOG': str(log), **(extra_env or {})})
    return r, (log.read_text().splitlines() if log.exists() else [])


def _setup_dirs(tmp_path):
    mls, client = tmp_path / 'mls', tmp_path / 'client'
    (mls / 'molmo_spaces').mkdir(parents=True)
    client.mkdir()
    (client / 'pyproject.toml').write_text('')
    return mls, client


@pytest.mark.parametrize('args,msg', [
    ((), '--molmospaces=<path> is required: a lasgroup/molmospaces checkout at commit 2a828d5'),
    (('--molmospaces=/nope',), '--openpi_client=<path> is required'),
    (('--molmospaces=/nope', '--openpi_client=/nope2'), 'ERROR: missing /nope/molmo_spaces'),
])
def test_setup_requires_both_source_checkouts(tmp_path, args, msg):
    r, calls = _run_setup(tmp_path, *args)
    assert r.returncode == 1 and msg in r.stdout + r.stderr
    assert calls == []  # stopped before uv was called


def test_setup_no_longer_assumes_a_sibling_vla_post_training_checkout():
    text = open(SETUP).read()
    assert 'vla-post-training' not in text and 'TRI' not in text and 'docs/changes' not in text


def test_setup_installs_the_expected_packages(tmp_path):
    mls, client = _setup_dirs(tmp_path)
    venv = tmp_path / 'venv'
    r, calls = _run_setup(tmp_path, f'--molmospaces={mls}', f'--openpi_client={client}', f'--venv={venv}')
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    assert any(c.startswith('uv export --frozen --no-hashes --extra cuda12 --no-emit-project') for c in calls)
    assert f'uv venv -q {venv} --python 3.12' in calls
    installs = [c for c in calls if c.startswith('uv pip install')]
    assert len(installs) == 2
    for part in (f'--python {venv}/bin/python', 'ogpo_base.txt', '-r scripts/ogpo/molmo_venv/requirements.txt', f'-e {client}',
                 '--override scripts/ogpo/molmo_venv/overrides.txt', '--extra-index-url https://download.pytorch.org/whl/cpu',
                 '--index-strategy unsafe-best-match'):
        assert part in installs[0], part
    assert installs[1] == f'uv pip install -q --python {venv}/bin/python --no-deps -e {mls} -e .'
    base = next(c for c in calls if c.startswith('FILE ogpo_base.txt'))
    assert 'numpy==1.0' in base and 'mujoco' not in base  # the locked mujoco pin is dropped (the overrides file pins 3.4.0)
    assert any(c.startswith('venvpython -c import jax, mujoco') for c in calls)  # the import check ran


def test_setup_resolves_relative_arguments_against_the_callers_directory(tmp_path):
    _setup_dirs(tmp_path)
    real = os.path.realpath(tmp_path)
    r, calls = _run_setup(tmp_path, '--molmospaces=mls', '--openpi_client=client', '--venv=venv')  # cwd = tmp_path, not the checkout
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    installs = [c for c in calls if c.startswith('uv pip install')]
    assert f'-e {real}/client' in installs[0] and f'-e {real}/mls -e .' in installs[1]
    assert f'uv venv -q {real}/venv --python 3.12' in calls


def test_setup_default_venv_is_the_one_the_recipes_look_for(tmp_path):
    mls, client = _setup_dirs(tmp_path)
    r, calls = _run_setup(tmp_path, f'--molmospaces={mls}', f'--openpi_client={client}', extra_env={'UV_NO_CREATE': '1'})
    assert f'uv venv -q {os.path.realpath(ROOT)}/.venv-molmo --python 3.12' in calls
    assert 'venv-molmo/bin/activate' in open(os.path.join(ROOT, RECIPES[('ogpo', 'mt16')])).read()


def test_setup_stops_when_an_install_fails(tmp_path):
    mls, client = _setup_dirs(tmp_path)
    r, calls = _run_setup(tmp_path, f'--molmospaces={mls}', f'--openpi_client={client}', f'--venv={tmp_path / "venv"}',
                          extra_env={'UV_FAIL': '1'})
    assert r.returncode != 0 and not any(c.startswith('venvpython') for c in calls)  # set -e: no import check after a failed install


FAKE_GSUTIL = r"""#!/bin/bash
echo "gsutil $*" >> "$DL_LOG"
last="${@: -1}"
for a in "$@"; do
  case "$a" in
    gs://openpi-assets/checkpoints/pi05_droid_jointpos)
      [ "${DL_FAIL:-}" = ckpt ] && exit 1
      mkdir -p "$last/pi05_droid_jointpos/params" "$last/pi05_droid_jointpos/assets/droid"
      echo "{}" > "$last/pi05_droid_jointpos/assets/droid/norm_stats.json" ;;
    gs://big_vision/paligemma_tokenizer.model)
      [ "${DL_FAIL:-}" = tok ] && exit 1
      mkdir -p "$last"; : > "$last/paligemma_tokenizer.model" ;;
  esac
done
"""
# /usr/bin may hold a REAL gsutil (a past test run started a real 9 GB download through it). A tripwire named gsutil always sits first on
# PATH, so a script that ever calls bare `gsutil` instead of "$GSUTIL" hits it (and shows up in the log), never the real one.
TRIPWIRE = '#!/bin/bash\necho "TRIPWIRE gsutil $*" >> "$DL_LOG"\nexit 97\n'


def _run_download(base, *, env=None, with_venv=True, with_gsutil=True, preseed=(), extra_env=None):
    """GSUTIL is always pinned to the fake (or to a path that does not exist), and a tripwire shadows any real gsutil on PATH."""
    root = base / 'checkout'
    (root / 'scripts' / 'ogpo').mkdir(parents=True)
    shutil.copy2(DOWNLOAD, root / 'scripts' / 'ogpo' / 'download_molmo_assets.sbatch')
    if with_venv:
        (root / '.venv-molmo' / 'bin').mkdir(parents=True)
        py = root / '.venv-molmo' / 'bin' / 'python'
        py.write_text('#!/bin/bash\nprintf \'%s\\n\' "$@" > "$DL_PY_LOG"\n')
        py.chmod(0o755)
    bindir = base / 'bin'
    bindir.mkdir()
    gs = bindir / 'gsutil-fake'  # what $GSUTIL points at; a bare `gsutil` on PATH is the tripwire
    gs.write_text(FAKE_GSUTIL)
    gs.chmod(0o755)
    trip = bindir / 'gsutil'
    trip.write_text(TRIPWIRE)
    trip.chmod(0o755)
    paths = {'OPENPI_CACHE': str(base / 'openpi'), 'MLSPACES_ASSETS_DIR': str(base / 'mls_assets'),
             'MLSPACES_CACHE_DIR': str(base / 'mls_cache'), 'NLTK_DATA': str(base / 'nltk')} if env is None else env
    for rel in preseed:
        pth = base / 'openpi' / rel
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.mkdir() if rel.endswith('params') else pth.write_text('')
    full = {'PATH': f'{bindir}:/usr/bin:/bin', 'HOME': str(base), 'DL_LOG': str(base / 'gs.log'), 'DL_PY_LOG': str(base / 'py.log'),
            'GSUTIL': str(gs) if with_gsutil else '/nonexistent/gsutil', **paths, **(extra_env or {})}
    r = subprocess.run(['bash', str(root / 'scripts' / 'ogpo' / 'download_molmo_assets.sbatch')], cwd=root, env=full,
                       capture_output=True, text=True, timeout=60)
    gs_calls = (base / 'gs.log').read_text().splitlines() if (base / 'gs.log').exists() else []
    assert not any(c.startswith('TRIPWIRE') for c in gs_calls), gs_calls  # something called a bare gsutil
    py_args = (base / 'py.log').read_text().splitlines() if (base / 'py.log').exists() else None
    return r, gs_calls, py_args, paths


def test_download_fetches_checkpoint_tokenizer_and_the_mt16_tasks(tmp_path):
    r, gs, py, paths = _run_download(tmp_path)
    assert r.returncode == 0 and 'DOWNLOAD_DONE' in r.stdout, r.stdout[-400:] + r.stderr[-400:]
    dst = f'{paths["OPENPI_CACHE"]}/openpi-assets/checkpoints/'
    assert gs == [f'gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid_jointpos {dst}',
                  f'gsutil cp gs://big_vision/paligemma_tokenizer.model {paths["OPENPI_CACHE"]}/big_vision/']
    assert py == ['scripts/ogpo/molmo_prefetch.py', '--tasks', MT16_TASKS]
    assert os.path.isfile(f'{paths["OPENPI_CACHE"]}/big_vision/paligemma_tokenizer.model')


def test_download_is_idempotent_and_does_not_need_gsutil_when_everything_is_there(tmp_path):
    r, gs, py, _ = _run_download(tmp_path, with_gsutil=False, preseed=(
        'openpi-assets/checkpoints/pi05_droid_jointpos/params', 'openpi-assets/checkpoints/pi05_droid_jointpos/assets/droid/norm_stats.json',
        'big_vision/paligemma_tokenizer.model'))
    assert r.returncode == 0 and gs == [] and 'already present' in r.stdout
    assert py[-2:] == ['--tasks', MT16_TASKS]


def test_download_downloads_only_the_tokenizer_when_the_checkpoint_is_there(tmp_path):
    r, gs, _, paths = _run_download(tmp_path, preseed=(
        'openpi-assets/checkpoints/pi05_droid_jointpos/params', 'openpi-assets/checkpoints/pi05_droid_jointpos/assets/droid/norm_stats.json'))
    assert r.returncode == 0 and gs == [f'gsutil cp gs://big_vision/paligemma_tokenizer.model {paths["OPENPI_CACHE"]}/big_vision/']


def test_download_tasks_can_be_overridden(tmp_path):
    r, _, py, _ = _run_download(tmp_path, extra_env={'TASKS': '164,86'})
    assert r.returncode == 0 and py == ['scripts/ogpo/molmo_prefetch.py', '--tasks', '164,86']


@pytest.mark.parametrize('var', ['OPENPI_CACHE', 'MLSPACES_ASSETS_DIR', 'MLSPACES_CACHE_DIR', 'NLTK_DATA'])
def test_download_fails_fast_without_its_paths(tmp_path, var):
    env = {k: str(tmp_path / k) for k in ('OPENPI_CACHE', 'MLSPACES_ASSETS_DIR', 'MLSPACES_CACHE_DIR', 'NLTK_DATA')}
    del env[var]
    r, gs, py, _ = _run_download(tmp_path, env=env)
    assert r.returncode != 0 and f'{var}: set {var}' in r.stderr + r.stdout and gs == [] and py is None


def test_download_needs_the_molmo_venv_and_gsutil(tmp_path):
    r, gs, py, _ = _run_download(tmp_path, with_venv=False)
    assert r.returncode != 0 and 'with .venv-molmo' in r.stdout + r.stderr and gs == [] and py is None
    (tmp_path / 'g').mkdir()
    r, gs, py, _ = _run_download(tmp_path / 'g', with_gsutil=False)
    assert r.returncode != 0 and 'gsutil not found' in r.stdout + r.stderr and gs == [] and py is None


@pytest.mark.parametrize('what', ['ckpt', 'tok'])
def test_download_stops_when_a_download_fails(tmp_path, what):
    r, gs, py, _ = _run_download(tmp_path, extra_env={'DL_FAIL': what})
    assert r.returncode != 0 and 'DOWNLOAD_DONE' not in r.stdout and py is None  # set -e: no prefetch after a failed download


@pytest.mark.parametrize('present', ['openpi-assets/checkpoints/pi05_droid_jointpos/params',
                                     'openpi-assets/checkpoints/pi05_droid_jointpos/assets/droid/norm_stats.json'])
def test_download_redownloads_a_partial_checkpoint(tmp_path, present):
    r, gs, _, _ = _run_download(tmp_path, preseed=(present, 'big_vision/paligemma_tokenizer.model'))
    assert r.returncode == 0 and len(gs) == 1 and 'pi05_droid_jointpos' in gs[0]  # params AND norm_stats must both be there to skip


def test_download_has_no_personal_paths_or_cluster_partition():
    text = open(DOWNLOAD).read()
    for bad in ('/data/user_data', '/home/', 'maxlab', '#SBATCH --partition'):
        assert bad not in text, bad


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
