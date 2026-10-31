"""`envs/__init__.py` turns off robosuite's /tmp/robosuite.log FileHandler.

The patch acts at import time, so every scenario runs in a fresh interpreter. No scenario creates
/tmp/robosuite.log: logging.FileHandler is replaced, in every child, by a stub that either raises
PermissionError for that path (simulating a file owned by another user) or records the path and
writes nothing.

Run:  uv run --no-sync pytest tests/test_envs_robosuite_file_log.py -q
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIMEOUT = 900  # libero imports can take minutes on a cold cache

# find_spec, not importorskip: importing robosuite in the pytest process would itself create /tmp/robosuite.log.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("robosuite") is None or importlib.util.find_spec("libero") is None,
    reason="needs the robosuite + libero extras")

# Simulates /tmp/robosuite.log created by another user: opening it raises PermissionError.
DENY = textwrap.dedent("""
    import logging
    class _DeniedFileHandler(logging.Handler):
        def __init__(self, filename, *a, **k):
            if str(filename) == "/tmp/robosuite.log":
                raise PermissionError(13, "Permission denied", filename)
            super().__init__()
    logging.FileHandler = _DeniedFileHandler
""")

# Records the paths robosuite tries to open, writes nothing.
RECORD = textwrap.dedent("""
    import logging
    OPENED = []
    class _RecordingFileHandler(logging.Handler):
        def __init__(self, filename, *a, **k):
            OPENED.append(str(filename))
            super().__init__()
        def emit(self, record):
            pass
    logging.FileHandler = _RecordingFileHandler
""")

DUMP_MACROS = textwrap.dedent("""
    import json, sys
    _m = sys.modules["robosuite.macros"]
    print("MACROS=" + json.dumps({k: getattr(_m, k) for k in dir(_m) if k.isupper()}))
""")


def _run(code):
    env = dict(os.environ, PYTHONPATH=REPO, MUJOCO_GL="egl")
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=TIMEOUT)


def _tagged(out, tag):
    lines = [l for l in out.splitlines() if l.startswith(tag + "=")]
    assert len(lines) == 1, out
    return json.loads(lines[0][len(tag) + 1:])


def test_envs_import_survives_denied_log_file():
    """(a) `import envs` first: libero imports under a denied /tmp/robosuite.log, no FileHandler attached."""
    code = DENY + textwrap.dedent("""
        import json, sys, logging
        import envs
        before = sorted(k for k in sys.modules if k.split(".")[0] == "robosuite")
        import libero.libero.envs
        import robosuite.utils.log_utils as lu
        print("OUT=" + json.dumps({
            "before": before,
            "handlers": [type(h).__name__ for h in logging.getLogger("robosuite_logs").handlers],
            "level": sys.modules["robosuite.macros"].FILE_LOGGING_LEVEL,
            "same_module": lu.macros is sys.modules["robosuite.macros"],
        }))
    """)
    p = _run(code)
    assert p.returncode == 0, p.stderr
    out = _tagged(p.stdout, "OUT")
    assert out["before"] == ["robosuite.macros"]  # envs does not import robosuite itself
    assert out["level"] is None
    assert out["same_module"]
    assert "_DeniedFileHandler" not in out["handlers"]
    assert out["handlers"] == ["StreamHandler"]  # console logging still on


def test_without_envs_denied_log_file_raises():
    """(b) control: the same denial without `import envs` kills the import, so (a) can fail."""
    p = _run(DENY + "import libero.libero.envs\n")
    assert p.returncode != 0
    assert "PermissionError" in p.stderr and "/tmp/robosuite.log" in p.stderr, p.stderr


def test_other_macros_unchanged():
    """(c) every macro except FILE_LOGGING_LEVEL matches an unpatched import."""
    base = _run(RECORD + "import robosuite\n" + DUMP_MACROS + "print('OPENED=' + json.dumps(OPENED))\n")
    assert base.returncode == 0, base.stderr
    patched = _run(RECORD + "import envs\nimport robosuite\n" + DUMP_MACROS
                   + "print('OPENED=' + json.dumps(OPENED))\n")
    assert patched.returncode == 0, patched.stderr

    m_base, m_patched = _tagged(base.stdout, "MACROS"), _tagged(patched.stdout, "MACROS")
    assert _tagged(base.stdout, "OPENED") == ["/tmp/robosuite.log"]  # the unpatched import does open it
    assert _tagged(patched.stdout, "OPENED") == []
    assert m_base["FILE_LOGGING_LEVEL"] == "DEBUG"
    assert m_patched["FILE_LOGGING_LEVEL"] is None
    m_base.pop("FILE_LOGGING_LEVEL"), m_patched.pop("FILE_LOGGING_LEVEL")
    assert m_base == m_patched
    assert len(m_base) >= 10


def test_spawn_child_sees_patch():
    """(d) a spawn worker that only gets `envs` by unpickling an envs.libero_pi05_utils function is patched."""
    code = textwrap.dedent("""
        import json, multiprocessing as mp
        import envs.libero_pi05_utils as L
        ctx = mp.get_context("spawn")
        with ctx.Pool(1, initializer=exec, initargs=(%r,)) as pool:
            fresh = pool.apply(eval, ("'envs' in __import__('sys').modules",))
            assert pool.apply(L.is_libero_pi05_env, ("libero_pi05-libero_90-59",)) is True
            res = pool.apply(eval, (
                "(__import__('libero.libero.envs'),"
                " __import__('sys').modules['robosuite.macros'].FILE_LOGGING_LEVEL,"
                " [type(h).__name__ for h in __import__('logging').getLogger('robosuite_logs').handlers])[1:]",))
        print("OUT=" + json.dumps({"fresh_had_envs": fresh, "level": res[0], "handlers": res[1]}))
    """ % DENY)
    p = _run(code)
    assert p.returncode == 0, p.stderr
    out = _tagged(p.stdout, "OUT")
    assert out["fresh_had_envs"] is False  # the worker did not inherit envs: unpickling imported it
    assert out["level"] is None
    assert out["handlers"] == ["StreamHandler"]


def test_noop_when_robosuite_already_imported():
    """robosuite imported before envs: the patch leaves the live macros module alone."""
    code = RECORD + textwrap.dedent("""
        import sys, robosuite
        m = sys.modules["robosuite.macros"]
        import envs
        assert sys.modules["robosuite.macros"] is m
        assert m.FILE_LOGGING_LEVEL == "DEBUG"
        print("OK")
    """)
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "OK" in p.stdout


def test_noop_when_robosuite_missing():
    """robosuite not installed (optional extra): `import envs` still works and registers nothing."""
    code = textwrap.dedent("""
        import importlib.util, sys
        _orig = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *a, **k: None if name == "robosuite" else _orig(name, *a, **k)
        import envs
        assert "robosuite.macros" not in sys.modules and "robosuite" not in sys.modules
        print("OK")
    """)
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "OK" in p.stdout
