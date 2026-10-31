"""Verifier wrapper: run tests/test_resume.py with the repo's `tests` package ahead of the site-packages `tests` that
.venv-molmo ships (which makes `from tests._pi05_stubs import ...` fail with ModuleNotFoundError).

Run:  JAX_PLATFORMS=cpu pytest tests/verify_run_test_resume.py -q
"""
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for name in [m for m in sys.modules if m == "tests" or m.startswith("tests.")]:
    del sys.modules[name]
sys.path.insert(0, REPO)
_pkg = types.ModuleType("tests")  # tests/ has no __init__.py, so a regular site-packages `tests` would win the import
_pkg.__path__ = [os.path.join(REPO, "tests")]
sys.modules["tests"] = _pkg

from tests.test_resume import *  # noqa: E402,F401,F403  (pytest collects the imported test_* functions)
