"""Env adapters. Importing this package first turns off robosuite's file log (see `_disable_robosuite_file_log`)."""
import importlib.util
import sys


def _disable_robosuite_file_log():
    if "robosuite" in sys.modules or "robosuite.macros" in sys.modules:
        return  # already imported: too late to change, and the import already succeeded
    spec = importlib.util.find_spec("robosuite")  # locates the package without executing its __init__
    if spec is None or not spec.submodule_search_locations:
        return  # robosuite not installed (it is an optional extra): nothing to configure
    macros_spec = importlib.util.spec_from_file_location(
        "robosuite.macros", f"{spec.submodule_search_locations[0]}/macros.py")
    macros = importlib.util.module_from_spec(macros_spec)
    macros_spec.loader.exec_module(macros)
    macros.FILE_LOGGING_LEVEL = None
    sys.modules["robosuite.macros"] = macros


_disable_robosuite_file_log()
