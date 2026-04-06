"""Root conftest.py: allow lightweight unit tests to import verl submodules
without requiring a full Ray / vLLM / tensordict installation.

Strategy: inject a stub ``verl`` package into sys.modules *before* pytest
collects tests.  Because Python only executes __init__.py once per package
object, subsequent ``import verl.subpkg`` calls locate sub-packages on the
filesystem without re-running verl/__init__.py.
"""

import importlib
import importlib.util
import sys
import types


def _stub_package(dotted_name: str, path: str | None = None) -> types.ModuleType:
    """Return a module registered in sys.modules under *dotted_name*.

    If *path* is given the module's __path__ is set so Python can find
    sub-packages beneath it without running __init__.py.
    """
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    mod = types.ModuleType(dotted_name)
    if path is not None:
        mod.__path__ = [path]  # type: ignore[attr-defined]
        mod.__package__ = dotted_name
    sys.modules[dotted_name] = mod
    return mod


# ------------------------------------------------------------------
# 1. Stub the top-level ``verl`` package so its __init__.py (which
#    imports ray, tensordict, torch…) is never executed.
# ------------------------------------------------------------------
import os as _os

_VERL_ROOT = _os.path.join(_os.path.dirname(__file__), "verl")
_stub_package("verl", _VERL_ROOT)

# ------------------------------------------------------------------
# 2. Walk the package hierarchy and stub each intermediate __init__
#    so that deep imports like
#    verl.utils.reward_score.symbolic_process_reward.format_gate
#    are resolved by the real __init__.py only when it is safe to do so.
#    For the utils layer we use the real __init__.py files IF they exist
#    and have no heavy side-effects; otherwise we stub them too.
# ------------------------------------------------------------------

def _safe_init(dotted_name: str, pkg_path: str) -> None:
    """Register *dotted_name* using its real __init__.py if it exists and is
    importable; otherwise fall back to a stub with __path__ set."""
    if dotted_name in sys.modules:
        return

    init_file = _os.path.join(pkg_path, "__init__.py")
    if not _os.path.exists(init_file):
        _stub_package(dotted_name, pkg_path)
        return

    try:
        spec = importlib.util.spec_from_file_location(
            dotted_name, init_file,
            submodule_search_locations=[pkg_path],
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        mod.__path__ = [pkg_path]  # type: ignore[attr-defined]
        mod.__package__ = dotted_name
        sys.modules[dotted_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        # Fall back to stub so we don't block test collection.
        sys.modules.pop(dotted_name, None)
        _stub_package(dotted_name, pkg_path)


# Intermediate packages — each real __init__.py should be safe to execute.
_CHAIN = [
    ("verl.utils", _os.path.join(_VERL_ROOT, "utils")),
    ("verl.utils.reward_score", _os.path.join(_VERL_ROOT, "utils", "reward_score")),
    (
        "verl.utils.reward_score.symbolic_process_reward",
        _os.path.join(_VERL_ROOT, "utils", "reward_score", "symbolic_process_reward"),
    ),
]

for _dotted, _path in _CHAIN:
    _safe_init(_dotted, _path)
