"""Import smoke test — the guardrail that fails CI when the module graph breaks.

A prior refactor landed on `main` with three `core/` modules missing, leaving the
whole pipeline non-importable. This test walks every first-party package and imports
each module so that class of breakage is caught before merge, not at runtime.
"""

import importlib
import pkgutil

import pytest

PACKAGES = [
    "config",
    "core",
    "motion",
    "ml",
    "ml.inference",
    "pipeline",
    "pipeline.steps_v2",
    "video_io",
]


def _iter_module_names():
    seen = set()
    for pkg_name in PACKAGES:
        pkg = importlib.import_module(pkg_name)
        if pkg_name not in seen:
            seen.add(pkg_name)
            yield pkg_name
        for info in pkgutil.iter_modules(pkg.__path__, prefix=f"{pkg_name}."):
            if info.name not in seen:
                seen.add(info.name)
                yield info.name


@pytest.mark.parametrize("module_name", list(_iter_module_names()))
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_process_entrypoint_imports():
    """The CLI entry point wires config + core together; import it explicitly."""
    import importlib.util
    from pathlib import Path

    process_path = Path(__file__).resolve().parent.parent / "process.py"
    spec = importlib.util.spec_from_file_location("wildcams_process", process_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "initialize_config_from_args")
