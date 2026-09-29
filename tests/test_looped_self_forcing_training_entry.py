from __future__ import annotations

import importlib
import sys
from importlib.machinery import PathFinder
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def test_training_entry_prioritizes_copied_pipeline_package_for_imports():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    training_entry = importlib.import_module(
        "looped_self_forcing_pipeline.training_entry"
    )
    search_path = [
        str(training_entry.PACKAGE_ROOT),
        str(training_entry.SOURCE_ROOT),
    ]

    prioritize_source_root = getattr(
        training_entry, "prioritize_source_root", None
    )
    assert callable(prioritize_source_root)
    prioritize_source_root(search_path)

    spec = PathFinder.find_spec("pipeline", search_path)

    assert spec is not None
    assert Path(spec.origin) == training_entry.SOURCE_ROOT / "pipeline" / "__init__.py"
