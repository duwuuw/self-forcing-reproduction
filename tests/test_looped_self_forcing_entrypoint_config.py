from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "scripts" / "looped_self_forcing_pipeline" / "source"


@pytest.mark.parametrize("entrypoint", ["train", "inference"])
def test_entrypoint_loads_copied_default_config_from_unrelated_cwd(
    tmp_path: Path, entrypoint: str
):
    override_path = tmp_path / "override.yaml"
    override_path.write_text("num_train_timestep: 777\n", encoding="utf-8")
    script = (
        "import importlib, json, sys; "
        f"config = importlib.import_module('{entrypoint}').load_config(sys.argv[1]); "
        "print(json.dumps([config.same_step_across_blocks, config.num_train_timestep]))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(override_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [True, 777]
