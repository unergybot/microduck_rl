import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_rom_mujoco_runtime import _write_verified_bundle


@pytest.mark.parametrize(
    "options,seeds", [(["--seed-count", "1"], [0]), (["--seeds", "7", "11"], [7, 11])]
)
def test_requested_evaluation_batch_can_succeed_without_fifty_runs(
    tmp_path, monkeypatch, options, seeds
):
    root = tmp_path / "bundle"
    root.mkdir()
    _write_verified_bundle(root)
    data = json.loads(Path("tests/fixtures/navigation/scenarios.json").read_text())
    data["scenarios"] = [s for s in data["scenarios"] if s["name"] == "unreachable"]
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps(data))
    output = tmp_path / "report.json"
    spec = importlib.util.spec_from_file_location(
        "navigation_evaluation", "scripts/qualify_rom_navigation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualify_rom_navigation.py",
            "--bundle",
            str(root),
            "--scenarios",
            str(scenarios),
            *options,
            "--output",
            str(output),
        ],
    )
    assert module.main() == 0
    report = json.loads(output.read_text())
    assert [run["seed"] for run in report["runs"]] == seeds
    assert report["runs"][0]["reason"] == "PATH_BLOCKED"
