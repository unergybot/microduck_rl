from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_rom_candidate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rom_candidate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_policy_validates_action_task_pair() -> None:
    policy = MODULE.parse_policy(
        "STAND=Mjlab-SitStand-Flat-MicroDuck=logs/stand/model_2000.pt"
    )
    assert policy.action_code == "STAND"
    assert policy.task_id == "Mjlab-SitStand-Flat-MicroDuck"
    assert policy.checkpoint_file.name == "model_2000.pt"

    with pytest.raises(MODULE.argparse.ArgumentTypeError, match="not valid"):
        MODULE.parse_policy("STAND=Mjlab-Velocity-Flat-MicroDuck=model_2000.pt")


def test_default_checkpoint_label_is_traceable() -> None:
    stand = MODULE.parse_policy("STAND=Mjlab-SitStand-Flat-MicroDuck=model_2000.pt")
    squat = MODULE.parse_policy(
        "SQUAT_REFERENCE=Mjlab-SquatReference-Flat-MicroDuck=model_1950.pt"
    )
    assert MODULE.default_checkpoint_label((stand,)) == "model_2000.pt"
    assert MODULE.default_checkpoint_label((stand, squat)) == "mixed-checkpoints"


def test_write_release_config_uses_code_owned_action_contract(tmp_path: Path) -> None:
    path = tmp_path / "release.json"
    MODULE.write_release_config(
        path,
        release="1.0.5",
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
        action_codes=("STAND", "SQUAT_REFERENCE"),
        mandatory_actions={"STAND"},
        seeds=(7, 11, 29),
        max_steps=100,
    )
    document = json.loads(path.read_text())
    assert document["release"] == "1.0.5"
    assert document["createdAt"] == "2026-09-03T00:00:00Z"
    stand, squat = document["actions"]
    assert stand["mandatory"] is True
    assert stand["resetProfile"] == "TRAINED_SITTING"
    assert stand["parameters"] == {}
    assert stand["thresholds"]["actionMetric"] == "standPoseError"
    assert stand["thresholds"]["actionMetricOperator"] == "lte"
    assert stand["thresholds"]["actionMetricThreshold"] == pytest.approx(0.08)
    assert stand["maxSteps"] == 100
    assert squat["mandatory"] is False
    assert squat["parameters"] == {}
    assert squat["maxSteps"] == 250
    assert squat["thresholds"]["actionMetric"] == "minimumCrouchHeightM"
    assert squat["thresholds"]["actionMetricOperator"] == "lte"
