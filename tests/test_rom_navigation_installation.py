import json
from pathlib import Path

import pytest

from mjlab_microduck.rom.navigation.installation import load


def config(tmp_path):
    source = json.loads(Path("tests/fixtures/navigation/candidate.json").read_text())
    value = {**source, "robotId": "1", "targetId": "2"}
    (tmp_path / "navigation.json").write_text(json.dumps(value))
    return value


def test_navigation_configuration_does_not_require_a_benchmark(tmp_path):
    config(tmp_path)
    installed = load(tmp_path, "sha256:" + "a" * 64)
    assert installed.evaluation_status == "NOT_EVALUATED"
    assert installed.robot_id == "1"


@pytest.mark.parametrize(
    "report", [None, {"runs": [{"passed": False}]}, {"runtimeSourceDigest": "old"}, []]
)
def test_evaluation_never_blocks_configuration_or_changes_execution_identity(
    tmp_path, report
):
    config(tmp_path)
    before = load(tmp_path, "sha256:" + "a" * 64)
    (tmp_path / "navigation-qualification.json").write_text(json.dumps(report))
    after = load(tmp_path, "sha256:" + "a" * 64)
    assert after.qualification_digest == before.qualification_digest


def test_invalid_controller_configuration_is_still_rejected(tmp_path):
    value = config(tmp_path)
    value["profile"]["leaseMs"] = 0
    (tmp_path / "navigation.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load(tmp_path, "sha256:" + "a" * 64)


def test_failed_matching_report_is_visible_but_not_an_execution_gate(tmp_path):
    from mjlab_microduck.rom.navigation.installation import source_digest
    from mjlab_microduck.rom.navigation_contracts import digest

    value = config(tmp_path)
    bundle = "sha256:" + "a" * 64
    before = load(tmp_path, bundle)
    report = {
        "schema": "MICRODUCK_NAVIGATION_QUALIFICATION_V1",
        "bundleDigest": bundle,
        "mapDigest": digest(value["scene"]),
        "profileDigest": digest(value["profile"]),
        "runtimeSourceDigest": source_digest(),
        "runs": [{"passed": False}],
    }
    (tmp_path / "navigation-qualification.json").write_text(json.dumps(report))
    after = load(tmp_path, bundle)
    assert after.evaluation_status == "REPORTED_FAILURE"
    assert after.qualification_digest == before.qualification_digest
