import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from mjlab_microduck.rom.navigation.installation import _NON_DEPLOYED_SOURCE, load


def test_canonical_v2_scene_binds_the_desk_and_both_door_posts():
    from mjlab_microduck.rom.navigation.calibrated_scene import canonical_v2_scene

    scene = canonical_v2_scene()
    assert scene.revision == "microduck-navigation-calibration-v2"
    assert [obstacle.model_dump() for obstacle in scene.obstacles] == [
        {"minX": 0.4, "maxX": 0.6, "minY": -0.1, "maxY": 0.3},
        {"minX": -0.305, "maxX": -0.255, "minY": 0.975, "maxY": 1.025},
        {"minX": 0.255, "maxX": 0.305, "minY": 0.975, "maxY": 1.025},
    ]
    assert {name: pose.model_dump() for name, pose in scene.landmarks.items()} == {
        "home": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "desk": {"x": 1.0, "y": 0.0, "yaw": 0.0},
        "door": {"x": 0.0, "y": 1.0, "yaw": 1.5707963267948966},
    }


@pytest.mark.parametrize("mutation", ["moved", "missing", "reordered"])
def test_v2_installation_rejects_noncanonical_door_posts(tmp_path, mutation):
    canonical = json.loads(
        Path("src/mjlab_microduck/rom/navigation/calibrated_v2.json").read_text()
    )
    value = config(tmp_path)
    value["scene"] = deepcopy(canonical["scene"])
    posts = value["scene"]["obstacles"]
    if mutation == "moved":
        posts[1]["minX"] -= 0.01
    elif mutation == "missing":
        posts.pop()
    else:
        posts[1], posts[2] = posts[2], posts[1]
    (tmp_path / "navigation.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="noncanonical v2 scene"):
        load(tmp_path, "sha256:" + "a" * 64)


def test_v2_scene_files_are_in_the_simulator_image_context():
    dockerfile = Path("docker/rom-simulator/Dockerfile").read_text()
    for name in ("calibrated_scene.py", "calibrated_v2.json"):
        source = "src/mjlab_microduck/rom/navigation/" + name
        assert source in dockerfile
        assert "!" + source in Path(".dockerignore").read_text()
        assert "!" + source in Path("docker/rom-simulator/Dockerfile.dockerignore").read_text()


def test_source_digest_covers_exactly_the_container_python_closure():
    root = Path("src/mjlab_microduck/rom")
    local = {path.relative_to(root).as_posix() for path in root.rglob("*.py")}
    dockerfile = Path("docker/rom-simulator/Dockerfile").read_text()
    copied = {
        path.removeprefix("src/mjlab_microduck/rom/")
        for path in re.findall(r"src/mjlab_microduck/rom/[\w/]+\.py", dockerfile)
    }
    assert _NON_DEPLOYED_SOURCE <= local
    assert local - _NON_DEPLOYED_SOURCE == copied


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
