"""Read pinned simulator navigation artifacts; missing qualification stays disabled."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..navigation_contracts import NavigationProfile, Scene, digest


@dataclass(frozen=True)
class Installation:
    scene: Scene
    profile: NavigationProfile
    robot_id: str
    target_id: str
    bundle_digest: str
    qualification_digest: str


def load(root, bundle_digest):
    root = Path(root).resolve()
    config_path = root / "navigation.json"
    report_path = root / "navigation-qualification.json"
    for path in (config_path, report_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 262144:
            raise ValueError("navigation qualification unavailable")
    config = json.loads(config_path.read_text())
    report = json.loads(report_path.read_text())
    if not isinstance(config, dict) or not isinstance(report, dict):
        raise ValueError("navigation artifacts must be JSON objects")
    if set(config) != {
        "scene",
        "profile",
        "robotId",
        "targetId",
        "qualificationDigest",
    }:
        raise ValueError("invalid navigation installation")
    from ..navigation_contracts import Binding

    Binding(
        robotId=config["robotId"],
        targetId=config["targetId"],
        runtimeSession="validation",
    )
    scene = Scene.model_validate(config["scene"])
    profile = NavigationProfile.model_validate(config["profile"])
    if (
        config["qualificationDigest"] != digest(report)
        or report.get("bundleDigest") != bundle_digest
        or report.get("mapDigest") != digest(scene)
        or report.get("profileDigest") != digest(profile)
        or report.get("provenance") != "SIM_GROUND_TRUTH"
        or report.get("engine") != "MUJOCO_ONNX"
    ):
        raise ValueError("navigation qualification identity mismatch")
    if (
        report.get("schema") != "MICRODUCK_NAVIGATION_QUALIFICATION_V1"
        or report.get("runtimeSourceDigest") != source_digest()
    ):
        raise ValueError("navigation runtime source does not match qualification")
    runs = report.get("runs", [])
    if not isinstance(runs, list) or len(runs) != 50:
        raise ValueError("navigation qualification requires exactly fifty runs")
    required = {"open", "turn", "detour", "unreachable", "settle"}
    if any(
        not isinstance(r, dict)
        or r.get("passed") is not True
        or type(r.get("seed")) is not int
        or r.get("collision") is not False
        for r in runs
    ):
        raise ValueError("navigation qualification failed")
    for scenario in required:
        seeds = {r.get("seed") for r in runs if r.get("scenario") == scenario}
        if not set(range(10)).issubset(seeds):
            raise ValueError("navigation qualification incomplete")
    return Installation(
        scene,
        profile,
        config["robotId"],
        config["targetId"],
        bundle_digest,
        digest(report),
    )


def source_digest():
    """Bind qualification to the complete simulator implementation, including safety IPC."""
    import hashlib

    root = Path(__file__).resolve().parents[1]
    values = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    return digest(values)


def main():
    import argparse

    from ..main import load_qualified_bundle

    parser = argparse.ArgumentParser(
        description="Read-only navigation installation verification"
    )
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_qualified_bundle(args.bundle)
    installed = load(args.bundle, bundle.bundleDigest)
    if not any(
        action.actionCode == "WALK_VELOCITY" and action.availability == "AVAILABLE"
        for action in bundle.actions
    ):
        raise ValueError("qualified locomotion is unavailable")
    print("Navigation qualification verified:", installed.qualification_digest)


if __name__ == "__main__":
    main()
