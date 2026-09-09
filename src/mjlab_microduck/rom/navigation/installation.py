"""Load simulator configuration independently of optional benchmark reports."""

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
    evaluation_status: str = "NOT_EVALUATED"


def load(root, bundle_digest):
    root = Path(root).resolve()
    config_path = root / "navigation.json"
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or config_path.stat().st_size > 262144
    ):
        raise ValueError("navigation configuration unavailable")
    config = json.loads(config_path.read_text())
    required = {"scene", "profile", "robotId", "targetId"}
    if (
        not isinstance(config, dict)
        or not required <= set(config)
        or set(config) - required - {"qualificationDigest"}
    ):
        raise ValueError("invalid navigation installation")
    from ..navigation_contracts import Binding

    Binding(
        robotId=config["robotId"],
        targetId=config["targetId"],
        runtimeSession="validation",
    )
    scene = Scene.model_validate(config["scene"])
    profile = NavigationProfile.model_validate(config["profile"])
    # Compatibility name on the private wire: this binds executable configuration,
    # never benchmark success. Updating an advisory report cannot invalidate a task.
    identity = digest(
        {
            "schema": "MICRODUCK_NAVIGATION_INSTALLATION_V1",
            "bundleDigest": bundle_digest,
            "scene": scene.model_dump(),
            "profile": profile.model_dump(),
            "robotId": config["robotId"],
            "targetId": config["targetId"],
        }
    )
    status = "NOT_EVALUATED"
    report_path = root / "navigation-qualification.json"
    if report_path.exists() or report_path.is_symlink():
        status = "INVALID_REPORT"
        try:
            if (
                report_path.is_symlink()
                or not report_path.is_file()
                or report_path.stat().st_size > 262144
            ):
                raise ValueError("invalid report")
            report = json.loads(report_path.read_text())
            if (
                isinstance(report, dict)
                and report.get("schema") == "MICRODUCK_NAVIGATION_QUALIFICATION_V1"
            ):
                status = "REPORT_AVAILABLE"
                if (
                    report.get("bundleDigest") != bundle_digest
                    or report.get("mapDigest") != digest(scene)
                    or report.get("profileDigest") != digest(profile)
                    or report.get("runtimeSourceDigest") != source_digest()
                ):
                    status = "STALE_REPORT"
                else:
                    runs = report.get("runs")
                    if (
                        isinstance(runs, list)
                        and runs
                        and all(
                            isinstance(r, dict) and type(r.get("passed")) is bool
                            for r in runs
                        )
                    ):
                        status = (
                            "REPORTED_PASS"
                            if all(r["passed"] for r in runs)
                            else "REPORTED_FAILURE"
                        )
        except (OSError, ValueError):
            pass
    return Installation(
        scene,
        profile,
        config["robotId"],
        config["targetId"],
        bundle_digest,
        identity,
        status,
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

    from ..main import load_verified_bundle

    parser = argparse.ArgumentParser(
        description="Read-only navigation installation verification"
    )
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_verified_bundle(args.bundle)
    installed = load(args.bundle, bundle.bundleDigest)
    if not any(
        action.actionCode == "WALK_VELOCITY" and action.availability == "AVAILABLE"
        for action in bundle.actions
    ):
        raise ValueError("compatible locomotion is unavailable")
    print("Navigation configuration verified:", installed.qualification_digest)
    print("Evaluation (advisory):", installed.evaluation_status)


if __name__ == "__main__":
    main()
