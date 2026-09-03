"""Export, stamp, build, and qualify a MicroDuck ROM evaluation candidate.

This script makes the provenance triple explicit so intermediate evaluation
bundles do not accidentally fail with POLICY_PROVENANCE_MISMATCH when policies
come from different local training runs. It exports ONNX policies on CPU,
restamps their ROM metadata to one auditable evaluation identity, builds an
immutable candidate ZIP, writes the release battery configuration, and then
runs the governed qualification battery unless --skip-qualify is set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Keep this utility runnable both as ``python scripts/...`` and when imported
# by the test suite from a checkout that has not been installed editable.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_path in (REPOSITORY_ROOT, SOURCE_ROOT, REPOSITORY_ROOT / "scripts"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from mjlab_microduck.rom.action_specs import ACTION_RUNTIME_SPECS, SQUAT_RETURN_LIMITS
from mjlab_microduck.rom.qualification import QualificationThresholds

DEFAULT_MODEL = REPOSITORY_ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "build/rom-eval"
DEFAULT_LICENSE = REPOSITORY_ROOT / "LICENSE"
DEFAULT_MODEL_LICENSE = REPOSITORY_ROOT / "README.md"


@dataclass(frozen=True)
class PolicyInput:
    action_code: str
    task_id: str
    checkpoint_file: Path


def parse_policy(value: str) -> PolicyInput:
    """Parse ACTION_CODE=TASK_ID=CHECKPOINT.pt."""
    action_code, first_sep, remainder = value.partition("=")
    task_id, second_sep, checkpoint = remainder.partition("=")
    if not first_sep or not second_sep or not action_code or not task_id or not checkpoint:
        raise argparse.ArgumentTypeError("policy must be ACTION_CODE=TASK_ID=CHECKPOINT.pt")
    if action_code not in ACTION_RUNTIME_SPECS:
        raise argparse.ArgumentTypeError(f"unknown ROM action: {action_code}")
    if task_id not in ACTION_RUNTIME_SPECS[action_code].task_ids:
        expected = ", ".join(ACTION_RUNTIME_SPECS[action_code].task_ids)
        raise argparse.ArgumentTypeError(
            f"task {task_id!r} is not valid for {action_code}; expected {expected}"
        )
    return PolicyInput(action_code, task_id, Path(checkpoint))


def default_checkpoint_label(policies: tuple[PolicyInput, ...]) -> str:
    names = {policy.checkpoint_file.name for policy in policies}
    if len(names) == 1:
        return names.pop()
    return "mixed-checkpoints"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPOSITORY_ROOT).strip()


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, check=True)


def release_action_config(
    action_code: str,
    *,
    mandatory: bool,
    seeds: tuple[int, ...],
    max_steps: int,
) -> dict[str, object]:
    spec = ACTION_RUNTIME_SPECS[action_code]
    metric, operator = spec.qualification_metric_operators[0]
    if action_code == "STAND":
        threshold = spec.qualification_completion_metric_max or 0.08
    elif action_code == "SQUAT_REFERENCE":
        threshold = SQUAT_RETURN_LIMITS.crouch_height_max_m
    elif operator == "lte":
        threshold = 10.0
    else:
        threshold = 0.0
    thresholds = QualificationThresholds(
        minSuccessRate=1.0,
        maxFallRate=0.0,
        maxMeanTrackingError=10.0,
        minMeanDistanceM=0.0,
        maxMeanEnergyProxy=10_000.0,
        maxActuatorClampSteps=100,
        maxPhysicalJointLimitViolations=0,
        actionMetric=metric,
        actionMetricOperator=operator,
        actionMetricThreshold=threshold,
    )
    # Squat completion is phase-latched after the descent/return cycle; the
    # governed runtime needs at least 250 control steps at 50 Hz to observe
    # the full 5 s reference period.  Keep the CLI's shorter smoke default
    # for walk/stand while making squat batteries physically meaningful.
    effective_max_steps = max_steps
    if action_code == "SQUAT_REFERENCE":
        effective_max_steps = max(max_steps, 250)
    return {
        "actionCode": action_code,
        "mandatory": mandatory,
        "terrain": spec.qualification_terrain,
        "resetProfile": spec.reset_profile,
        "seeds": list(seeds),
        "maxSteps": effective_max_steps,
        "parameters": dict(spec.qualification_parameters),
        "thresholds": thresholds.model_dump(mode="json"),
    }


def write_release_config(
    path: Path,
    *,
    release: str,
    created_at: datetime,
    action_codes: tuple[str, ...],
    mandatory_actions: set[str],
    seeds: tuple[int, ...],
    max_steps: int,
) -> None:
    document = {
        "schema": "MICRODUCK_ROM_RELEASE_V1",
        "release": release,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "actions": [
            release_action_config(
                action_code,
                mandatory=action_code in mandatory_actions,
                seeds=seeds,
                max_steps=max_steps,
            )
            for action_code in action_codes
        ],
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def stamp_policy(
    onnx_path: Path,
    *,
    task_id: str,
    source_commit: str,
    checkpoint_label: str,
    experiment_ref: str,
) -> None:
    from export import attach_microduck_metadata

    attach_microduck_metadata(
        onnx_path,
        task_id=task_id,
        source_commit=source_commit,
        checkpoint=checkpoint_label,
        run_identity=experiment_ref,
    )


def export_policy(
    policy: PolicyInput,
    output: Path,
    *,
    num_envs: int,
    device: str,
    motion_file: Path | None,
) -> None:
    command = [
        sys.executable,
        "scripts/export.py",
        policy.task_id,
        "--checkpoint-file",
        str(policy.checkpoint_file),
        "--onnx-file",
        str(output),
        "--num-envs",
        str(num_envs),
        "--device",
        device,
    ]
    if motion_file is not None and policy.action_code == "SQUAT_REFERENCE":
        command.extend(["--motion-file", str(motion_file)])
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    if motion_file is not None and policy.action_code == "SQUAT_REFERENCE":
        # The task reset hook reads the validated reference from the
        # environment; export.py's CLI option configures the command object
        # but does not populate this task-level variable.
        env["MICRODUCK_REFERENCE_MOTION"] = str(motion_file)
    run(command, env=env)


def build_bundle(
    *,
    candidate_release: str,
    output_zip: Path,
    policy_paths: dict[str, Path],
    source_commit: str,
    checkpoint_label: str,
    experiment_ref: str,
    created_at: datetime,
    model: Path,
    software_license_id: str,
    software_license_file: Path,
    model_license_id: str,
    model_license_status: str,
    model_license_file: Path,
) -> None:
    command = [
        sys.executable,
        "scripts/build_rom_bundle.py",
        "--release",
        candidate_release,
        "--output",
        str(output_zip),
        "--model",
        str(model),
        "--terrain",
        "flat",
        "--scenario-profile",
        "SEEDED_SERVO_RESET_V1",
        "--source-commit",
        source_commit,
        "--checkpoint",
        checkpoint_label,
        "--experiment-ref",
        experiment_ref,
        "--created-at",
        created_at.isoformat(),
        "--software-license-id",
        software_license_id,
        "--software-license-file",
        str(software_license_file),
        "--model-license-id",
        model_license_id,
        "--model-license-status",
        model_license_status,
        "--model-license-file",
        str(model_license_file),
    ]
    for action_code, path in sorted(policy_paths.items()):
        command.extend(["--artifact", f"{action_code}={path}"])
    run(command)


def qualify_candidate(candidate_dir: Path, release_config: Path, output_zip: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    run(
        [
            sys.executable,
            "scripts/qualify_rom_bundle.py",
            "--bundle-dir",
            str(candidate_dir),
            "--release-config",
            str(release_config),
            "--output",
            str(output_zip),
        ],
        env=env,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        type=parse_policy,
        required=True,
        help="repeatable ACTION_CODE=TASK_ID=CHECKPOINT.pt",
    )
    parser.add_argument("--candidate-release", required=True)
    parser.add_argument("--qualified-release", required=True)
    parser.add_argument("--experiment-ref", required=True)
    parser.add_argument("--checkpoint-label")
    parser.add_argument("--source-commit", default=git_head())
    parser.add_argument("--created-at", default=datetime.now(UTC).isoformat())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stamp", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--seeds", default="7,11,29")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--mandatory-action", action="append", default=[])
    parser.add_argument("--mandatory-all", action="store_true")
    parser.add_argument("--software-license-id", default="Apache-2.0")
    parser.add_argument("--software-license-file", type=Path, default=DEFAULT_LICENSE)
    parser.add_argument("--model-license-id", default="LicenseRef-MicroDuck-CC-BY-SA-NC")
    parser.add_argument(
        "--model-license-status",
        choices=("DEVELOPMENT_ONLY", "DISTRIBUTION_CLEARED"),
        default="DEVELOPMENT_ONLY",
    )
    parser.add_argument("--model-license-file", type=Path, default=DEFAULT_MODEL_LICENSE)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-qualify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policies = tuple(args.policy)
    created_at = datetime.fromisoformat(args.created_at.replace("Z", "+00:00"))
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must contain at least three unique integers")
    action_codes = tuple(policy.action_code for policy in policies)
    if len(action_codes) != len(set(action_codes)):
        raise SystemExit("each action may be supplied only once")
    mandatory_actions = set(action_codes if args.mandatory_all else args.mandatory_action)
    unknown_mandatory = mandatory_actions - set(action_codes)
    if unknown_mandatory:
        raise SystemExit(f"mandatory actions were not supplied: {sorted(unknown_mandatory)}")

    output_root = (args.output_root / args.stamp).resolve()
    onnx_dir = output_root / "onnx"
    candidate_dir = output_root / "candidate"
    output_root.mkdir(parents=True, exist_ok=False)
    onnx_dir.mkdir()

    checkpoint_label = args.checkpoint_label or default_checkpoint_label(policies)
    policy_paths: dict[str, Path] = {}
    for policy in policies:
        onnx_path = onnx_dir / f"{policy.action_code.lower()}.onnx"
        if not args.skip_export:
            export_policy(
                policy,
                onnx_path,
                num_envs=args.num_envs,
                device=args.device,
                motion_file=args.motion_file,
            )
        if not onnx_path.is_file():
            raise SystemExit(f"missing exported ONNX: {onnx_path}")
        stamp_policy(
            onnx_path,
            task_id=policy.task_id,
            source_commit=args.source_commit,
            checkpoint_label=checkpoint_label,
            experiment_ref=args.experiment_ref,
        )
        policy_paths[policy.action_code] = onnx_path

    candidate_zip = output_root / f"microduck-candidate-{args.candidate_release}.zip"
    build_bundle(
        candidate_release=args.candidate_release,
        output_zip=candidate_zip,
        policy_paths=policy_paths,
        source_commit=args.source_commit,
        checkpoint_label=checkpoint_label,
        experiment_ref=args.experiment_ref,
        created_at=created_at,
        model=args.model,
        software_license_id=args.software_license_id,
        software_license_file=args.software_license_file,
        model_license_id=args.model_license_id,
        model_license_status=args.model_license_status,
        model_license_file=args.model_license_file,
    )
    candidate_dir.mkdir()
    with zipfile.ZipFile(candidate_zip) as archive:
        archive.extractall(candidate_dir)

    release_config = output_root / "release.json"
    write_release_config(
        release_config,
        release=args.qualified_release,
        created_at=created_at,
        action_codes=action_codes,
        mandatory_actions=mandatory_actions,
        seeds=seeds,
        max_steps=args.max_steps,
    )

    qualified_zip = output_root / f"microduck-qualified-{args.qualified_release}.zip"
    if args.skip_qualify:
        print(f"Skipped qualification; release config written to {release_config}")
    else:
        qualify_candidate(candidate_dir, release_config, qualified_zip)

    summary = {
        "outputRoot": str(output_root),
        "candidateZip": str(candidate_zip),
        "candidateDir": str(candidate_dir),
        "releaseConfig": str(release_config),
        "qualifiedZip": str(qualified_zip) if qualified_zip.exists() else None,
        "sourceCommit": args.source_commit,
        "checkpoint": checkpoint_label,
        "experimentRef": args.experiment_ref,
        "actions": list(action_codes),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
