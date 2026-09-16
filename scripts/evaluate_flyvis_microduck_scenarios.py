#!/usr/bin/env python3
"""Motion-required MicroDuck scenarios with synchronized outside/head-camera video."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("FLYVIS_ROOT_DIR", str(Path.home() / ".cache/microduck-flyvis"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("collect", "train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
        if name in ("collect", "evaluate"):
            p.add_argument("--count", type=int, default=30 if name == "collect" else 10)
            p.add_argument("--seconds", type=float, default=20.0)
        if name == "collect":
            p.add_argument("--validation-count", type=int, default=10)
        if name == "train":
            p.add_argument("--dataset", type=Path, required=True)
            p.add_argument("--kind", choices=("flyvis", "retina"), default="flyvis")
            p.add_argument("--seed", type=int, default=1)
            p.add_argument("--rounds", type=int, default=1)
            p.add_argument("--round-episodes", type=int, default=20)
            p.add_argument("--epochs", type=int, default=30)
        if name == "evaluate":
            p.add_argument(
                "--kind",
                choices=("flyvis", "retina"),
                default="flyvis",
                help="visual feature adapter for the privileged teacher; learned checkpoints declare their own kind",
            )
            role = p.add_mutually_exclusive_group(required=True)
            role.add_argument("--checkpoint", type=Path)
            role.add_argument(
                "--teacher",
                action="store_true",
                help="explicit privileged reference, not learned control",
            )
            p.add_argument(
                "--interventions",
                nargs="+",
                choices=(
                    "none",
                    "frozen_camera",
                    "zero_vision",
                    "no_body",
                    "stale_camera",
                ),
                default=["none"],
            )
    p = sub.add_parser("replay")
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    name = args.pop("command")
    if name == "prepare":
        from mjlab_microduck.rom.flyvis_scenarios import prepare_scenarios

        prepare_scenarios(args["source"], args["output"])
    elif name == "replay":
        from mjlab_microduck.rom.flyvis_scenarios import replay_scenario

        replay_scenario(**args)
    else:
        from mjlab_microduck.rom import flyvis_scenario_training

        args.pop("teacher", None)
        getattr(flyvis_scenario_training, name)(**args)


if __name__ == "__main__":
    main()
