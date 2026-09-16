#!/usr/bin/env python3
"""Local embodied visual tracking; all commands operate on offline MuJoCo instances."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("FLYVIS_ROOT_DIR", str(Path.home() / ".cache/microduck-flyvis"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="copy and bind an experiment bundle")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("collect", "train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument(
            "--model",
            type=Path,
            required=True,
            help="official pretrained results/flow/0000/000 directory",
        )
        p.add_argument("--output", type=Path, required=True)
        p.add_argument(
            "--jobs",
            type=int,
            choices=range(1, 9),
            default=1,
            help="independent offline simulator processes (1–8)",
        )
        if name in ("collect", "evaluate"):
            p.add_argument("--count", type=int, default=80 if name == "collect" else 50)
            p.add_argument("--seconds", type=float, default=20.0)
        if name == "collect":
            p.add_argument("--validation-count", type=int, default=20)
        if name == "train":
            p.add_argument("--dataset", type=Path, required=True)
            p.add_argument("--kind", choices=("flyvis", "retina"), default="flyvis")
            p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
            p.add_argument("--rounds", type=int, default=2)
            p.add_argument("--round-episodes", type=int, default=40)
            p.add_argument("--epochs", type=int, default=50)
        if name == "evaluate":
            p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
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
                default=["none", "frozen_camera", "zero_vision", "no_body"],
            )
    p = sub.add_parser("replay")
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from mjlab_microduck.rom.flyvis_tracking.simulation import prepare_bundle

        prepare_bundle(args["source"], args["output"])
    else:
        from mjlab_microduck.rom.flyvis_tracking import experiment

        getattr(experiment, command)(**args)


if __name__ == "__main__":
    main()
