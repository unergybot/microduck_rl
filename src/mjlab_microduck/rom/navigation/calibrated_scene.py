"""Canonical, versioned calibration scene used by navigation and MuJoCo."""

import json
from pathlib import Path

from ..navigation_contracts import Scene

V2_REVISION = "microduck-navigation-calibration-v2"
_V2_DOCUMENT = Path(__file__).with_name("calibrated_v2.json")


def canonical_v2_scene() -> Scene:
    return Scene.model_validate(json.loads(_V2_DOCUMENT.read_text())["scene"])


def validate_v2_scene(scene: Scene) -> None:
    if getattr(scene, "revision", None) == V2_REVISION and scene.model_dump() != canonical_v2_scene().model_dump():
        raise ValueError("noncanonical v2 scene")
