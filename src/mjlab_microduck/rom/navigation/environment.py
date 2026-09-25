"""Draw the installed navigation scene from its approved obstacle footprint."""

from math import cos, sin
from xml.etree import ElementTree as ET


def add_geometry(model_path, scene):
    tree = ET.parse(model_path)
    world = tree.getroot().find("worldbody")
    if world is None:
        raise ValueError("navigation model has no world")

    def marker(parent, name, pos, size, rgba):
        ET.SubElement(
            parent,
            "geom",
            name=name,
            type="box",
            pos=pos,
            size=size,
            contype="0",
            conaffinity="0",
            rgba=rgba,
        )

    calibrated = (
        getattr(scene, "revision", None) == "microduck-navigation-calibration-v1"
    )
    for index, obstacle in enumerate(scene.obstacles):
        center_x = (obstacle.minX + obstacle.maxX) / 2
        center_y = (obstacle.minY + obstacle.maxY) / 2
        half_x = (obstacle.maxX - obstacle.minX) / 2
        half_y = (obstacle.maxY - obstacle.minY) / 2
        if calibrated and index == 0:
            # In this calibrated scene obstacle 0 is the desk footprint;
            # the desk landmark is its approach pose, not its physical center.
            marker(
                world,
                f"rom_navigation_obstacle_{index}",
                f"{center_x} {center_y} 0.32",
                f"{half_x} {half_y} 0.018",
                "0.7 0.4 0.15 0.85",
            )
            for x_side, x in (
                ("left", obstacle.minX + 0.018),
                ("right", obstacle.maxX - 0.018),
            ):
                for y_side, y in (
                    ("front", obstacle.minY + 0.018),
                    ("back", obstacle.maxY - 0.018),
                ):
                    marker(
                        world,
                        f"rom_navigation_obstacle_{index}_leg_{x_side}_{y_side}",
                        f"{x} {y} 0.16",
                        "0.018 0.018 0.16",
                        "0.7 0.4 0.15 0.85",
                    )
        else:
            marker(
                world,
                f"rom_navigation_obstacle_{index}",
                f"{center_x} {center_y} 0.15",
                f"{half_x} {half_y} 0.15",
                "0.4 0.4 0.4 0.65",
            )
    for index, (name, pose) in enumerate(sorted(scene.landmarks.items())):
        body = ET.SubElement(
            world,
            "body",
            name=f"rom_landmark_{index}",
            pos=f"{pose.x} {pose.y} 0",
            quat=f"{cos(pose.yaw / 2)} 0 0 {sin(pose.yaw / 2)}",
        )
        if name == "door":
            for side, lateral in (("left", -0.28), ("right", 0.28)):
                marker(
                    body,
                    f"rom_door_{side}_{index}",
                    f"0 {lateral} 0.19",
                    "0.025 0.025 0.19",
                    "0.2 0.45 0.9 1",
                )
            marker(
                body,
                f"rom_door_header_{index}",
                "0 0 0.39",
                "0.025 0.305 0.02",
                "0.2 0.45 0.9 1",
            )
        elif name == "desk" and not calibrated:
            marker(
                body,
                f"rom_desk_top_{index}",
                "0 0 0.32",
                "0.18 0.18 0.018",
                "0.7 0.4 0.15 0.85",
            )
            for side, lateral in (("left", -0.15), ("right", 0.15)):
                marker(
                    body,
                    f"rom_desk_leg_{side}_{index}",
                    f"0 {lateral} 0.16",
                    "0.018 0.018 0.16",
                    "0.7 0.4 0.15 0.85",
                )
        else:
            marker(
                body,
                f"rom_landmark_pad_{index}",
                "0 0 0.004",
                "0.09 0.09 0.004",
                "0.2 0.7 0.35 0.8",
            )
    tree.write(model_path, encoding="utf-8")
