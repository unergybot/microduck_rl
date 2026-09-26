"""Draw the installed navigation scene from its approved obstacle footprint."""

from math import cos, isclose, sin
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


def add_apriltag_probe(model_path, scene):
    """Place calibrated visual-only AprilTags in the MuJoCo scene."""
    import cv2

    from .apriltag import PROBE_TAGS, tag_texture

    expected = {"home": (0.0, 0.0), "desk": (1.0, 0.0), "door": (0.0, 1.0)}
    if (
        getattr(scene, "revision", None) != "microduck-navigation-calibration-v1"
        or any(
            name not in scene.landmarks
            or not isclose(scene.landmarks[name].x, xy[0], abs_tol=1e-9)
            or not isclose(scene.landmarks[name].y, xy[1], abs_tol=1e-9)
            for name, xy in expected.items()
        )
        or not scene.obstacles
        or any(
            not isclose(getattr(scene.obstacles[0], field), value, abs_tol=1e-9)
            for field, value in (
                ("minX", 0.4),
                ("maxX", 0.6),
                ("minY", -0.1),
                ("maxY", 0.3),
            )
        )
    ):
        raise ValueError("AprilTag probe requires the calibrated navigation scene")

    tree = ET.parse(model_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("navigation model has no world")
    for probe in PROBE_TAGS:
        texture_path = model_path.parent / f"rom_apriltag_{probe.tag_id}.png"
        if not cv2.imwrite(str(texture_path), tag_texture(probe.tag_id)):
            raise ValueError("could not write offline AprilTag texture")
        texture_name = f"rom_apriltag_texture_{probe.tag_id}"
        material_name = f"rom_apriltag_material_{probe.tag_id}"
        ET.SubElement(
            asset,
            "texture",
            name=texture_name,
            type="2d",
            file=texture_path.name,
        )
        ET.SubElement(
            asset,
            "material",
            name=material_name,
            texture=texture_name,
            texuniform="false",
            emission="1",
        )
        ET.SubElement(
            world,
            "geom",
            name=f"rom_apriltag_probe_{probe.tag_id}",
            type="plane",
            pos=" ".join(map(str, probe.center_xyz)),
            quat=" ".join(map(str, probe.quat_wxyz)),
            size="0.0875 0.0875 0.001",
            contype="0",
            conaffinity="0",
            material=material_name,
        )
    tree.write(model_path, encoding="utf-8")
