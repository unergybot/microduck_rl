"""Instantiate collision geometry from the same immutable scene used by the planner."""

from xml.etree import ElementTree as ET


def add_geometry(model_path, scene):
    tree = ET.parse(model_path)
    world = tree.getroot().find("worldbody")
    if world is None:
        raise ValueError("navigation model has no world")
    for index, obstacle in enumerate(scene.obstacles):
        ET.SubElement(
            world,
            "geom",
            name=f"rom_navigation_obstacle_{index}",
            type="box",
            pos=f"{(obstacle.minX + obstacle.maxX) / 2} {(obstacle.minY + obstacle.maxY) / 2} 0.15",
            size=f"{(obstacle.maxX - obstacle.minX) / 2} {(obstacle.maxY - obstacle.minY) / 2} 0.15",
            contype="0",
            conaffinity="0",
            rgba="0.4 0.4 0.4 0",
        )
    tree.write(model_path, encoding="utf-8")
