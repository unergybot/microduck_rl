"""Draw the installed navigation scene without changing robot contact physics."""

from xml.etree import ElementTree as ET
from math import cos, sin


def add_geometry(model_path, scene):
    tree = ET.parse(model_path)
    world = tree.getroot().find("worldbody")
    if world is None:
        raise ValueError("navigation model has no world")
    def marker(parent, name, pos, size, rgba):
        ET.SubElement(parent, "geom", name=name, type="box", pos=pos,
                      size=size, contype="0", conaffinity="0", rgba=rgba)

    for index, obstacle in enumerate(scene.obstacles):
        marker(world, f"rom_navigation_obstacle_{index}",
               f"{(obstacle.minX + obstacle.maxX) / 2} {(obstacle.minY + obstacle.maxY) / 2} 0.15",
               f"{(obstacle.maxX - obstacle.minX) / 2} {(obstacle.maxY - obstacle.minY) / 2} 0.15",
               "0.4 0.4 0.4 0.65")
    for index, (name, pose) in enumerate(sorted(scene.landmarks.items())):
        body = ET.SubElement(world, "body", name=f"rom_landmark_{index}",
                             pos=f"{pose.x} {pose.y} 0",
                             quat=f"{cos(pose.yaw / 2)} 0 0 {sin(pose.yaw / 2)}")
        if name == "door":
            for side, lateral in (("left", -0.28), ("right", 0.28)):
                marker(body, f"rom_door_{side}_{index}", f"0 {lateral} 0.19",
                       "0.025 0.025 0.19", "0.2 0.45 0.9 1")
            marker(body, f"rom_door_header_{index}", "0 0 0.39",
                   "0.025 0.305 0.02", "0.2 0.45 0.9 1")
        elif name == "desk":
            marker(body, f"rom_desk_top_{index}", "0 0 0.32",
                   "0.18 0.18 0.018", "0.7 0.4 0.15 0.85")
            for side, lateral in (("left", -0.15), ("right", 0.15)):
                marker(body, f"rom_desk_leg_{side}_{index}", f"0 {lateral} 0.16",
                       "0.018 0.018 0.16", "0.7 0.4 0.15 0.85")
        else:
            marker(body, f"rom_landmark_pad_{index}", "0 0 0.004",
                   "0.09 0.09 0.004", "0.2 0.7 0.35 0.8")
    tree.write(model_path, encoding="utf-8")
