"""Reference-assisted squat task for Blender-authored MicroDuck motions."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    MicroduckGroundPickRlCfg,
    make_microduck_ground_pick_env_cfg,
)


_GROUND_PICK_ONLY_REWARDS = (
    "mouth_ground_proximity",
    "mouth_perpendicular_to_ground",
    "ground_pick_return_pose_legs",
    "ground_pick_return_pose_neck",
    "return_upright",
    "neck_vel_descent",
    "mouth_payload_force",
    "head_impact_penalty",
)


def make_microduck_squat_reference_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat reference-assisted squat environment configuration."""
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=False)

    for reward_name in _GROUND_PICK_ONLY_REWARDS:
        cfg.rewards.pop(reward_name, None)
    cfg.events.pop("sample_mouth_payload", None)

    cfg.commands["twist"].period = 4.0
    cfg.commands["twist"].randomize_phase = True

    robot_cfg = SceneEntityCfg("robot")
    cfg.rewards["reference_joint"] = RewardTermCfg(
        func=microduck_mdp.squat_reference_joint_track,
        weight=4.0,
        params={"asset_cfg": robot_cfg, "command_name": "twist", "std": 0.35},
    )
    cfg.rewards["reference_height"] = RewardTermCfg(
        func=microduck_mdp.squat_reference_height_track,
        weight=2.0,
        params={"asset_cfg": robot_cfg, "command_name": "twist", "std": 0.04},
    )
    cfg.rewards["reference_completion"] = RewardTermCfg(
        func=microduck_mdp.squat_reference_completion,
        weight=8.0,
        params={
            "asset_cfg": robot_cfg,
            "command_name": "twist",
            "completion_phase": 0.85,
            "error_threshold": 0.12,
        },
    )

    cfg.events["reset_reference_state"] = EventTermCfg(
        func=microduck_mdp.reset_squat_reference_state,
        mode="reset",
        params={"asset_cfg": robot_cfg, "command_name": "twist", "probability": 0.25},
    )
    cfg.events["reset_squat_latch"] = EventTermCfg(
        func=microduck_mdp.reset_squat_latch,
        mode="reset",
    )

    return cfg


MicroduckSquatReferenceRlCfg = deepcopy(MicroduckGroundPickRlCfg)
MicroduckSquatReferenceRlCfg.experiment_name = "squat_reference"
MicroduckSquatReferenceRlCfg.run_name = "squat_reference"
MicroduckSquatReferenceRlCfg.save_interval = 5
MicroduckSquatReferenceRlCfg.max_iterations = 20_000
