from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_squat_reference_env_cfg import (
    MicroduckSquatReferenceRlCfg,
    make_microduck_squat_reference_env_cfg,
)


def test_reference_task_keeps_contract_and_replaces_ground_pick_objectives():
    cfg = make_microduck_squat_reference_env_cfg()

    assert cfg.actions["joint_pos"].scale == 1.0
    assert cfg.commands["twist"].randomize_phase is True
    assert cfg.rewards["reference_joint"].func is mdp.squat_reference_joint_track
    assert cfg.rewards["reference_height"].func is mdp.squat_reference_height_track
    assert cfg.rewards["reference_completion"].func is mdp.squat_reference_completion
    assert "mouth_ground_proximity" not in cfg.rewards
    assert "sample_mouth_payload" not in cfg.events
    assert "reset_reference_state" in cfg.events
    assert "reset_squat_latch" in cfg.events


def test_reference_runner_uses_normalized_ppo_without_changing_dimensions():
    assert MicroduckSquatReferenceRlCfg.actor.obs_normalization is True
    assert MicroduckSquatReferenceRlCfg.critic.obs_normalization is True
    assert MicroduckSquatReferenceRlCfg.experiment_name == "squat_reference"
