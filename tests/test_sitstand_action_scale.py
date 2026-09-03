from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg


def test_sitstand_policy_action_scale_has_dynamic_limit_margin() -> None:
    cfg = make_microduck_sitstand_env_cfg()
    assert cfg.actions["joint_pos"].scale == 0.8


def test_sitstand_policy_penalizes_over_commanded_joint_targets() -> None:
    cfg = make_microduck_sitstand_env_cfg()
    term = cfg.rewards["action_over_limit"]
    assert term.weight < 0.0
    assert term.params["action_name"] == "joint_pos"
    assert term.params["overshoot"] == 0.05
