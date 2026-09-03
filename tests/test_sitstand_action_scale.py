from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg


def test_sitstand_policy_action_scale_has_dynamic_limit_margin() -> None:
    cfg = make_microduck_sitstand_env_cfg()
    assert cfg.actions["joint_pos"].scale == 0.8


def test_sitstand_policy_keeps_commanded_targets_inside_joint_limits() -> None:
    cfg = make_microduck_sitstand_env_cfg()
    term = cfg.rewards["action_limit_margin"]
    assert term.weight <= -5.0
    assert term.params["action_name"] == "joint_pos"
    assert term.params["margin"] == 0.15


def test_sitstand_starts_with_strong_action_rate_damping() -> None:
    cfg = make_microduck_sitstand_env_cfg()
    assert cfg.rewards["action_rate_l2"].weight <= -0.5
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["weight"] <= -0.5
