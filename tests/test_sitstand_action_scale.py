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


def test_sitstand_curriculum_prioritizes_qualification_rise() -> None:
    cfg = make_microduck_sitstand_env_cfg()
    event = cfg.events["set_ground_state"]
    assert event.params["sitting_prob"] > event.params["standing_prob"]
    assert event.params["sitting_prob"] == 0.75
    assert event.params["standing_prob"] == 0.25
    # The raw command is STAND more often than SIT; together with the reset
    # mix this makes seated→STAND the dominant transition.
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SIT_PROB
    assert SIT_PROB == 0.25
