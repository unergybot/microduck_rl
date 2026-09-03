from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg


def test_sitstand_policy_action_scale_has_dynamic_limit_margin() -> None:
    cfg = make_microduck_sitstand_env_cfg()
    assert cfg.actions["joint_pos"].scale == 0.8
