import json
import math
from pathlib import Path

from mjlab_microduck.rom.navigation.follower import Navigator
from mjlab_microduck.rom.navigation.planner import plan_cells
from mjlab_microduck.rom.navigation_contracts import NavigationProfile, Pose, Scene


def setup_nav(goal="desk"):
    c = json.loads(Path("tests/fixtures/navigation/candidate.json").read_text())
    return Navigator(
        Scene.model_validate(c["scene"]),
        NavigationProfile.model_validate(c["profile"]),
        goal,
        0.0,
    )


def test_obstacle_detour_and_unreachable_goal():
    path = plan_cells({(1, 0)}, (0, 0), (2, 0), (3, 3))
    assert path[0] == (0, 0) and path[-1] == (2, 0) and (1, 0) not in path
    assert plan_cells({(1, 0), (1, 1), (1, 2)}, (0, 0), (2, 0), (3, 3)) is None
    assert plan_cells(set(), (0, 0), (-1, 0), (3, 3)) is None
    # A clearance preference must not turn a valid narrow corridor into a wall.
    blocked = {(x, y) for x in range(5) for y in (0, 2)}
    assert plan_cells(blocked, (0, 1), (4, 1), (5, 3), prefer_clearance=True) == [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
    ]


def test_heading_first_never_strafes_or_reverses():
    nav = setup_nav("door")
    command = nav.update(
        Pose(x=0.0, y=0.0, yaw=0.0), now=0.0, captured=0.0, speed=0.0, yaw_rate=0.0
    )
    assert command.vx == 0 and command.vy == 0 and 0 < command.yaw <= 0.2


def test_stale_pose_and_deadline_stop():
    nav = setup_nav()
    result = nav.update(
        Pose(x=0.0, y=0.0, yaw=0.0), now=1.0, captured=0.0, speed=0.0, yaw_rate=0.0
    )
    assert result.reason == "LOCALIZATION_FAILED" and result.vx == 0 and result.yaw == 0
    nav = setup_nav()
    result = nav.update(
        Pose(x=0.0, y=0.0, yaw=0.0), now=181.0, captured=181.0, speed=0.0, yaw_rate=0.0
    )
    assert result.reason == "DEADLINE_EXCEEDED"


def test_arrival_requires_continuous_settling_and_measured_motion_stop():
    nav = setup_nav("home")
    pose = Pose(x=0.0, y=0.0, yaw=0.0)
    assert not nav.update(pose, now=0.0, captured=0.0, speed=0.0, yaw_rate=0.0).arrived
    assert not nav.update(pose, now=0.6, captured=0.6, speed=0.1, yaw_rate=0.0).arrived
    assert not nav.update(pose, now=0.8, captured=0.8, speed=0.0, yaw_rate=0.0).arrived
    assert nav.update(pose, now=1.4, captured=1.4, speed=0.0, yaw_rate=0.0).arrived


def test_grid_blocks_clearance_near_obstacle_and_boundary():
    nav = setup_nav()
    for pose in [Pose(x=0.3, y=0.0, yaw=0.0), Pose(x=-0.9, y=0.0, yaw=0.0)]:
        result = nav.update(pose, now=0.0, captured=0.0, speed=0.0, yaw_rate=0.0)
        assert result.reason == "PATH_BLOCKED"


def test_fixed_seed_kinematic_routes_are_repeatable():
    # Controller-only regression. Physics qualification uses a separate runner.
    for seed in range(10):
        nav = setup_nav("desk")
        x, y, yaw = 0.0, 0.0, 0.0
        for i in range(10000):
            t = i * 0.02
            out = nav.update(
                Pose(x=x, y=y, yaw=yaw), now=t, captured=t, speed=0.0, yaw_rate=0.0
            )
            if out.arrived:
                break
            assert out.reason is None
            x += out.vx * math.cos(yaw) * 0.02
            y += out.vx * math.sin(yaw) * 0.02
            yaw = math.atan2(
                math.sin(yaw + out.yaw * 0.02), math.cos(yaw + out.yaw * 0.02)
            )
        assert out.arrived and math.hypot(x - 1, y) <= 0.08


def test_local_drift_replans_to_same_goal_when_route_exists():
    nav = setup_nav()
    nav.update(
        Pose(x=0.0, y=0.0, yaw=0.0), now=0.0, captured=0.0, speed=0.0, yaw_rate=0.0
    )
    result = nav.update(
        Pose(x=0.2, y=-0.5, yaw=0.0), now=0.1, captured=0.1, speed=0.0, yaw_rate=0.0
    )
    assert result.reason is None


def test_final_heading_uses_effective_turn_command_without_forward_motion():
    nav = setup_nav("home")
    out = nav.update(
        Pose(x=0.0, y=0.0, yaw=-0.18), now=0.0, captured=0.0, speed=0.0, yaw_rate=0.0
    )
    assert out.vx == 0.0 and out.yaw == 0.2


def test_near_destination_does_not_decay_into_policy_deadband():
    nav = setup_nav("home")
    nav.profile = nav.profile.model_copy(update={"maxSpeedMps": 0.2})
    out = nav.update(
        Pose(x=-0.09, y=0.0, yaw=0.1), now=0.0, captured=0.0, speed=0.0, yaw_rate=0.0
    )
    assert out.vx == 0.2 and out.yaw == 0.0


def test_heading_drift_stops_advance_and_commands_full_bounded_turn():
    nav = setup_nav("home")
    out = nav.update(
        Pose(x=-0.3, y=0.0, yaw=0.18), now=0.0, captured=0.0, speed=0.0, yaw_rate=0.0
    )
    assert out.vx == 0.0 and out.yaw == -0.2


def test_detour_prefers_two_cells_for_measured_tracking_drift():
    nav = setup_nav("desk")
    nav.update(
        Pose(x=0.0, y=0.0, yaw=0.0), now=0.0, captured=0.0, speed=0.0, yaw_rate=0.0
    )
    for point in nav.path:
        x, y = nav.grid.cell(*point)
        assert all(
            (x + dx, y + dy) not in nav.grid.blocked
            for dx in range(-2, 3)
            for dy in range(-2, 3)
        )
