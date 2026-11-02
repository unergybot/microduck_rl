"""Config-only regressions for rough train/play factory aliasing; no Scene builds."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
)
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
)
from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
)
from mjlab_microduck.workbench import resolve_experiment
from mjlab_microduck.workbench.engine import build_configs


FACTORIES = (
    make_microduck_velocity_env_cfg,
    make_microduck_ground_pick_env_cfg,
    make_microduck_standup_env_cfg,
    make_microduck_sitstand_env_cfg,
    make_microduck_velstand_env_cfg,
)
ROUGH_TASKS = (
    "Mjlab-Velocity-Rough-MicroDuck",
    "Mjlab-Velocity-Rough-Backlash-MicroDuck",
    "Mjlab-GroundPick-Rough-MicroDuck",
    "Mjlab-GroundPick-Rough-Backlash-MicroDuck",
    "Mjlab-StandUp-Rough-MicroDuck",
    "Mjlab-StandUp-Rough-Backlash-MicroDuck",
    "Mjlab-SitStand-Rough-MicroDuck",
    "Mjlab-SitStand-Rough-Backlash-MicroDuck",
    "Mjlab-VelStand-Rough-MicroDuck",
    "Mjlab-VelStand-Rough-Backlash-MicroDuck",
)


def assert_defaults(terrain, play=False):
    assert (terrain.num_rows, terrain.num_cols, terrain.curriculum) == (
        (5, 5, False) if play else (10, 20, False)
    )
    assert terrain.sub_terrains["random_grid"].grid_height_range == (0.0, 0.010)
    assert terrain.sub_terrains["random_grid"].proportion == 0.30


def assert_detached(first, second):
    assert first is not second
    assert first.sub_terrains is not second.sub_terrains
    for name in first.sub_terrains:
        assert first.sub_terrains[name] is not second.sub_terrains[name]


def test_rough_terrain_registration_in_fresh_process_preserves_training_defaults():
    # No global repair: the child's first tasks import must register correct configs.
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import mjlab_microduck.tasks
            from mjlab.tasks.registry import list_tasks, load_env_cfg
            from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
                MICRODUCK_ROUGH_TERRAINS_CFG as terrain,
            )
            assert (terrain.num_rows, terrain.num_cols, terrain.curriculum) == (10, 20, False)
            expected = {
                f"Mjlab-{family}-Rough{variant}-MicroDuck"
                for family in ("Velocity", "GroundPick", "StandUp", "SitStand", "VelStand")
                for variant in ("", "-Backlash")
            }
            assert {
                task for task in list_tasks()
                if "-Rough-" in task and task.endswith("-MicroDuck")
            } == expected
            for task in sorted(expected):
                for play, shape in ((False, (10, 20, False)), (True, (5, 5, False))):
                    cfg = load_env_cfg(task, play=play)
                    generator = cfg.scene.terrain.terrain_generator
                    assert (generator.num_rows, generator.num_cols, generator.curriculum) == shape, task
                    assert generator.sub_terrains["random_grid"].grid_height_range == (0.0, 0.010), task
                    assert "terrain_levels" in cfg.curriculum, task
            print("fresh registration: global and all 10 rough train/play pairs correct")
        """)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "",
             "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all 10 rough train/play pairs correct" in result.stdout


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda factory: factory.__name__)
@pytest.mark.parametrize("order", [(False, True), (True, False)], ids=["train-play", "play-train"])
def test_rough_terrain_factory_order_and_nested_mutations_are_isolated(factory, order):
    configs = [factory(rough=True, play=play) for play in order]
    terrains = [cfg.scene.terrain.terrain_generator for cfg in configs]
    for terrain, play in zip(terrains, order):
        assert_defaults(terrain, play)
        assert_detached(terrain, MICRODUCK_ROUGH_TERRAINS_CFG)
    assert_detached(*terrains)
    assert all("terrain_levels" in cfg.curriculum for cfg in configs)

    terrains[0].num_rows = 99
    terrains[0].curriculum = True
    terrains[0].sub_terrains["random_grid"].grid_height_range = (0.4, 0.5)
    terrains[0].sub_terrains["random_grid"].proportion = 0.9
    assert_defaults(terrains[1], order[1])
    assert_defaults(MICRODUCK_ROUGH_TERRAINS_CFG)
    # Repeat this factory and cross every rough family after a caller's mutation.
    for next_factory in FACTORIES:
        for play in (False, True):
            next_terrain = next_factory(rough=True, play=play).scene.terrain.terrain_generator
            assert_defaults(next_terrain, play)
            assert_detached(next_terrain, terrains[0])
            assert_detached(next_terrain, terrains[1])
            assert_detached(next_terrain, MICRODUCK_ROUGH_TERRAINS_CFG)
    assert_defaults(terrains[1], order[1])
    assert_defaults(MICRODUCK_ROUGH_TERRAINS_CFG)


@pytest.mark.parametrize("task", ROUGH_TASKS)
@pytest.mark.parametrize("play", [False, True], ids=["train", "play"])
def test_rough_terrain_registry_loads_are_independently_modifiable(task, play):
    first_cfg = load_env_cfg(task, play=play)
    second_cfg = load_env_cfg(task, play=play)
    first = first_cfg.scene.terrain.terrain_generator
    second = second_cfg.scene.terrain.terrain_generator
    assert_defaults(first, play)
    assert_defaults(second, play)
    assert_detached(first, second)
    first.num_cols = 99
    first.sub_terrains["random_grid"].grid_height_range = (0.4, 0.5)
    second.num_rows = 88
    second.sub_terrains["pyramid_stairs"].step_height_range = (0.3, 0.4)
    assert second.num_cols == (5 if play else 20)
    assert second.sub_terrains["random_grid"].grid_height_range == (0.0, 0.010)
    assert first.num_rows == (5 if play else 10)
    assert first.sub_terrains["pyramid_stairs"].step_height_range == (0.0, 0.015)
    assert_defaults(load_env_cfg(task, play=play).scene.terrain.terrain_generator, play)
    assert_defaults(load_env_cfg(task, play=not play).scene.terrain.terrain_generator, not play)
    assert_defaults(MICRODUCK_ROUGH_TERRAINS_CFG)


@pytest.mark.parametrize("recipe", ["Mjlab-Velocity-Rough-MicroDuck", "Mjlab-StandUp-Rough-Backlash-MicroDuck"])
@pytest.mark.parametrize("profile", ["rom-xl330-v1", "hd1910-reference-v1"])
def test_rough_terrain_workbench_uses_training_grid_without_adapting_registry(recipe, profile):
    registered = load_env_cfg(recipe)
    original_actuator = registered.scene.entities["robot"].articulation.actuators[0]
    spec = resolve_experiment({
        "schemaVersion": "ROM_TRAINING_EXPERIMENT_V1",
        "recipeId": recipe,
        "robotProfileId": profile,
        "config": {"gains": {"kpFirmware": 180}},
    })
    env, _ = build_configs(spec)
    terrain = env.scene.terrain.terrain_generator
    assert_defaults(terrain)
    assert "terrain_levels" in env.curriculum
    assert env.scene.entities["robot"].articulation.actuators[0].kp_fw == 180
    terrain.num_rows = 77
    terrain.sub_terrains["random_grid"].grid_height_range = (0.4, 0.5)
    fresh = load_env_cfg(recipe)
    assert_defaults(fresh.scene.terrain.terrain_generator)
    fresh_actuator = fresh.scene.entities["robot"].articulation.actuators[0]
    assert fresh_actuator.motor_name == original_actuator.motor_name == "xl330"
    assert fresh_actuator.kp_fw == original_actuator.kp_fw
    assert_defaults(build_configs(spec)[0].scene.terrain.terrain_generator)
    assert_defaults(MICRODUCK_ROUGH_TERRAINS_CFG)
