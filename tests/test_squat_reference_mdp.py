import torch

from mjlab_microduck.tasks.mdp import (
    squat_completion_latch_update,
    squat_phase_from_command,
    squat_reference_indices,
)


def test_reference_indices_interpolate_first_middle_and_last_frames():
    lower, upper, alpha = squat_reference_indices(
        torch.tensor([0.0, 0.25, 0.5, 1.0]), frames=3
    )

    assert lower.tolist() == [0, 0, 1, 2]
    assert upper.tolist() == [1, 1, 2, 2]
    torch.testing.assert_close(alpha, torch.tensor([0.0, 0.5, 0.0, 0.0]))


def test_completion_latch_pays_once_and_rearms_at_new_cycle():
    latched = torch.tensor([False, False])
    phase = torch.tensor([0.9, 0.9])
    pose_error = torch.tensor([0.05, 0.4])

    latched, reward = squat_completion_latch_update(
        latched, phase, pose_error, completion_phase=0.85, error_threshold=0.1
    )
    assert latched.tolist() == [True, False]
    assert reward.tolist() == [1.0, 0.0]

    latched, reward = squat_completion_latch_update(
        latched, phase, pose_error, completion_phase=0.85, error_threshold=0.1
    )
    assert reward.tolist() == [0.0, 0.0]

    latched, _ = squat_completion_latch_update(
        latched,
        torch.tensor([0.02, 0.02]),
        pose_error,
        completion_phase=0.85,
        error_threshold=0.1,
    )
    assert latched.tolist() == [False, False]


def test_phase_is_recovered_from_cyclic_command():
    phase = squat_phase_from_command(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(phase, torch.tensor([0.0, 0.25, 0.5]))
