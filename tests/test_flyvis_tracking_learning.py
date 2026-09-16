import numpy as np
import pytest


def test_readout_learns_and_checkpoint_preserves_train_normalization(tmp_path):
    from mjlab_microduck.rom.flyvis_tracking.learning import Readout, fit_readout

    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 38)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    model, history = fit_readout(x, y, x, y, seed=1, epochs=100)
    assert (model.predict_batch(x) == y).mean() > 0.9
    model.save(tmp_path / "readout.npz", {"partition": "train"})
    restored = Readout.load(tmp_path / "readout.npz")
    assert np.array_equal(restored.predict_batch(x), model.predict_batch(x))
    assert np.allclose(restored.mean, x.mean(axis=0))
    assert history[-1]["validationAccuracy"] > 0.9
    with pytest.raises(ValueError):
        restored.predict_batch(np.full_like(x, np.nan))


def test_pretrained_online_state_matches_batch():
    import os

    checkpoint = os.environ.get("FLYVIS_TRACKING_MODEL")
    if not checkpoint:
        pytest.skip("set FLYVIS_TRACKING_MODEL for real pretrained model parity")
    from mjlab_microduck.rom.flyvis_tracking.vision import FlyvisVision

    vision = FlyvisVision(checkpoint)
    rng = np.random.default_rng(7)
    import torch

    images = rng.integers(0, 256, (25, 240, 320, 3), dtype=np.uint8)
    online = []
    vision.reset()
    for frame in images:
        online.append(vision.encode(frame)[0])
    vision.reset()
    with torch.inference_mode():
        states = vision.network.simulate(
            vision.retina(images).repeat_interleave(4, dim=1),
            0.01,
            initial_state=vision.initial_state,
            as_states=True,
        )
    batched = np.array([vision.pool(s.nodes.activity.numpy()) for s in states[3::4]])
    assert np.allclose(online, batched, atol=1e-5, rtol=1e-5)
    assert not any(p.requires_grad for p in vision.network.parameters())


def test_effective_network_configuration_and_topology_are_bound():
    from mjlab_microduck.rom.flyvis_tracking.vision import network_identity

    nodes = {
        "type": np.array(["R1", "T4"]),
        "u": np.array([0, 1]),
        "v": np.array([0, 0]),
    }
    edges = {
        "source_index": np.array([0]),
        "target_index": np.array([1]),
        "sign": np.array([1.0]),
        "n_syn": np.array([2.0]),
    }
    config = {"dynamics": {"activation": "relu"}}
    original = network_identity(config, nodes, edges)
    assert original != network_identity(
        {"dynamics": {"activation": "sigmoid"}}, nodes, edges
    )
    edges["target_index"] = np.array([0])
    assert original != network_identity(config, nodes, edges)


def test_fast_retina_matches_official_boxeye_on_asymmetric_pixels():
    from importlib.util import find_spec

    if find_spec("flyvis") is None:
        pytest.skip("requires optional flyvis dependency group")
    import torch

    from mjlab_microduck.rom.flyvis_tracking.vision import RetinaVision

    vision = RetinaVision()
    rng = np.random.default_rng(19)
    frames = rng.integers(0, 256, (2, 240, 320, 3), dtype=np.uint8)
    grey = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    expected = vision.eye(torch.from_numpy(grey[None])).numpy()
    assert np.allclose(vision.retina(frames).numpy(), expected, atol=2e-6, rtol=2e-6)


def test_full_episode_parity_on_recorded_camera_frames():
    import os

    import torch

    checkpoint = os.environ.get("FLYVIS_TRACKING_MODEL")
    replay = os.environ.get("FLYVIS_TRACKING_REPLAY")
    if not checkpoint or not replay:
        pytest.skip("requires official checkpoint and a recorded camera NPZ")
    from mjlab_microduck.rom.flyvis_tracking.vision import FlyvisVision

    vision = FlyvisVision(checkpoint)
    with np.load(replay, allow_pickle=False) as data:
        images = data["rgb"]
    vision.reset()
    reference_state = vision.initial_state
    for i in range(500):
        rgb = images[i % len(images)]
        actual, _ = vision.encode(rgb)
        with torch.inference_mode():
            states = vision.network.simulate(
                vision.retina(rgb[None]).repeat(1, 4, 1, 1),
                0.01,
                initial_state=reference_state,
                as_states=True,
            )
        reference_state = states[-1]
        expected = vision.pool(reference_state.nodes.activity.numpy())
        assert np.allclose(actual, expected, atol=1e-5, rtol=1e-5), (
            f"diverged at frame {i}"
        )
