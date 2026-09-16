"""Frozen, stateful flyvis vision and a matched retinal-input baseline."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from scipy.sparse import csr_matrix

from .core import SpatialPool


def network_identity(config, nodes, edges):
    hasher = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    for group, names in (
        (nodes, ("type", "u", "v")),
        (edges, ("source_index", "target_index", "sign", "n_syn")),
    ):
        for name in names:
            array = np.asarray(group[name][:])
            if array.dtype.kind in "OU":
                array = array.astype("S")
            hasher.update(json.dumps([name, array.dtype.str, array.shape]).encode())
            hasher.update(array.tobytes())
    return hasher.hexdigest()


def configure_flyvis():
    # flyvis modules copy their device at import time, so choose CPU before import.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault(
        "FLYVIS_ROOT_DIR", str(Path.home() / ".cache/microduck-flyvis")
    )
    import flyvis

    # Keep the first experiment reproducible on CPU; do not inherit global GPU defaults.
    if flyvis.device.type != "cpu":
        raise RuntimeError(
            "start a fresh process with CUDA_VISIBLE_DEVICES='' for this CPU experiment"
        )
    torch.set_default_device("cpu")
    torch.set_num_threads(2)
    return flyvis


class RetinaVision:
    kind = "retina"

    def __init__(self):
        configure_flyvis()
        from flyvis.datasets.rendering import BoxEye

        self.eye = BoxEye(extent=15, kernel_size=13)
        self.size = 721
        self.manifest = {
            "kind": self.kind,
            "extent": 15,
            "kernelSize": 13,
            "grayscaleWeights": [0.299, 0.587, 0.114],
            "retinaSites": 721,
        }

    def reset(self):
        pass

    def retina(self, frames):
        grey = (
            np.asarray(frames, np.float32)
            @ np.array([0.299, 0.587, 0.114], np.float32)
            / 255
        )
        with torch.inference_mode():
            from torchvision.transforms.functional import resize

            sequence = torch.from_numpy(grey[None])
            if (self.eye.min_frame_size > torch.tensor(sequence.shape[-2:])).any():
                sequence = resize(sequence, self.eye.min_frame_size.tolist())
            # Same zero-padded 13x13 box mean, implemented as separable sums.
            filtered = uniform_filter(
                sequence.numpy(), size=(1, 1, 13, 13), mode="constant"
            )
            return self.eye.hex_render(torch.from_numpy(filtered)).reshape(
                1, len(frames), 1, 721
            )

    def encode(self, rgb):
        retina = self.retina(rgb[None]).reshape(-1).cpu().numpy().copy()
        return retina, retina


class FlyvisVision(RetinaVision):
    kind = "flyvis"

    def __init__(self, checkpoint_dir):
        super().__init__()
        import flyvis

        checkpoint_dir = Path(checkpoint_dir).resolve()
        if not checkpoint_dir.is_dir():
            raise ValueError(
                "pretrained model directory missing; download official flyvis weights"
            )
        view = flyvis.NetworkView(checkpoint_dir)
        checkpoint = Path(view.get_checkpoint())
        self.network = view.init_network().cpu().eval()
        self.network.requires_grad_(False)
        if (
            self.network.dynamics.__class__.__name__ != "PPNeuronIGRSynapses"
            or not isinstance(self.network.dynamics.activation, torch.nn.ReLU)
        ):
            raise ValueError("frozen evaluator requires PPNeuronIGRSynapses with ReLU")
        self.frozen_params = self.network._param_api()
        self.bias = self.frozen_params.nodes.bias.detach().numpy().copy()
        self.tau = np.maximum(
            self.frozen_params.nodes.time_const.detach().numpy(), np.float32(0.01)
        )
        edges = self.network.connectome.edges
        self.weights = csr_matrix(
            (
                self.frozen_params.edges.weight.detach().numpy(),
                (edges.target_index[:], edges.source_index[:]),
            ),
            shape=(len(self.bias), len(self.bias)),
        )
        nodes = self.network.connectome.nodes
        types = nodes.type[:].astype(str)
        # Same oblique-to-image mapping as BoxEye: image x=v, image y=u+v/2.
        self.pool = SpatialPool(types, nodes.v[:], nodes.u[:] + nodes.v[:] / 2)
        self.size = self.pool.size
        self.manifest.update(
            {
                "kind": self.kind,
                "version": flyvis.__version__,
                "checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "connectomeSha256": hashlib.sha256(
                    Path(flyvis.connectome_file).read_bytes()
                ).hexdigest(),
                "neuralDtS": 0.01,
                "integration": "frozen PPNeuronIGRSynapses ReLU Euler; CSR target sums",
                "networkIdentitySha256": network_identity(
                    view.dir.config.network.to_dict(),
                    self.network.connectome.nodes,
                    self.network.connectome.edges,
                ),
                "cellTypes": self.pool.types,
                "pool": "3x3 means per cell type",
                "featureDimension": self.size,
            }
        )
        self.state = None
        self.initial_state = None

    def reset(self):
        with torch.inference_mode():
            if self.initial_state is None:
                self.initial_state = self.network.steady_state(1.0, 0.01, 1)
            self.state = self.initial_state
            self.activity = (
                self.state.nodes.activity.detach().numpy().reshape(-1).copy()
            )

    def encode(self, rgb):
        retina = self.retina(rgb[None])
        with torch.inference_mode():
            # Identical flyvis integration with immutable parameters cached once.
            # Official simulate() reconstructs every expanded synapse parameter
            # on each short chunk; online/batch parity tests guard this fast path.
            self.network.stimulus.zero(1, 4)
            self.network.stimulus.add_input(retina.repeat(1, 4, 1, 1))
            inputs = self.network.stimulus()
            for i in range(4):
                velocity = (
                    np.float32(1.0)
                    / self.tau
                    * (
                        -self.activity
                        + self.bias
                        + self.weights @ np.maximum(self.activity, np.float32(0.0))
                        + inputs[0, i].numpy()
                    )
                )
                self.activity = self.activity + velocity * np.float32(0.01)
        features = self.pool(self.activity)
        return features, retina.reshape(-1).cpu().numpy().copy()
