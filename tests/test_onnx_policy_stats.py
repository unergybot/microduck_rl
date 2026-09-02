import onnx
from onnx import TensorProto, helper
import numpy as np

from mjlab_microduck.rom.onnx_policy import inspect_normalized_actor


def test_normalizer_validator_accepts_exporter_broadcast_stats_shape(tmp_path):
    policy = tmp_path / "normalized-broadcast.onnx"
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])
    actions = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 61])
    mean = helper.make_tensor("mean", TensorProto.FLOAT, [1, 61], np.zeros((1, 61), dtype=np.float32).ravel())
    std = helper.make_tensor("std", TensorProto.FLOAT, [1, 61], np.ones((1, 61), dtype=np.float32).ravel())
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["obs", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "std"], ["actions"]),
        ],
        "broadcast-normalizer",
        [obs],
        [actions],
        [mean, std],
    )
    onnx.save(helper.make_model(graph), policy)

    inspected = inspect_normalized_actor(onnx.load(policy, load_external_data=False))

    assert inspected.fingerprint.startswith("sha256:")
