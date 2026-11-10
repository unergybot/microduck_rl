"""CPU-only ONNX fixtures, independent of training/robot imports."""
from pathlib import Path
import onnx
from onnx import TensorProto, helper

def write_minimal_onnx(path: Path) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["observation"], ["action"])],
        "microduck-test-policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 14])],
    )
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
        ),
        path,
    )
    return path


def write_normalized_onnx(path: Path) -> Path:
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["observation", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "std"], ["normalized"]),
            helper.make_node("MatMul", ["normalized", "weights"], ["action"]),
        ],
        "microduck-normalized-test-policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 14])],
        [
            helper.make_tensor("mean", TensorProto.FLOAT, [61], [0.0] * 61),
            helper.make_tensor("std", TensorProto.FLOAT, [61], [1.0] * 61),
            helper.make_tensor(
                "weights", TensorProto.FLOAT, [61, 14], [0.0] * (61 * 14)
            ),
        ],
    )
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
        ),
        path,
    )
    return path


