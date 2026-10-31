"""Shared provenance validation for supported normalized ONNX exports."""
from pathlib import Path
from mjlab_microduck.rom.onnx_policy import inspect_normalized_actor

def attach_microduck_metadata(
    onnx_path: str | Path,
    *,
    task_id: str,
    source_commit: str,
    checkpoint: str | None,
    run_identity: str | None,
) -> None:
    """Append ROM provenance without replacing the exported, normalized actor graph."""
    import onnx

    model = onnx.load(str(onnx_path))
    try:
        normalized_graph = inspect_normalized_actor(model)
    except ValueError as exc:
        raise ValueError(
            "exported actor does not contain the expected empirical normalizer"
        ) from exc
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata.update(
        {
            "microduck.task_id": task_id,
            "microduck.source_commit": source_commit,
            "microduck.observation_contract": "MICRODUCK_OBS_61_V1",
            "microduck.action_contract": "MICRODUCK_ACTION_14_V1",
            "microduck.checkpoint": checkpoint or "",
            "microduck.run_identity": run_identity or "",
            "microduck.normalization": "EMPIRICAL_NORMALIZATION_V1",
            "microduck.normalization_graph_sha256": normalized_graph.graph_sha256,
            "microduck.normalized_graph_fingerprint": normalized_graph.fingerprint,
        }
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(model, str(onnx_path))
