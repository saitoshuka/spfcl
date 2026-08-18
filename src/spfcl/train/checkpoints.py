from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMAT_NAME = "spfcl.weights"
FORMAT_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_METADATA = {
    "condition",
    "format",
    "format_version",
    "parent_checkpoint",
    "parent_state_sha256",
    "saved_at_utc",
    "seed",
    "stage",
    "state_sha256",
}


@dataclass(frozen=True)
class WeightsCheckpoint:
    """Provenance for a checkpoint that deliberately excludes trainer state."""

    path: Path
    state_sha256: str
    metadata: dict[str, Any]
    format_name: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION


@dataclass(frozen=True)
class WeightsLoadResult:
    checkpoint: WeightsCheckpoint
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on the GPU server
        raise RuntimeError(
            "Weights checkpoints require PyTorch; run scripts/bootstrap_remote.sh first."
        ) from exc
    return torch


def _tensor_bytes(tensor: Any) -> bytes:
    """Return dtype-agnostic CPU bytes, including for bfloat16 tensors."""

    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"State dictionaries may contain only tensors, got {type(tensor)!r}")
    value = tensor.detach().cpu().contiguous()
    if value.is_quantized:
        value = value.int_repr().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in a stable key order."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(name, str):
            raise TypeError(f"State-dict keys must be strings, got {type(name)!r}")
        digest.update(name.encode("utf8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    torch = _torch()
    state: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Model state {name!r} is not a tensor")
        state[name] = value.detach().cpu().contiguous().clone()
    if not state:
        raise ValueError("Refusing to save an empty model state dictionary")
    return state


def _validate_sha256(value: str | None, *, field: str) -> None:
    if value is not None and not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")


def _validate_json_value(value: Any, *, field: str) -> None:
    """Keep metadata safe for ``torch.load(weights_only=True)`` and portable."""

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Checkpoint metadata {field!r} must be JSON-compatible") from exc


def _load_payload(path: Path) -> Any:
    torch = _torch()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Could not safely read checkpoint {path}. Convert legacy checkpoints to a "
            "tensor-only state_dict before using them as a continual-learning boundary."
        ) from exc


def _payload_parts(
    payload: Any, *, path: Path
) -> tuple[Mapping[str, Any], dict[str, Any], str, int]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping: {path}")

    if payload.get("format") == FORMAT_NAME:
        version = payload.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported {FORMAT_NAME} version {version!r}; expected {FORMAT_VERSION}"
            )
        state = payload.get("state_dict")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Checkpoint metadata must be a mapping: {path}")
        metadata = dict(metadata)
        format_name = FORMAT_NAME
        format_version = FORMAT_VERSION
    elif "state_dict" in payload:
        state = payload["state_dict"]
        metadata = {}
        format_name = "legacy.state_dict"
        format_version = 0
    else:
        state = payload
        metadata = {}
        format_name = "plain.state_dict"
        format_version = 0

    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"Checkpoint does not contain a non-empty state dictionary: {path}")
    torch = _torch()
    non_tensors = [name for name, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        sample = ", ".join(map(str, non_tensors[:3]))
        raise TypeError(f"Checkpoint state has non-tensor entries ({sample}): {path}")
    return state, metadata, format_name, format_version


def _normalise_model_keys(state: Mapping[str, Any], expected_keys: set[str]) -> dict[str, Any]:
    """Strip common wrapper prefixes while refusing ambiguous collisions.

    A Lightning ``BrainModule`` checkpoint may also contain metric or loss
    buffers.  When the official ``model.`` namespace is present, only that
    namespace (plus already-exact model keys) is a model-weights boundary.
    """

    output: dict[str, Any] = {}
    prefixes = ("model.", "module.", "_orig_mod.")
    has_lightning_model_namespace = any(
        str(name).startswith("model.") for name in state
    )
    for original, value in state.items():
        name = str(original)
        if (
            has_lightning_model_namespace
            and not name.startswith("model.")
            and name not in expected_keys
        ):
            continue
        if name not in expected_keys:
            for prefix in prefixes:
                if name.startswith(prefix) and name[len(prefix) :] in expected_keys:
                    name = name[len(prefix) :]
                    break
        if name in output:
            raise ValueError(f"Checkpoint keys collide after prefix removal: {name}")
        output[name] = value
    return output


def read_weights_metadata(path: str | Path) -> WeightsCheckpoint:
    """Read provenance and verify the tensor hash without constructing a model."""

    checkpoint_path = Path(path).expanduser().resolve()
    state, metadata, format_name, format_version = _payload_parts(
        _load_payload(checkpoint_path), path=checkpoint_path
    )
    actual_sha256 = state_dict_sha256(state)
    recorded_sha256 = metadata.get("state_sha256")
    _validate_sha256(recorded_sha256, field="metadata.state_sha256")
    if recorded_sha256 is not None and recorded_sha256 != actual_sha256:
        raise ValueError(
            f"Checkpoint tensor hash mismatch for {checkpoint_path}: "
            f"recorded={recorded_sha256}, actual={actual_sha256}"
        )
    return WeightsCheckpoint(
        path=checkpoint_path,
        state_sha256=actual_sha256,
        metadata=metadata,
        format_name=format_name,
        format_version=format_version,
    )


def save_weights_checkpoint(
    model: Any,
    path: str | Path,
    *,
    stage: str,
    condition: str,
    seed: int,
    parent_checkpoint: str | Path | None = None,
    parent_state_sha256: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> WeightsCheckpoint:
    """Atomically save model tensors and provenance, never optimizer/trainer state."""

    if not stage.strip() or not condition.strip():
        raise ValueError("stage and condition must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    _validate_sha256(parent_state_sha256, field="parent_state_sha256")

    parent_path: Path | None = None
    if parent_checkpoint is not None:
        parent = read_weights_metadata(parent_checkpoint)
        parent_path = parent.path
        if parent_state_sha256 is None:
            parent_state_sha256 = parent.state_sha256
        elif parent_state_sha256 != parent.state_sha256:
            raise ValueError(
                "parent_state_sha256 does not match parent_checkpoint: "
                f"recorded={parent_state_sha256}, actual={parent.state_sha256}"
            )

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = _cpu_state_dict(model)
    digest = state_dict_sha256(state)

    if extra_metadata:
        collisions = _RESERVED_METADATA.intersection(extra_metadata)
        if collisions:
            raise ValueError(f"extra_metadata uses reserved keys: {sorted(collisions)}")
        for name, value in extra_metadata.items():
            if not isinstance(name, str):
                raise TypeError("extra_metadata keys must be strings")
            _validate_json_value(value, field=name)

    metadata: dict[str, Any] = {
        "condition": condition,
        "stage": stage,
        "seed": seed,
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "state_sha256": digest,
        "parent_checkpoint": str(parent_path) if parent_path is not None else None,
        "parent_state_sha256": parent_state_sha256,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "state_dict": state,
        "metadata": metadata,
    }

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    torch = _torch()
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    # Some otherwise-safe NFS/Lustre mounts reject directory fsync.
                    pass
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()

    return WeightsCheckpoint(
        path=destination,
        state_sha256=digest,
        metadata=metadata,
    )


def load_weights_checkpoint(
    model: Any,
    path: str | Path,
    *,
    strict: bool = True,
    expected_sha256: str | None = None,
) -> WeightsLoadResult:
    """Load model tensors only, ignoring any legacy optimizer or scheduler payload."""

    _validate_sha256(expected_sha256, field="expected_sha256")
    checkpoint_path = Path(path).expanduser().resolve()
    state, metadata, format_name, format_version = _payload_parts(
        _load_payload(checkpoint_path), path=checkpoint_path
    )
    normalised = _normalise_model_keys(state, set(model.state_dict()))
    actual_sha256 = state_dict_sha256(normalised)
    recorded_sha256 = metadata.get("state_sha256")
    _validate_sha256(recorded_sha256, field="metadata.state_sha256")
    required_sha256 = expected_sha256 or recorded_sha256
    if required_sha256 is not None and actual_sha256 != required_sha256:
        raise ValueError(
            f"Checkpoint tensor hash mismatch for {checkpoint_path}: "
            f"expected={required_sha256}, actual={actual_sha256}"
        )

    incompatible = model.load_state_dict(normalised, strict=strict)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if not missing and not unexpected:
        loaded_sha256 = state_dict_sha256(model.state_dict())
        if loaded_sha256 != actual_sha256:
            raise RuntimeError(
                "Model state changed while loading the checkpoint; refusing an unauditable fork"
            )

    descriptor = WeightsCheckpoint(
        path=checkpoint_path,
        state_sha256=actual_sha256,
        metadata=metadata,
        format_name=format_name,
        format_version=format_version,
    )
    return WeightsLoadResult(
        checkpoint=descriptor,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
