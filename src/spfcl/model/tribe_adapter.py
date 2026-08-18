from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TribeLiteSpec:
    hidden: int = 384
    depth: int = 4
    heads: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    low_rank_head: int = 128
    n_subjects: int = 4
    n_outputs: int = 360
    input_hz: int = 2
    output_hz: int = 1
    window_seconds: int = 100
    subject_dropout: float = 0.1
    subject_embedding: bool = False
    modality_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if self.hidden // self.heads < 32:
            raise ValueError("Official neuraltrain rotary attention requires dim/head >= 32")
        if self.subject_embedding:
            raise ValueError("Phase-1 primary config requires subject_embedding=false")
        if self.modality_dropout != 0:
            raise ValueError("Video-only Phase-1 requires modality_dropout=0")
        if self.input_hz * self.window_seconds > self.max_seq_len:
            raise ValueError("max_seq_len is shorter than the 100-second input window")
        if not 0 < self.subject_dropout < 1:
            raise ValueError("subject_dropout must be in (0, 1) to create/train the group row")

    @property
    def max_seq_len(self) -> int:
        return self.input_hz * self.window_seconds

    @property
    def output_timesteps(self) -> int:
        return self.output_hz * self.window_seconds

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> TribeLiteSpec:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown TRIBE-lite setting(s): {sorted(unknown)}")
        return cls(**value)


def build_tribe_lite(
    *,
    feature_dims: dict[str, tuple[int, int]],
    spec: TribeLiteSpec | None = None,
):
    """Build the official FmriEncoder class with a smaller, explicit config."""

    spec = spec or TribeLiteSpec()
    if set(feature_dims) != {"video"}:
        raise ValueError(f"Phase-1 is video-only; got modalities {sorted(feature_dims)}")
    try:
        from neuraltrain.models.common import Mlp, SubjectLayers
        from neuraltrain.models.transformer import TransformerEncoder
        from tribev2.model import FmriEncoder
    except ImportError as exc:  # pragma: no cover - remote environment
        raise RuntimeError(
            "TRIBE dependencies are not installed. Run scripts/bootstrap_remote.sh."
        ) from exc

    config = FmriEncoder(
        projector=Mlp(hidden_sizes=None, norm_layer=None, activation_layer=None),
        combiner=None,
        encoder=TransformerEncoder(
            heads=spec.heads,
            depth=spec.depth,
            ff_mult=spec.ff_mult,
            attn_dropout=spec.dropout,
            ff_dropout=spec.dropout,
            layer_dropout=0.0,
        ),
        time_pos_embedding=True,
        subject_embedding=spec.subject_embedding,
        subject_layers=SubjectLayers(
            n_subjects=spec.n_subjects,
            subject_dropout=spec.subject_dropout,
            average_subjects=False,
        ),
        hidden=spec.hidden,
        max_seq_len=spec.max_seq_len,
        dropout=spec.dropout,
        extractor_aggregation="cat",
        layer_aggregation="cat",
        modality_dropout=spec.modality_dropout,
        low_rank_head=spec.low_rank_head,
    )
    return config.build(
        feature_dims=feature_dims,
        n_outputs=spec.n_outputs,
        n_output_timesteps=spec.output_timesteps,
    )


@contextlib.contextmanager
def head_mode(model, mode: Literal["subject", "group"]) -> Iterator[None]:
    """Temporarily select known-subject or official special group output row."""

    if not hasattr(model, "predictor") or not hasattr(model.predictor, "average_subjects"):
        raise TypeError("Model does not expose TRIBE SubjectLayers predictor")
    previous = bool(model.predictor.average_subjects)
    model.predictor.average_subjects = mode == "group"
    try:
        yield
    finally:
        model.predictor.average_subjects = previous


def forward_with_features(model, batch, *, pool_outputs: bool = True):
    """Official forward path with named shared/head activations for nuisance probes."""

    taps: dict[str, Any] = {}
    x = model.aggregate_features(batch)
    taps["projected"] = x
    subject_id = batch.data.get("subject_id")
    if hasattr(model, "temporal_smoothing"):
        x = model.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
        taps["smoothed"] = x
    if not model.config.linear_baseline:
        x = model.combiner(x)
        taps["pre_position"] = x
        if hasattr(model, "time_pos_embed"):
            x = x + model.time_pos_embed[:, : x.size(1)]
        taps["post_position"] = x
        if hasattr(model, "subject_embed"):
            raise RuntimeError("subject_embedding must be disabled in the primary Phase-1 model")
        x = model.encoder(x)
        taps["transformer"] = x
    channels_first = x.transpose(1, 2)
    if model.config.low_rank_head is not None:
        channels_first = model.low_rank_head(channels_first.transpose(1, 2)).transpose(1, 2)
        taps["shared_bottleneck"] = channels_first.transpose(1, 2)
    output = model.predictor(channels_first, subject_id)
    taps["head_output_2hz"] = output.transpose(1, 2)
    if pool_outputs:
        output = model.pooler(output)
    taps["output"] = output.transpose(1, 2)
    return output, taps


def state_sha256(module) -> str:
    """Stable SHA-256 over tensor names, dtypes, shapes, and CPU bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_weights_only(model, checkpoint: str | Path, *, strict: bool = True) -> tuple[list[str], list[str]]:
    """Load only model tensors from a Lightning or plain checkpoint.

    Optimizer, scheduler, epoch, and global-step state are deliberately ignored.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Checkpoint loading requires PyTorch") from exc
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {checkpoint}")
    model_state = {}
    for key, value in state.items():
        if key.startswith("model."):
            model_state[key[len("model.") :]] = value
        elif key in model.state_dict():
            model_state[key] = value
    result = model.load_state_dict(model_state, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


def synthetic_smoke(
    *,
    spec: TribeLiteSpec,
    feature_layers: int = 2,
    feature_dim: int = 16,
    batch_size: int = 2,
    seed: int = 7,
) -> dict[str, Any]:
    """CPU-safe synthetic model contract check; downloads no model or dataset."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("model-smoke requires the remote training environment") from exc
    from types import SimpleNamespace

    torch.manual_seed(seed)
    model = build_tribe_lite(feature_dims={"video": (feature_layers, feature_dim)}, spec=spec)
    model.eval()
    batch = SimpleNamespace(
        data={
            "video": torch.randn(
                batch_size, feature_layers, feature_dim, spec.max_seq_len
            ),
            "subject_id": torch.tensor([index % spec.n_subjects for index in range(batch_size)]),
        }
    )
    with torch.no_grad(), head_mode(model, "subject"):
        known = model(batch)
    with torch.no_grad(), head_mode(model, "group"):
        group = model(batch)
        alternate = SimpleNamespace(data=dict(batch.data))
        alternate.data["subject_id"] = (batch.data["subject_id"] + 1) % spec.n_subjects
        group_alternate = model(alternate)
    expected = (batch_size, spec.n_outputs, spec.output_timesteps)
    if tuple(known.shape) != expected or tuple(group.shape) != expected:
        raise RuntimeError(
            f"Unexpected output shape: known={tuple(known.shape)}, group={tuple(group.shape)}"
        )
    if not torch.equal(group, group_alternate):
        raise RuntimeError("Group head output changed with subject ID")
    return {
        "known_shape": list(known.shape),
        "group_shape": list(group.shape),
        "group_subject_invariant": True,
        "predictor_rows": int(model.predictor.weights.shape[0]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_sha256": state_sha256(model),
    }

