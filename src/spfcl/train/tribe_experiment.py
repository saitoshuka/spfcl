from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .checkpoints import (
    WeightsCheckpoint,
    load_weights_checkpoint,
    save_weights_checkpoint,
    state_dict_sha256,
)

_TRIBE_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - unavailable in the lightweight local test environment
    import pydantic
    from tribev2.main import Data, TribeExperiment
except ImportError as exc:  # pragma: no cover
    _TRIBE_IMPORT_ERROR = exc
    pydantic = None
    TribeExperiment = None


def _session_by_timeline(events: Any) -> dict[Any, Any]:
    """Map timelines from real acquisition events, excluding neuralset dummies."""

    if "session" not in events or "timeline" not in events:
        return {}
    valid_session = events["session"].notna() & (
        events["session"].astype(str).str.strip() != ""
    )
    return (
        events.loc[valid_session]
        .dropna(subset=["timeline"])
        .groupby("timeline", sort=False)["session"]
        .first()
        .to_dict()
    )


if TribeExperiment is not None:

    class Phase1Data(Data):  # type: ignore[misc]
        """Official data path plus deterministic old-study replay sampling."""

        replay_fraction: float = 0.0
        replay_session: int | None = None
        replay_seed: int = 0

        def model_post_init(self, context: Any) -> None:
            if not 0 <= self.replay_fraction < 1:
                raise ValueError("replay_fraction must lie in [0, 1)")
            if self.replay_fraction and self.replay_session is None:
                raise ValueError("replay_session is required when replay_fraction > 0")
            super().model_post_init(context)

        def get_loaders(self, *args: Any, **kwargs: Any):
            if not self.replay_fraction:
                return super().get_loaders(*args, **kwargs)

            # Upstream exposes no sampler hook. Intercept its pure segment
            # enumeration in this single-threaded setup step, keep every new-study
            # segment, and select a stable fraction of old-study segments.
            import hashlib

            import neuralset as ns

            original = ns.segments.list_segments

            def list_segments(events: Any, *values: Any, **settings: Any):
                segments = list(original(events, *values, **settings))
                splits = set(events.split.dropna().astype(str)) if "split" in events else set()
                if splits != {"train"} or "session" not in events:
                    return segments
                # neuralset appends a dummy CategoricalEvent whose standardized
                # session is the empty string. It can become groupby.first() and
                # silently classify every old-study window as new, defeating the
                # 1% replay cap. Derive the timeline mapping only from events with
                # a real acquisition session.
                session_by_timeline = _session_by_timeline(events)
                old, new = [], []
                for segment in segments:
                    timeline = segment.ns_events[0].timeline if segment.ns_events else ""
                    session = session_by_timeline.get(timeline)
                    target = (
                        old
                        if session is not None and int(session) == self.replay_session
                        else new
                    )
                    target.append(segment)
                count = max(1, round(self.replay_fraction * len(old))) if old else 0

                def stable_key(segment: Any) -> str:
                    timeline = segment.ns_events[0].timeline if segment.ns_events else ""
                    value = f"{self.replay_seed}:{timeline}:{segment.start}:{segment.duration}"
                    return hashlib.sha256(value.encode("utf8")).hexdigest()

                return new + sorted(old, key=stable_key)[:count]

            ns.segments.list_segments = list_segments
            try:
                return super().get_loaders(*args, **kwargs)
            finally:
                ns.segments.list_segments = original

    class WeightsOnlyTribeExperiment(TribeExperiment):  # type: ignore[misc]
        """Official experiment with an auditable continual-stage boundary."""

        checkpoint_path: str | None = None
        load_checkpoint: bool = False
        stage_parent_checkpoint: str | None = None
        stage_parent_sha256: str | None = None
        data: Phase1Data
        precision: str | int = "32-true"
        gradient_clip_val: float | None = None

        _stage_load_result: Any = pydantic.PrivateAttr(default=None)
        _initial_state_sha256: str | None = pydantic.PrivateAttr(default=None)

        def model_post_init(self, context: Any) -> None:
            if self.checkpoint_path is not None or self.load_checkpoint:
                raise ValueError(
                    "WeightsOnlyTribeExperiment forbids checkpoint_path/load_checkpoint. "
                    "Use stage_parent_checkpoint; Lightning resume would restore trainer state."
                )
            if self.stage_parent_sha256 is not None and self.stage_parent_checkpoint is None:
                raise ValueError(
                    "stage_parent_sha256 requires stage_parent_checkpoint"
                )
            super().model_post_init(context)

        def _get_checkpoint_path(self) -> None:
            # Both BrainModule.load_from_checkpoint and Trainer.fit(ckpt_path=...)
            # call this method upstream. Returning None blocks both full resumes.
            return None

        def _init_module(self, model: Any) -> Any:
            module = super()._init_module(model)
            if self.stage_parent_checkpoint is not None:
                self._stage_load_result = load_weights_checkpoint(
                    module.model,
                    self.stage_parent_checkpoint,
                    strict=True,
                    expected_sha256=self.stage_parent_sha256,
                )
            self._initial_state_sha256 = state_dict_sha256(module.model.state_dict())
            return module

        @property
        def initial_state_sha256(self) -> str | None:
            return self._initial_state_sha256

        def _setup_trainer(self, *args: Any, **kwargs: Any) -> Any:
            # The release schema omits mixed precision and clipping. Preserve its
            # setup code and inject only those Lightning constructor arguments.
            import lightning.pytorch as pl
            import numpy as np
            import torch

            # Feature/fMRI preparation can consume process RNG before the official
            # model is instantiated. Re-seeding at this exact boundary makes cold-
            # and warm-cache sibling branches start from identical tensors.
            if self.seed is not None:
                pl.seed_everything(self.seed, workers=True)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)

            trainer_class = pl.Trainer

            def trainer_factory(*values: Any, **settings: Any):
                settings.setdefault("precision", self.precision)
                if self.gradient_clip_val is not None:
                    settings.setdefault("gradient_clip_val", self.gradient_clip_val)
                return trainer_class(*values, **settings)

            pl.Trainer = trainer_factory  # type: ignore[assignment]
            try:
                return super()._setup_trainer(*args, **kwargs)
            finally:
                pl.Trainer = trainer_class  # type: ignore[assignment]

        def run_fixed_epochs(self) -> None:
            """Train with the official stack and no validation-based selection.

            Upstream ``run`` always requests train+val loaders and later restores
            the validation-selected ``best.ckpt``.  BOLD Moments randomizes train
            stimuli across run numbers for each subject, so a run-based internal
            validation split leaks stimuli.  This Phase-1 entry point instead
            retains the official train/test boundary, fits the preregistered epoch
            count, and leaves final-epoch tensors resident for post-stage tests.
            """

            import lightning.pytorch as pl
            import numpy as np
            import torch
            from lightning.pytorch.loggers import CSVLogger

            if self.save_checkpoints:
                raise ValueError("run_fixed_epochs requires save_checkpoints=False")
            if self.patience is not None:
                raise ValueError("run_fixed_epochs forbids validation-based early stopping")
            if self.n_epochs is None or self.n_epochs <= 0:
                raise ValueError("run_fixed_epochs requires a positive fixed n_epochs")

            self.setup_run()
            if self.wandb_config is not None:
                raise ValueError("Phase-1 remote runner expects wandb_config=None")
            # Upstream always installs LearningRateMonitor, which Lightning
            # rejects when no logger is attached.  CSVLogger is local, offline,
            # and keeps the learning-rate trace auditable without credentials.
            self._logger = CSVLogger(
                save_dir=str(Path(self.infra.folder) / "logs"),
                name="lightning",
                version="fixed-epochs",
            )
            if self.seed is not None:
                pl.seed_everything(self.seed, workers=True)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)

            events = self.data.get_events()
            loaders = self.data.get_loaders(events=events, split_to_build="train")
            if "train" not in loaders:
                raise RuntimeError("The official BOLD Moments training split is empty")
            train_loader = loaders["train"]
            self._setup_trainer(train_loader)
            self._trainer.fit(
                model=self._model,
                train_dataloaders=train_loader,
                val_dataloaders=None,
                ckpt_path=None,
            )

        @property
        def stage_load_audit(self) -> dict[str, Any] | None:
            if self._stage_load_result is None:
                return None
            checkpoint = self._stage_load_result.checkpoint
            return {
                "parent_checkpoint": str(checkpoint.path),
                "parent_state_sha256": checkpoint.state_sha256,
                "missing_keys": list(self._stage_load_result.missing_keys),
                "unexpected_keys": list(self._stage_load_result.unexpected_keys),
                "lightning_resume": False,
            }

else:
    WeightsOnlyTribeExperiment = None


def weights_only_experiment_class():
    """Return the importable official subclass, or explain remote setup."""

    if WeightsOnlyTribeExperiment is None:
        raise RuntimeError(
            "Official TRIBE training dependencies are unavailable; "
            "run scripts/bootstrap_remote.sh."
        ) from _TRIBE_IMPORT_ERROR
    return WeightsOnlyTribeExperiment


def build_weights_only_experiment(
    config: Mapping[str, Any],
    *,
    parent_checkpoint: str | Path | None = None,
    parent_state_sha256: str | None = None,
):
    """Build an official experiment with a fresh trainer for this stage."""

    if config.get("checkpoint_path") is not None or config.get("load_checkpoint") is True:
        raise ValueError(
            "Remove official checkpoint_path/load_checkpoint from stage config; "
            "continual boundaries use parent_checkpoint weights only."
        )
    if parent_state_sha256 is not None and parent_checkpoint is None:
        raise ValueError("parent_state_sha256 requires parent_checkpoint")
    parent = None
    if parent_checkpoint is not None:
        parent = Path(parent_checkpoint).expanduser().resolve()
        if not parent.is_file():
            raise FileNotFoundError(f"Parent stage checkpoint does not exist: {parent}")

    values = dict(config)
    values.update(
        {
            "checkpoint_path": None,
            "load_checkpoint": False,
            "stage_parent_checkpoint": str(parent) if parent is not None else None,
            "stage_parent_sha256": parent_state_sha256,
        }
    )
    return weights_only_experiment_class()(**values)


def export_experiment_weights(
    experiment: Any,
    path: str | Path,
    *,
    stage: str,
    condition: str,
    seed: int,
    source_checkpoint: str | Path | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> WeightsCheckpoint:
    """Export the trained TRIBE model as the only legal next-stage input.

    ``source_checkpoint`` can select an upstream Lightning ``best.ckpt``. Its
    model tensors are loaded, but its optimizer and trainer payload is ignored.
    """

    module = getattr(experiment, "_model", None)
    model = getattr(module, "model", None)
    if model is None:
        raise RuntimeError("Experiment has no built/trained BrainModule to export")
    source_sha256 = None
    if source_checkpoint is not None:
        source = load_weights_checkpoint(model, source_checkpoint, strict=True)
        source_sha256 = source.checkpoint.state_sha256

    metadata = {
        "upstream_commit": "af58661791a351a448a489042a28f6c37e1c14b7",
        "boundary_semantics": "model_weights_only",
        "source_lightning_checkpoint": (
            str(Path(source_checkpoint).expanduser().resolve())
            if source_checkpoint is not None
            else None
        ),
        "source_state_sha256": source_sha256,
    }
    if extra_metadata:
        overlap = set(metadata).intersection(extra_metadata)
        if overlap:
            raise ValueError(f"extra_metadata uses reserved export keys: {sorted(overlap)}")
        metadata.update(extra_metadata)

    parent = getattr(experiment, "stage_parent_checkpoint", None)
    parent_sha256 = getattr(experiment, "stage_parent_sha256", None)
    if parent is not None and parent_sha256 is None:
        audit = getattr(experiment, "stage_load_audit", None)
        if audit:
            parent_sha256 = audit["parent_state_sha256"]
    return save_weights_checkpoint(
        model,
        path,
        stage=stage,
        condition=condition,
        seed=seed,
        parent_checkpoint=parent,
        parent_state_sha256=parent_sha256,
        extra_metadata=metadata,
    )
