from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from spfcl.config import load_yaml

from .checkpoints import WeightsCheckpoint, read_weights_metadata
from .tribe_experiment import build_weights_only_experiment, export_experiment_weights

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = "af58661791a351a448a489042a28f6c37e1c14b7"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf8")
    return hashlib.sha256(serialized).hexdigest()


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    roots = (REPO_ROOT / "src" / "spfcl", REPO_ROOT / "vendor" / "tribev2" / "tribev2")
    files = sorted(path for root in roots for path in root.rglob("*.py"))
    for path in files:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _experiment_fingerprint(
    experiment: Mapping[str, Any], config_file: Path
) -> dict[str, Any]:
    """Hash every code/config/asset identity that can change stage semantics."""

    data_root = Path(experiment["data"]["root"])
    cache_root = Path(experiment["experiment"]["cache_root"])
    model_config = _resolve_project_path(experiment["model"]["config"])
    nuisance_config = _resolve_project_path(
        experiment["data"]["studies"]["B_lambda"]["nuisance_config"]
    )
    atlas = experiment.get("atlas", {})
    candidates = {
        "training_yaml": config_file,
        "model_yaml": model_config,
        "nuisance_yaml": nuisance_config,
        "dependency_constraints": REPO_ROOT
        / "configs"
        / "training"
        / "constraints_remote.txt",
        "frozen_run_manifest": Path(experiment["experiment"]["manifest"]),
        "atlas_operator": Path(
            atlas.get("operator", cache_root / "atlas" / "glasser360_fsaverage5.npz")
        ),
        "atlas_basis": Path(
            atlas.get("basis", cache_root / "atlas" / "glasser360_laplacian_basis.npz")
        ),
        "vjepa_provenance": Path(experiment["features"]["video_model_provenance"]),
        "openneuro_provenance": data_root / "download" / "spfcl_download.json",
        "openneuro_inventory": data_root
        / "download"
        / "spfcl_snapshot_inventory.json",
    }
    missing = [str(path) for path in candidates.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Reproducibility inputs are missing: " + ", ".join(missing)
        )
    components = {name: _file_sha256(path) for name, path in candidates.items()}
    components["resolved_experiment"] = _canonical_sha256(experiment)
    components["source_tree"] = _source_tree_sha256()
    components["upstream_commit"] = UPSTREAM_COMMIT
    return {
        "sha256": _canonical_sha256(components),
        "components": components,
    }


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _stage_plan(experiment: Mapping[str, Any], condition: str) -> list[dict[str, Any]]:
    """Load and strictly validate the executable stage DAG from the YAML."""

    condition_spec = experiment["matrix"]["conditions"].get(condition)
    if not isinstance(condition_spec, dict) or not isinstance(
        condition_spec.get("stages"), list
    ):
        raise TypeError(f"Condition {condition!r} must define a stages list")
    stages = copy.deepcopy(condition_spec["stages"])
    allowed = {
        "id",
        "view",
        "train_sessions",
        "parent",
        "pair",
        "replay_fraction",
        "replay_session",
    }
    seen: set[str] = set()
    pair_groups: dict[str, list[dict[str, Any]]] = {}
    a_session = int(experiment["data"]["studies"]["A"]["session"])
    b_session = int(experiment["data"]["studies"]["B0"]["session"])
    for stage in stages:
        if not isinstance(stage, dict):
            raise TypeError(f"Condition {condition!r} contains a non-mapping stage")
        unknown = set(stage).difference(allowed)
        if unknown:
            raise ValueError(f"Stage {stage.get('id')!r} has unknown keys: {sorted(unknown)}")
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not stage_id or stage_id in seen:
            raise ValueError(f"Stage IDs must be unique non-empty strings: {stage_id!r}")
        if stage.get("view") not in {"A", "B0", "B_lambda"}:
            raise ValueError(f"Stage {stage_id!r} has an unknown data view")
        sessions = stage.get("train_sessions")
        if not isinstance(sessions, list) or not sessions or any(
            not isinstance(value, int) for value in sessions
        ):
            raise ValueError(f"Stage {stage_id!r} requires integer train_sessions")
        expected_sessions = (
            [a_session]
            if stage["view"] == "A"
            else [b_session]
        )
        if sessions != expected_sessions and not (
            stage["view"] in {"B0", "B_lambda"}
            and sessions == [a_session, b_session]
        ):
            raise ValueError(
                f"Stage {stage_id!r} sessions {sessions} conflict with its {stage['view']} view"
            )
        parent = stage.get("parent")
        if parent is not None and parent not in seen:
            raise ValueError(f"Stage {stage_id!r} references unavailable parent {parent!r}")
        replay = float(stage.get("replay_fraction", 0.0))
        if not 0 <= replay < 1:
            raise ValueError(f"Stage {stage_id!r} has invalid replay_fraction={replay}")
        if replay and stage.get("replay_session") not in sessions:
            raise ValueError(f"Stage {stage_id!r} replay_session must be trained")
        if replay and stage.get("replay_session") != a_session:
            raise ValueError(f"Stage {stage_id!r} must replay the clean A session")
        pair = stage.get("pair")
        if stage["view"] in {"B0", "B_lambda"} and not isinstance(pair, str):
            raise ValueError(f"Paired B stage {stage_id!r} requires a pair tag")
        if pair is not None:
            pair_groups.setdefault(pair, []).append(stage)
        seen.add(stage_id)
    if not stages:
        raise ValueError(f"Condition {condition!r} has no stages")
    for pair, paired in pair_groups.items():
        if len(paired) != 2 or {stage["view"] for stage in paired} != {"B0", "B_lambda"}:
            raise ValueError(f"Pair {pair!r} must contain exactly one B0 and one B_lambda stage")
        comparable = ("train_sessions", "parent", "replay_fraction", "replay_session")
        if any(paired[0].get(key) != paired[1].get(key) for key in comparable):
            raise ValueError(f"Pair {pair!r} differs in acquisition or parent semantics")
    return stages


def _study_query(
    *,
    subjects: list[int],
    train_sessions: list[int],
    test_sessions: list[int],
) -> str:
    subject_query = " or ".join(
        f"subject == 'Lahner2024Bold/{subject}'" for subject in subjects
    )
    train_query = " or ".join(f"session == {session}" for session in train_sessions)
    test_query = " or ".join(f"session == {session}" for session in test_sessions)
    return (
        f"({subject_query}) and "
        f"((({train_query}) and split == 'train') or "
        f"(({test_query}) and split == 'test'))"
    )


def _drop_conf_syntax(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_conf_syntax(item)
            for key, item in value.items()
            if not str(key).startswith("=")
        }
    if isinstance(value, list):
        return [_drop_conf_syntax(item) for item in value]
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a JSON completion record without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _savez_atomic(path: Path, **arrays: Any) -> None:
    """Publish a compressed NumPy bundle atomically on local or shared storage."""

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _inspect_prediction_archive(path: Path, required: set[str]) -> dict[str, Any]:
    """Validate a completed stage artifact before it is reused."""

    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as bundle:
            missing = required.difference(bundle.files)
            if missing:
                raise RuntimeError(
                    f"Prediction archive {path} is missing arrays: {sorted(missing)}"
                )
            if "encoding_empirical" in bundle.files:
                empirical = bundle["encoding_empirical"]
                predicted = bundle["encoding_predicted"]
                group = bundle["group_predicted"]
                if empirical.ndim != 3 or not (
                    empirical.shape == predicted.shape == group.shape
                ):
                    raise RuntimeError(
                        "Encoding prediction arrays must share shape (segments, parcels, time): "
                        f"{path}"
                    )
                segments = int(empirical.shape[0])
                if bundle["subject_id"].reshape(-1).shape[0] != segments:
                    raise RuntimeError(
                        f"Encoding subject_id length differs from segment count: {path}"
                    )
                if bundle["segment_uid"].reshape(-1).shape[0] != segments:
                    raise RuntimeError(
                        f"Encoding segment_uid length differs from segment count: {path}"
                    )
            else:
                empirical = bundle["localizer_empirical"]
                predicted = bundle["localizer_predicted"]
                group = bundle["localizer_group_predicted"]
                if empirical.ndim != 3 or not (
                    empirical.shape == predicted.shape == group.shape
                ):
                    raise RuntimeError(
                        "Localizer prediction arrays must share shape "
                        f"(subjects, conditions, parcels): {path}"
                    )
                segments = int(empirical.shape[0])
                if bundle["subject_id"].reshape(-1).shape[0] != segments:
                    raise RuntimeError(
                        f"Localizer subject_id length differs from subject count: {path}"
                    )
                if bundle["conditions"].reshape(-1).shape[0] != empirical.shape[1]:
                    raise RuntimeError(
                        f"Localizer condition labels differ from condition axis: {path}"
                    )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Prediction archive is incomplete or corrupt: {path}") from exc
    return {"path": str(path), "segments": segments}


def _local_infra(folder: Path, *, cpus: int, gpu: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "cluster": None,
        "folder": str(folder),
        "keep_in_ram": False,
        "cpus_per_task": cpus,
        "max_jobs": 1,
        "mode": "cached",
        "timeout_min": 60 * 24,
    }
    if gpu:
        value["gpus_per_node"] = 1
    return value


def _official_stage_config(
    experiment: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    condition: str,
    seed: int,
    stage_dir: Path,
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    # Importing defaults has filesystem side effects, so the runner establishes
    # the two required roots first and imports only in actual execution mode.
    from tribev2.grids.defaults import default_config

    from spfcl.data import localizer as _localizer_registration
    from spfcl.data import nuisance as _nuisance_registration
    from spfcl.data import parcellate as _parcellate_registration
    from spfcl.data import splits as _split_registration
    from spfcl.model.hub import validate_vjepa_snapshot

    del (
        _localizer_registration,
        _nuisance_registration,
        _parcellate_registration,
        _split_registration,
    )
    config = _drop_conf_syntax(copy.deepcopy(default_config))

    data_settings = experiment["data"]
    feature_settings = experiment["features"]
    training = experiment["training"]
    subjects = [int(value) for value in data_settings["subjects"]]
    test_sessions = [int(value) for value in data_settings["evaluation"].get("test_sessions", [2, 3])]
    train_sessions = [int(value) for value in stage["train_sessions"]]
    cache_root = Path(experiment["experiment"]["cache_root"])
    study_root = Path(data_settings["root"])
    study_parent = study_root.parent
    cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "8"))

    atlas = experiment.get("atlas", {})
    operator_path = Path(
        atlas.get("operator", cache_root / "atlas" / "glasser360_fsaverage5.npz")
    )
    basis_path = Path(
        atlas.get("basis", cache_root / "atlas" / "glasser360_laplacian_basis.npz")
    )
    nuisance_path = _resolve_project_path(
        data_settings["studies"]["B_lambda"]["nuisance_config"]
    )
    nuisance_values = load_yaml(nuisance_path).get("nuisance", {})
    expected_offset = 5.0
    if float(nuisance_values.get("sample_hz", data_settings["output_hz"])) != float(
        data_settings["output_hz"]
    ):
        raise ValueError("Nuisance sample_hz must equal the fMRI output frequency")
    if int(nuisance_values.get("chunk_seconds", 100)) != int(
        data_settings["window_seconds"]
    ):
        raise ValueError("Nuisance chunks must equal the model window duration")
    nuisance_cycles = tuple(int(value) for value in nuisance_values.get("cycles", (1, 2)))
    if not nuisance_cycles or any(value <= 0 for value in nuisance_cycles):
        raise ValueError("Nuisance cycles must contain positive integers")
    nuisance_phase = float(nuisance_values.get("phase_offset_seconds", expected_offset))
    if nuisance_phase != expected_offset:
        raise ValueError("Nuisance phase_offset_seconds must match the +5 s fMRI offset")
    dose_levels = tuple(
        float(value)
        for value in data_settings["studies"]["B_lambda"]["nuisance_dose_levels"]
    )
    use_nuisance = stage["view"] == "B_lambda"

    root_infra = config["infra"]
    root_infra.update(
        {
            "cluster": None,
            "folder": str(stage_dir),
            "gpus_per_node": 1,
            "cpus_per_task": cpus,
            "mode": "retry",
        }
    )
    root_infra.pop("workdir", None)

    data = config["data"]
    timeline_cache = cache_root / "timelines"
    timeline_cache.parent.mkdir(parents=True, exist_ok=True)
    data["study"] = {
        "names": "Lahner2024Bold",
        "path": str(study_parent),
        "query": _study_query(
            subjects=subjects,
            train_sessions=train_sessions,
            test_sessions=test_sessions,
        ),
        "infra_timelines": _local_infra(timeline_cache, cpus=cpus),
        "transforms": {"fixed_split": {"name": "FixedBmdTaskSplit"}},
    }
    data.update(
        {
            "frequency": int(data_settings["input_hz"]),
            "duration_trs": int(data_settings["window_seconds"]),
            "overlap_trs_train": 0,
            "overlap_trs_val": 0,
            "stride_drop_incomplete": True,
            "split_segments_by_time": False,
            "shuffle_train": True,
            "shuffle_val": False,
            "num_workers": cpus,
            "batch_size": int(training["batch_size"]),
            "layers_to_use": [0.75, 1.0],
            "n_layers_to_use": None,
            # Keep the selected layer axis intact in neuralset; concatenation
            # belongs to FmriEncoder.layer_aggregation below.  Data only accepts
            # group_mean, mean, or None.
            "layer_aggregation": None,
            "features_to_use": ["video"],
            "features_to_mask": [],
            "text_feature": None,
            "audio_feature": None,
            "image_feature": None,
            "replay_fraction": float(stage.get("replay_fraction", 0.0)),
            "replay_session": stage.get("replay_session"),
            "replay_seed": seed,
        }
    )

    feature_cache = Path(feature_settings["cache"])
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    model_snapshot = validate_vjepa_snapshot(
        feature_settings["video_model_provenance"],
        repo_id=feature_settings["video_model"],
        revision=feature_settings["video_model_revision"],
        expected_model_sha256=feature_settings["video_model_sha256"],
    )
    video_feature = config["data"]["video_feature"]
    video_feature["infra"] = _local_infra(feature_cache, cpus=cpus, gpu=True)
    video_feature["infra"]["version"] = (
        f"vjepa2-{feature_settings['video_model_revision'][:12]}"
    )
    video_feature["use_audio"] = False
    video_feature["image"]["name"] = "PinnedLocalHuggingFaceImage"
    video_feature["image"]["model_name"] = str(model_snapshot)
    # HuggingFaceVideo owns the persistent cache and execution backend.  The
    # nested image model is deliberately in-process: neuralset 0.0.2 rejects a
    # nested folder/cluster here to prevent two competing cache layers.
    video_feature["image"]["infra"] = {
        "cluster": None,
        "folder": None,
        "keep_in_ram": False,
    }
    video_feature["image"]["batch_size"] = 1
    data["video_feature"] = video_feature

    # The extractor's Exca UID already includes its full configuration and the
    # event/timeline UID. Keep one shared folder across conditions, stages, and
    # model seeds so the expensive MNI -> fsaverage5 -> Glasser projection is
    # computed once per unique run/view instead of once per training stage.
    neuro_cache = cache_root / "fmri"
    neuro_cache.mkdir(parents=True, exist_ok=True)
    neuro = {
        "name": "NuisanceFmriExtractor",
        "allow_missing": False,
        "offset": expected_offset,
        "frequency": int(data_settings["output_hz"]),
        "projection": {
            "name": "CompiledGlasser360Projector",
            "mesh": "fsaverage5",
            "kind": "ball",
            "radius": 3,
            "operator_path": str(operator_path),
            "mesh_data_dir": str(operator_path.parent / "nilearn"),
        },
        "nuisance_amplitude": float(nuisance_values.get("amplitude", 0.23)) if use_nuisance else 0.0,
        "nuisance_seed": int(nuisance_values.get("seed", 20260818)),
        "nuisance_basis_path": str(basis_path),
        "nuisance_chunk_seconds": int(nuisance_values.get("chunk_seconds", 100)),
        "nuisance_cycles": nuisance_cycles,
        "nuisance_temporal_jitter": float(nuisance_values.get("temporal_jitter", 0.3)),
        "nuisance_spatial_jitter": float(nuisance_values.get("spatial_jitter", 0.25)),
        "nuisance_dose_levels": dose_levels if use_nuisance else (),
        "nuisance_phase_offset_seconds": nuisance_phase,
        "nuisance_train_only": True,
        "nuisance_sessions": (3,) if use_nuisance else (),
        "infra": _local_infra(neuro_cache, cpus=cpus),
    }
    cache_identity = reproducibility["components"]
    neuro["infra"]["version"] = "-".join(
        (
            "glasser360-nuisance" if use_nuisance else "glasser360-clean",
            str(cache_identity["source_tree"])[:10],
            str(cache_identity["atlas_operator"])[:10],
            str(cache_identity["atlas_basis"])[:10],
        )
    )
    data["neuro"] = neuro

    model_path = _resolve_project_path(experiment["model"]["config"])
    model = load_yaml(model_path).get("model", {})
    config["brain_model_config"] = {
        "name": "FmriEncoder",
        "hidden": int(model["hidden"]),
        "max_seq_len": int(model["input_hz"] * model["window_seconds"]),
        "low_rank_head": int(model["low_rank_head"]),
        "dropout": float(model["dropout"]),
        "extractor_aggregation": "cat",
        "layer_aggregation": "cat",
        "combiner": None,
        "encoder": {
            "depth": int(model["depth"]),
            "heads": int(model["heads"]),
            "ff_mult": int(model["ff_mult"]),
        },
        "subject_layers": {
            "n_subjects": int(model["n_subjects"]),
            "subject_dropout": float(model["subject_dropout"]),
            "average_subjects": False,
        },
        "subject_embedding": False,
        "modality_dropout": 0.0,
    }
    optimizer = training["optimizer"]
    scheduler = training["scheduler"]
    config["optim"] = {
        "name": "LightningOptimizer",
        "optimizer": {
            "name": optimizer["name"],
            "lr": float(optimizer["learning_rate"]),
            "kwargs": {"weight_decay": float(optimizer["weight_decay"])},
        },
        "scheduler": {
            "name": scheduler["name"],
            "kwargs": {
                "max_lr": float(scheduler["max_learning_rate"]),
                "pct_start": float(scheduler["warmup_fraction"]),
            },
        },
    }
    config.update(
        {
            "seed": seed,
            "n_epochs": int(training["epochs_per_stage"]),
            "accumulate_grad_batches": int(training["gradient_accumulation"]),
            "precision": training["precision"],
            "gradient_clip_val": float(training["gradient_clip_norm"]),
            "accelerator": "gpu",
            # There is deliberately no validation-based model selection.  Every
            # stage runs the fixed epoch budget and exports its final tensors;
            # the official test task stays untouched for post-stage evaluation.
            "save_checkpoints": False,
            "checkpoint_filename": "unused",
            "checkpoint_path": None,
            "load_checkpoint": False,
            "patience": None,
            "wandb_config": None,
        }
    )
    return config


def _required_paths(experiment: Mapping[str, Any]) -> list[Path]:
    data = experiment["data"]
    cache = Path(experiment["experiment"]["cache_root"])
    atlas = experiment.get("atlas", {})
    return [
        Path(data["root"]),
        Path(experiment["experiment"]["manifest"]),
        Path(atlas.get("operator", cache / "atlas" / "glasser360_fsaverage5.npz")),
        Path(atlas.get("basis", cache / "atlas" / "glasser360_laplacian_basis.npz")),
        Path(experiment["features"]["video_model_provenance"]),
    ]


def _validate_manifest_contract(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Require the frozen manifest to describe the exact live Phase-1 subset."""

    from spfcl.data.manifest import (
        audit_pseudostudies,
        build_manifest,
        validate_bmd_layout,
    )
    from spfcl.data.snapshot import OpenNeuroSnapshotDownloader

    data = experiment["data"]
    download_root = Path(data["root"]) / "download"
    download_provenance = json.loads(
        (download_root / "spfcl_download.json").read_text(encoding="utf8")
    )
    if download_provenance.get("backend") != "snapshot":
        raise RuntimeError(
            "Phase-1 forbids the mutable s3-latest backend; use the immutable snapshot backend"
        )
    if download_provenance.get("expected_snapshot") != str(data.get("snapshot", "1.0.4")):
        raise RuntimeError("OpenNeuro download provenance has the wrong snapshot")
    inventory_path = download_root / "spfcl_snapshot_inventory.json"
    inventory_metadata = OpenNeuroSnapshotDownloader.read_inventory_metadata(
        inventory_path
    )
    if inventory_metadata is None:
        raise RuntimeError("OpenNeuro inventory lacks its schema-v1 selection contract")
    required_task_sessions = {
        "train": {2, 3},
        "test": {
            int(value)
            for value in data["evaluation"].get("test_sessions", [2, 3])
        },
        "localizer": {1},
    }
    inventory_tasks = {
        name: {int(value) for value in values}
        for name, values in (inventory_metadata.get("task_sessions") or {}).items()
    }
    identity_ok = (
        inventory_metadata.get("dataset") == "ds005165"
        and inventory_metadata.get("snapshot") == str(data.get("snapshot", "1.0.4"))
        and inventory_metadata.get("space") == str(data.get("source_space", "mni"))
        and inventory_metadata.get("subjects")
        == sorted(int(value) for value in data["subjects"])
    )
    sessions_ok = all(
        required.issubset(inventory_tasks.get(task, set()))
        for task, required in required_task_sessions.items()
    )
    if not identity_ok or not sessions_ok:
        raise RuntimeError(
            "OpenNeuro inventory selection does not cover the preregistered Phase-1 subset"
        )
    manifest_path = Path(experiment["experiment"]["manifest"])
    frozen = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    current_records = build_manifest(
        data["root"],
        subjects=tuple(int(value) for value in data["subjects"]),
        sessions=(1, 2, 3),
        space=str(data.get("source_space", "mni")),
    )
    current = [asdict(record) for record in current_records]
    if frozen != current:
        raise RuntimeError(
            "The frozen manifest no longer matches the live BOLD Moments files. "
            "Rerun validate-data, make-manifest, and audit-pseudostudies before training."
        )
    issues = validate_bmd_layout(
        data["root"],
        subjects=tuple(int(value) for value in data["subjects"]),
        expected_snapshot=str(data.get("snapshot", "1.0.4")),
        check_videos=True,
        space=str(data.get("source_space", "mni")),
        deep=False,
    )
    errors = [issue.message for issue in issues if issue.level == "error"]
    if errors:
        raise RuntimeError("BOLD Moments preflight failed: " + " | ".join(errors))
    counts: dict[str, int] = {}
    for record in current_records:
        counts[record.role] = counts.get(record.role, 0) + 1
    expected = {"A": 40, "B": 40, "clean_test": 24, "probe": 20}
    if counts != expected:
        raise RuntimeError(f"Phase-1 manifest role counts differ: {counts} != {expected}")
    report = audit_pseudostudies(data["root"], current_records)
    if not all(report["cross_subject_sets_identical"].values()):
        raise RuntimeError("Stimulus sets differ across subjects in the frozen manifest")
    if report["train_test_overlap"] or report["train_probe_overlap"]:
        raise RuntimeError(f"Training/evaluation stimulus leakage detected: {report}")
    return {
        "counts": counts,
        "stimulus_audit": report,
        "warnings": [issue.message for issue in issues if issue.level != "error"],
    }


def _segment_uid(segment: Any) -> str:
    timeline = "unknown"
    events = getattr(segment, "ns_events", None) or []
    if events:
        timeline = str(getattr(events[0], "timeline", timeline))
    return f"{timeline}@{float(segment.start):.3f}+{float(segment.duration):.3f}"


def _export_clean_test_predictions(experiment: Any, output: Path) -> dict[str, Any]:
    """Run the untouched official test task with known and group heads."""

    import numpy as np
    import torch

    from spfcl.eval.metrics import encoding_metrics
    from spfcl.model.tribe_adapter import head_mode

    events = experiment.data.get_events()
    loaders = experiment.data.get_loaders(events=events, split_to_build="clean_test")
    if "clean_test" not in loaders:
        raise RuntimeError("FixedBmdTaskSplit did not expose a clean_test loader")
    brain = experiment._model.model
    original_device = next(brain.parameters()).device
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else original_device
    )
    brain.to(device)
    was_training = brain.training
    brain.eval()
    empirical_batches = []
    subject_batches = []
    group_batches = []
    subject_ids = []
    segment_uids: list[str] = []
    with torch.inference_mode():
        for batch in loaders["clean_test"]:
            moved = batch.to(device)
            empirical_batches.append(batch.data["fmri"].detach().cpu().numpy())
            subject_ids.append(batch.data["subject_id"].detach().cpu().numpy())
            segment_uids.extend(_segment_uid(segment) for segment in batch.segments)
            with head_mode(brain, "subject"):
                subject_batches.append(brain(moved).detach().cpu().numpy())
            with head_mode(brain, "group"):
                group_batches.append(brain(moved).detach().cpu().numpy())
    if was_training:
        brain.train()
    if device != original_device:
        brain.to(original_device)
    empirical = np.concatenate(empirical_batches, axis=0).astype(np.float32)
    subject_predicted = np.concatenate(subject_batches, axis=0).astype(np.float32)
    group_predicted = np.concatenate(group_batches, axis=0).astype(np.float32)
    subjects = np.concatenate(subject_ids, axis=0).reshape(-1).astype(np.int64)
    _savez_atomic(
        output,
        encoding_empirical=empirical,
        encoding_predicted=subject_predicted,
        group_predicted=group_predicted,
        subject_id=subjects,
        segment_uid=np.asarray(segment_uids),
        split=np.asarray("clean_test"),
    )

    def summarize(predicted: Any) -> dict[str, float]:
        metrics = encoding_metrics(empirical, predicted)
        per_segment_z = np.nanmean(metrics["parcel_fisher_z"], axis=-1)
        subject_z = [
            float(np.nanmean(per_segment_z[subjects == subject]))
            for subject in sorted(set(subjects.tolist()))
        ]
        return {
            "subject_equal_mean_fisher_z": float(np.nanmean(subject_z)),
            "subject_equal_mean_pearson": float(np.tanh(np.nanmean(subject_z))),
            "mean_mse": float(metrics["mean_mse"]),
            "mean_r2": float(metrics["mean_r2"]),
        }

    return {
        "path": str(output),
        "segments": int(empirical.shape[0]),
        "known_subject": summarize(subject_predicted),
        "group": summarize(group_predicted),
    }


def _localizer_labels(segment: Any, n_time: int, frequency: float = 1.0):
    import numpy as np

    labels = np.full(n_time, "", dtype="<U16")
    events = segment.events
    if "localizer_condition" not in events:
        return labels
    times = float(segment.start) + np.arange(n_time, dtype=np.float64) / frequency
    videos = events[events["type"].astype(str) == "Video"]
    for _, event in videos.iterrows():
        condition = str(event.get("localizer_condition", "")).lower()
        if condition not in {"faces", "bodies", "scenes", "objects", "scrambled"}:
            continue
        start = float(event["start"])
        stop = float(event.get("stop", start + float(event.get("duration", 0.0))))
        labels[(times >= start) & (times < stop)] = condition
    return labels


def _localizer_probe_loader(experiment: Any) -> Any:
    """Build the probe loader, preparing its frozen features before brain training."""

    from tribev2.utils import MultiStudyLoader

    subjects = sorted(experiment.data.subject_id.predefined_mapping)
    subject_numbers = [int(value.rsplit("/", 1)[1]) for value in subjects]
    query = " or ".join(
        f"subject == 'Lahner2024BoldLocalizer/{subject}'"
        for subject in subject_numbers
    )
    timeline_infra = experiment.data.study.infra_timelines.model_dump(
        exclude_defaults=True
    )
    study = MultiStudyLoader(
        names="Lahner2024BoldLocalizer",
        path=experiment.data.study.path,
        query=f"({query})",
        infra_timelines=timeline_infra,
        transforms={},
    )
    events = study.run().copy(deep=True)
    events["subject"] = events["subject"].astype(str).str.replace(
        "Lahner2024BoldLocalizer/", "Lahner2024Bold/", regex=False
    )
    events["split"] = "probe"
    loaders = experiment.data.get_loaders(events=events, split_to_build="probe")
    if "probe" not in loaders:
        raise RuntimeError("The BOLD Moments localizer produced no complete probe segments")
    return loaders["probe"]


def _prepare_localizer_cache(experiment: Any) -> None:
    """Populate V-JEPA/fMRI probe caches before the trainable model occupies GPU RAM."""

    loader = _localizer_probe_loader(experiment)
    del loader
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _export_localizer_predictions(experiment: Any, output: Path) -> dict[str, Any]:
    """Evaluate the same four subjects on the independent silent-video localizer."""

    import numpy as np
    import torch

    from spfcl.eval.metrics import localizer_contrast_metrics
    from spfcl.model.tribe_adapter import head_mode
    probe_loader = _localizer_probe_loader(experiment)

    brain = experiment._model.model
    original_device = next(brain.parameters()).device
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else original_device
    )
    brain.to(device)
    was_training = brain.training
    brain.eval()
    empirical_batches = []
    subject_batches = []
    group_batches = []
    id_batches = []
    label_batches = []
    with torch.inference_mode():
        for batch in probe_loader:
            moved = batch.to(device)
            empirical = batch.data["fmri"].detach().cpu().numpy()
            empirical_batches.append(empirical)
            id_batches.append(batch.data["subject_id"].detach().cpu().numpy())
            label_batches.append(
                np.stack(
                    [_localizer_labels(segment, empirical.shape[-1]) for segment in batch.segments]
                )
            )
            with head_mode(brain, "subject"):
                subject_batches.append(brain(moved).detach().cpu().numpy())
            with head_mode(brain, "group"):
                group_batches.append(brain(moved).detach().cpu().numpy())
    if was_training:
        brain.train()
    if device != original_device:
        brain.to(original_device)

    empirical = np.concatenate(empirical_batches, axis=0)
    subject_predicted = np.concatenate(subject_batches, axis=0)
    group_predicted = np.concatenate(group_batches, axis=0)
    subject_id = np.concatenate(id_batches, axis=0).reshape(-1).astype(np.int64)
    labels = np.concatenate(label_batches, axis=0)
    conditions = ("faces", "bodies", "scenes", "objects", "scrambled")
    unique_subjects = sorted(set(subject_id.tolist()))

    def aggregate(values: Any) -> Any:
        output_values = np.empty(
            (len(unique_subjects), len(conditions), values.shape[1]), dtype=np.float32
        )
        for subject_index, subject in enumerate(unique_subjects):
            for condition_index, condition in enumerate(conditions):
                selected = (subject_id[:, None] == subject) & (labels == condition)
                if not np.any(selected):
                    raise RuntimeError(
                        f"Localizer has no samples for subject={subject}, condition={condition}"
                    )
                # B,P,T -> B,T,P, followed by a boolean B,T selection.
                output_values[subject_index, condition_index] = values.transpose(0, 2, 1)[
                    selected
                ].mean(axis=0)
        return output_values

    empirical_conditions = aggregate(empirical)
    subject_conditions = aggregate(subject_predicted)
    group_conditions = aggregate(group_predicted)
    _savez_atomic(
        output,
        localizer_empirical=empirical_conditions,
        localizer_predicted=subject_conditions,
        localizer_group_predicted=group_conditions,
        conditions=np.asarray(conditions),
        subject_id=np.asarray(unique_subjects, dtype=np.int64),
        positive_condition=np.asarray(0),
        negative_condition=np.asarray(1),
    )

    def contrasts(predicted: Any) -> dict[str, Any]:
        output_metrics: dict[str, Any] = {}
        for name, positive, negative in (
            ("faces_vs_bodies", 0, 1),
            ("scenes_vs_objects", 2, 3),
        ):
            metrics = localizer_contrast_metrics(
                empirical_conditions,
                predicted,
                positive=positive,
                negative=negative,
            )
            output_metrics[name] = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in metrics.items()
            }
        return output_metrics

    return {
        "path": str(output),
        "segments": int(empirical.shape[0]),
        "conditions": list(conditions),
        "known_subject": contrasts(subject_conditions),
        "group": contrasts(group_conditions),
    }


def _run_stage(
    experiment: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    condition: str,
    seed: int,
    output_root: Path,
    parent: WeightsCheckpoint | None,
    reproducibility: Mapping[str, Any],
) -> tuple[WeightsCheckpoint, dict[str, Any]]:
    stage_dir = output_root / stage["id"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    exported_path = stage_dir / "stage.weights.pt"
    audit_path = stage_dir / "spfcl_stage_audit.json"
    clean_test_path = stage_dir / "clean_test_predictions.npz"
    localizer_path = stage_dir / "localizer_predictions.npz"
    stage_fingerprint = _canonical_sha256(
        {"experiment": reproducibility["sha256"], "stage": stage}
    )
    if exported_path.is_file():
        exported = read_weights_metadata(exported_path)
        metadata = exported.metadata
        expected_parent_sha256 = parent.state_sha256 if parent is not None else None
        expected = {
            "condition": condition,
            "stage": stage["id"],
            "seed": seed,
            "parent_state_sha256": expected_parent_sha256,
            "experiment_fingerprint_sha256": reproducibility["sha256"],
            "stage_fingerprint_sha256": stage_fingerprint,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing stage export has incompatible provenance: {mismatches}. "
                "Choose a new output root instead of overwriting it."
            )
        clean_artifact = _inspect_prediction_archive(
            clean_test_path,
            {
                "encoding_empirical",
                "encoding_predicted",
                "group_predicted",
                "subject_id",
                "segment_uid",
            },
        )
        localizer_artifact = _inspect_prediction_archive(
            localizer_path,
            {
                "localizer_empirical",
                "localizer_predicted",
                "localizer_group_predicted",
                "conditions",
                "subject_id",
            },
        )
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf8"))
            if audit.get("final_state_sha256") != exported.state_sha256:
                raise RuntimeError(
                    "Stage audit final hash differs from the weights artifact: "
                    f"{audit_path}"
                )
        else:
            audit = {
                "stage": stage["id"],
                "view": stage["view"],
                "parent": stage.get("parent"),
                "initial_state_sha256": metadata.get("initial_state_sha256"),
                "final_state_sha256": exported.state_sha256,
                "stage_load": None,
                "weights": str(exported.path),
                "clean_test": clean_artifact,
                "localizer": localizer_artifact,
                "experiment_fingerprint_sha256": reproducibility["sha256"],
                "stage_fingerprint_sha256": stage_fingerprint,
                "reused_existing_export": True,
            }
            _write_json_atomic(audit_path, audit)
        return exported, audit

    official_config = _official_stage_config(
        experiment,
        stage,
        condition=condition,
        seed=seed,
        stage_dir=stage_dir,
        reproducibility=reproducibility,
    )
    model = build_weights_only_experiment(
        official_config,
        parent_checkpoint=parent.path if parent else None,
        parent_state_sha256=parent.state_sha256 if parent else None,
    )
    _prepare_localizer_cache(model)
    model.run_fixed_epochs()
    initial_hash = model.initial_state_sha256
    clean_test = _export_clean_test_predictions(
        model, clean_test_path
    )
    localizer = _export_localizer_predictions(
        model, localizer_path
    )
    # Publish the legal next-stage boundary only after both independent
    # evaluations succeed. An interrupted evaluation can never masquerade as a
    # completed continual stage on restart.
    exported = export_experiment_weights(
        model,
        exported_path,
        stage=stage["id"],
        condition=condition,
        seed=seed,
        source_checkpoint=None,
        extra_metadata={
            "view": stage["view"],
            "initial_state_sha256": initial_hash,
            "replay_fraction": float(stage.get("replay_fraction", 0.0)),
            "checkpoint_policy": "fixed_final_epoch_no_validation_selection",
            "completed_epochs": int(experiment["training"]["epochs_per_stage"]),
            "experiment_fingerprint_sha256": reproducibility["sha256"],
            "stage_fingerprint_sha256": stage_fingerprint,
            "reproducibility_components": reproducibility["components"],
        },
    )
    audit = {
        "stage": stage["id"],
        "view": stage["view"],
        "parent": stage.get("parent"),
        "initial_state_sha256": initial_hash,
        "final_state_sha256": exported.state_sha256,
        "stage_load": model.stage_load_audit,
        "weights": str(exported.path),
        "clean_test": clean_test,
        "localizer": localizer,
        "experiment_fingerprint_sha256": reproducibility["sha256"],
        "stage_fingerprint_sha256": stage_fingerprint,
    }
    _write_json_atomic(audit_path, audit)
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return exported, audit


def run_condition(
    config_path: str | Path,
    *,
    condition: str,
    seed: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_file = _resolve_project_path(config_path)
    experiment = load_yaml(config_file)
    conditions = experiment["matrix"]["conditions"]
    if condition not in conditions:
        raise ValueError(f"Condition {condition!r} is absent from {config_file}")
    if seed not in [int(value) for value in experiment["matrix"]["seeds"]]:
        raise ValueError(f"Seed {seed} is not preregistered in {config_file}")
    if experiment["model"].get("initialization") != "random":
        raise ValueError("Phase-1 must start from random TRIBE trainable layers")
    if experiment["model"].get("public_joint_checkpoint") is not None:
        raise ValueError("The public joint checkpoint is forbidden as Phase-1 initialization")
    if experiment["data"].get("input_modalities") != ["video"]:
        raise ValueError("Phase-1 production configuration must be video-only")
    if int(experiment["data"].get("window_seconds", 0)) != 100:
        raise ValueError("Phase-1 uses fixed non-overlapping 100-second windows")
    if int(experiment["data"].get("overlap_seconds", 0)) != 0:
        raise ValueError("Chunk-synchronous nuisance requires overlap_seconds=0")
    scope_fraction = float(
        experiment["data"].get("selected_subject_session_fraction_of_full_bmd", -1)
    )
    if abs(scope_fraction - 0.20) > 1e-12:
        raise ValueError("Phase-1 must use the preregistered 20% subject-session scope")
    subjects = [int(value) for value in experiment["data"].get("subjects", [])]
    if subjects != [1, 2, 3, 4]:
        raise ValueError("The preregistered Phase-1 subject mapping is exactly [1, 2, 3, 4]")
    if experiment["data"]["studies"]["A"]["session"] == experiment["data"]["studies"]["B0"]["session"]:
        raise ValueError("Pseudo-studies A and B must use different sessions")
    doses = experiment["data"]["studies"]["B_lambda"].get("nuisance_dose_levels", [])
    if len(doses) != 3 or any(float(value) < 0 for value in doses):
        raise ValueError("B_lambda must preregister exactly three non-negative dose levels")
    from spfcl.model.tribe_adapter import TribeLiteSpec

    model_values = load_yaml(_resolve_project_path(experiment["model"]["config"]))["model"]
    TribeLiteSpec.from_mapping(model_values)

    plan = _stage_plan(experiment, condition)
    output_root = (
        Path(experiment["experiment"]["output_root"]) / condition / f"seed-{seed}"
    )
    required = _required_paths(experiment)
    result: dict[str, Any] = {
        "config": str(config_file),
        "condition": condition,
        "seed": seed,
        "upstream_commit": UPSTREAM_COMMIT,
        "output_root": str(output_root),
        "stages": plan,
        "required_paths": [
            {"path": str(path), "exists": path.exists()} for path in required
        ],
        "dry_run": dry_run,
    }
    if dry_run:
        return result
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Training prerequisites are missing: " + ", ".join(str(path) for path in missing)
        )
    result["manifest_contract"] = _validate_manifest_contract(experiment)
    reproducibility = _experiment_fingerprint(experiment, config_file)
    result["reproducibility"] = reproducibility

    os.environ.setdefault("DATAPATH", str(Path(experiment["data"]["root"]).parent))
    os.environ.setdefault("SAVEPATH", str(output_root.parent))
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, WeightsCheckpoint] = {}
    audits: list[dict[str, Any]] = []
    pair_hashes: dict[str, str] = {}
    for stage in plan:
        parent_name = stage.get("parent")
        parent = checkpoints[parent_name] if parent_name else None
        checkpoint, audit = _run_stage(
            experiment,
            stage,
            condition=condition,
            seed=seed,
            output_root=output_root,
            parent=parent,
            reproducibility=reproducibility,
        )
        checkpoints[stage["id"]] = checkpoint
        audits.append(audit)
        pair = stage.get("pair")
        if pair:
            previous = pair_hashes.setdefault(pair, audit["initial_state_sha256"])
            if previous != audit["initial_state_sha256"]:
                raise RuntimeError(
                    f"Paired branches {pair} did not share the same random initial state"
                )
    result["dry_run"] = False
    result["audits"] = audits
    result["final_weights"] = {
        name: {"path": str(value.path), "sha256": value.state_sha256}
        for name, value in checkpoints.items()
    }
    _write_json_atomic(output_root / "spfcl_condition_audit.json", result)
    return result
