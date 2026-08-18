from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_yaml
from .data.download import (
    DEFAULT_SNAPSHOT,
    OpenNeuroSelectiveDownloader,
    bmd_study_root,
    write_download_provenance,
)
from .data.manifest import (
    audit_pseudostudies,
    build_manifest,
    validate_bmd_layout,
    write_manifest,
)
from .data.snapshot import OpenNeuroSnapshotDownloader
from .data.stimuli import download_stimulus_archive, install_stimuli


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=os.fspath))


def _doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail})

    add(
        "python",
        sys.version_info >= (3, 11),
        platform.python_version(),
    )
    root = Path(__file__).resolve().parents[2]
    vendor = root / "vendor" / "tribev2" / "tribev2" / "model.py"
    add("vendored TRIBE v2", vendor.is_file(), os.fspath(vendor))
    for command, required in (("ffmpeg", True), ("nvidia-smi", False)):
        found = shutil.which(command)
        add(command, found is not None, found or "not found", required=required)
    for variable, required in (("DATAPATH", True), ("SAVEPATH", True), ("HF_HOME", False)):
        raw = os.getenv(variable)
        if raw:
            path = Path(raw).expanduser()
            writable = path.exists() and os.access(path, os.W_OK)
            add(variable, writable, f"{path} (writable={writable})", required=required)
        else:
            add(variable, False, "unset", required=required)
    try:
        import torch

        add(
            "torch",
            True,
            f"{torch.__version__}; cuda={torch.cuda.is_available()}",
        )
    except ImportError:
        add("torch", False, "not installed")

    if args.json:
        _json({"checks": checks})
    else:
        for item in checks:
            state = "OK" if item["ok"] else ("WARN" if not item["required"] else "FAIL")
            print(f"[{state:4}] {item['name']}: {item['detail']}")
    return 0 if all(item["ok"] or not item["required"] for item in checks) else 5


def _download_bmd(args: argparse.Namespace) -> int:
    study_root = bmd_study_root(args.data_root)
    destination = study_root / "download"
    if args.sessions:
        sessions = sorted(set(args.sessions))
        task_sessions = None
    else:
        task_sessions = {
            "train": set(args.train_sessions or [2, 3]),
            "test": set(args.test_sessions or [2, 3]),
            "localizer": set(args.localizer_sessions or [1]),
        }
        sessions = sorted(set().union(*task_sessions.values()))
    if args.backend == "snapshot":
        downloader = OpenNeuroSnapshotDownloader(snapshot=args.snapshot)
        selection_metadata = {
            "dataset": downloader.dataset,
            "snapshot": args.snapshot,
            "subjects": sorted(set(args.subjects)),
            "sessions": sessions,
            "task_sessions": (
                {key: sorted(value) for key, value in task_sessions.items()}
                if task_sessions is not None
                else None
            ),
            "space": args.space,
        }
        if args.inventory_in:
            actual_metadata = downloader.read_inventory_metadata(args.inventory_in)
            if actual_metadata is None:
                raise ValueError(
                    "--inventory-in requires a schema-v1 inventory with selection metadata"
                )
            if actual_metadata != selection_metadata:
                raise ValueError(
                    "--inventory-in selection differs from command arguments: "
                    f"inventory={actual_metadata}, requested={selection_metadata}"
                )
            objects = downloader.read_inventory(args.inventory_in)
        else:
            objects = downloader.select_bmd(
                args.subjects,
                sessions,
                space=args.space,
                task_sessions=task_sessions,
            )
        summary = downloader.download(
            objects,
            destination,
            workers=args.workers,
            dry_run=args.dry_run,
            inventory_metadata=selection_metadata,
        )
        if not args.dry_run:
            inventory = (
                Path(args.inventory)
                if args.inventory
                else study_root
                / "manifests"
                / f"ds005165-{args.snapshot}-{args.space}.json"
            )
            downloader.write_inventory(objects, inventory, metadata=selection_metadata)
    else:
        if args.snapshot != DEFAULT_SNAPSHOT:
            raise ValueError("--backend s3-latest cannot promise a historical snapshot")
        downloader = OpenNeuroSelectiveDownloader()
        objects = downloader.select_bmd(
            args.subjects,
            sessions,
            space=args.space,
            task_sessions=task_sessions,
        )
        summary = downloader.download(
            objects, destination, workers=args.workers, dry_run=args.dry_run
        )
    if not args.dry_run:
        write_download_provenance(
            study_root,
            subjects=args.subjects,
            sessions=sessions,
            snapshot=args.snapshot,
            summary=summary,
            backend=args.backend,
            space=args.space,
            task_sessions=task_sessions,
        )
    _json(
        {
            "backend": args.backend,
            "snapshot": args.snapshot,
            "space": args.space,
            "subjects": args.subjects,
            "sessions": sessions,
            "task_sessions": (
                {key: sorted(value) for key, value in task_sessions.items()}
                if task_sessions is not None
                else None
            ),
            "selected_files": summary.selected,
            "selected_gib": round(summary.bytes_selected / 2**30, 3),
            "downloaded": summary.downloaded,
            "skipped": summary.skipped,
            "destination": summary.destination,
            "dry_run": args.dry_run,
        }
    )
    return 0


def _download_stimuli(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else (
        bmd_study_root(args.data_root) / "stimuli" / "stimulus_set.zip"
    )
    path = download_stimulus_archive(
        output,
        accept_stimulus_terms=args.accept_stimulus_terms,
        dry_run=args.dry_run,
    )
    print(path)
    return 0


def _install_stimuli(args: argparse.Namespace) -> int:
    path = install_stimuli(
        args.data_root,
        archive=args.archive,
        source_dir=args.source_dir,
        password_env=args.password_env,
        link=args.link,
        force=args.force,
        dry_run=args.dry_run,
        accept_stimulus_terms=args.accept_stimulus_terms,
    )
    print(path)
    return 0


def _validate_data(args: argparse.Namespace) -> int:
    issues = validate_bmd_layout(
        args.data_root,
        subjects=args.subjects,
        expected_snapshot=args.snapshot,
        check_videos=not args.skip_videos,
        space=args.space,
        deep=args.deep,
    )
    payload = [item.__dict__ for item in issues]
    if args.json:
        _json({"issues": payload, "ready": not any(x.level == "error" for x in issues)})
    elif not issues:
        print("BOLD Moments subset is ready.")
    else:
        for item in issues:
            print(f"[{item.level.upper()}] {item.message}")
    return 3 if any(item.level == "error" for item in issues) else 0


def _make_manifest(args: argparse.Namespace) -> int:
    records = build_manifest(
        args.data_root,
        subjects=args.subjects,
        sessions=args.sessions,
        space=args.space,
    )
    output = write_manifest(records, args.output)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.role] = counts.get(record.role, 0) + 1
    _json({"output": output, "records": len(records), "roles": counts})
    return 0


def _audit_pseudostudies(args: argparse.Namespace) -> int:
    records = build_manifest(
        args.data_root,
        subjects=args.subjects,
        sessions=args.sessions,
        space=args.space,
    )
    report = audit_pseudostudies(args.data_root, records)
    _json(report)
    if not all(report["cross_subject_sets_identical"].values()):
        return 3
    if report["train_test_overlap"] or report["train_probe_overlap"]:
        return 3
    return 0


def _prepare_atlas(args: argparse.Namespace) -> int:
    from .data.atlas import prepare_glasser360

    paths = prepare_glasser360(
        args.output_dir,
        accept_hcp_license=args.accept_hcp_license,
        n_modes=args.n_modes,
    )
    _json(paths)
    return 0


def _nuisance_preview(args: argparse.Namespace) -> int:
    from .data.nuisance import DriftConfig, load_basis, save_preview

    payload = load_yaml(args.config)
    values = payload.get("nuisance", payload)
    values = dict(values)
    if "cycles" in values:
        values["cycles"] = tuple(values["cycles"])
    if "dose_levels" in values:
        values["dose_levels"] = tuple(values["dose_levels"])
    basis_path = values.pop("basis_path", None)
    config = DriftConfig(**values)
    if basis_path:
        basis = load_basis(basis_path)
    else:
        # Deterministic smooth ring modes keep this command dataset/atlas independent.
        theta = np.linspace(0, 2 * np.pi, args.parcels, endpoint=False)
        basis = np.stack(
            [function(mode * theta) for mode in range(1, 9) for function in (np.sin, np.cos)],
            axis=1,
        ).astype(np.float32)
    path = save_preview(args.output, config=config, spatial_basis=basis)
    print(path)
    return 0


def _model_smoke(args: argparse.Namespace) -> int:
    from .model.tribe_adapter import TribeLiteSpec, synthetic_smoke

    payload = load_yaml(args.config)
    values = payload.get("model", payload)
    result = synthetic_smoke(spec=TribeLiteSpec.from_mapping(values))
    _json(result)
    return 0


def _download_vjepa(args: argparse.Namespace) -> int:
    from .model.hub import download_vjepa_snapshot

    payload = load_yaml(args.config)
    features = payload["features"]
    cache_dir = args.cache_dir or os.getenv("HF_HOME")
    if not cache_dir:
        raise ValueError("Set HF_HOME or pass --cache-dir")
    result = download_vjepa_snapshot(
        repo_id=features["video_model"],
        revision=features["video_model_revision"],
        expected_model_sha256=features["video_model_sha256"],
        cache_dir=cache_dir,
        provenance_path=args.provenance or features["video_model_provenance"],
        dry_run=args.dry_run,
    )
    _json(result)
    return 0


def _evaluate_bundle(args: argparse.Namespace) -> int:
    from .eval.pipeline import evaluate_npz_bundle

    output = evaluate_npz_bundle(args.input, args.output)
    print(output)
    return 0


def _train(args: argparse.Namespace) -> int:
    from .train.runner import run_condition

    result = run_condition(
        args.config,
        condition=args.condition,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    _json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spfcl", description="Remote TRIBE-lite / SPF continual-learning toolkit"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check the remote training environment")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(function=_doctor)

    download = sub.add_parser("download-bmd", help="selectively fetch OpenNeuro ds005165")
    download.add_argument("--data-root", required=True)
    download.add_argument("--subjects", nargs="+", type=_positive_int, default=[1, 2, 3, 4])
    download.add_argument(
        "--sessions",
        nargs="+",
        type=_positive_int,
        help="legacy shorthand: take permitted tasks from every listed session",
    )
    download.add_argument("--train-sessions", nargs="+", type=_positive_int)
    download.add_argument("--test-sessions", nargs="+", type=_positive_int)
    download.add_argument("--localizer-sessions", nargs="+", type=_positive_int)
    download.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    download.add_argument("--space", choices=("mni", "fsaverage"), default="mni")
    download.add_argument("--backend", choices=("snapshot", "s3-latest"), default="snapshot")
    download.add_argument("--inventory")
    download.add_argument(
        "--inventory-in", help="reuse a previously frozen versioned inventory without GraphQL"
    )
    download.add_argument("--workers", type=_positive_int, default=4)
    download.add_argument("--dry-run", action="store_true")
    download.set_defaults(function=_download_bmd)

    stimulus_download = sub.add_parser(
        "download-stimuli", help="download the separately licensed stimulus archive"
    )
    stimulus_download.add_argument("--data-root", required=True)
    stimulus_download.add_argument("--output")
    stimulus_download.add_argument("--accept-stimulus-terms", action="store_true")
    stimulus_download.add_argument("--dry-run", action="store_true")
    stimulus_download.set_defaults(function=_download_stimuli)

    stimulus_install = sub.add_parser(
        "install-stimuli", help="selectively install train/test/localizer videos"
    )
    stimulus_install.add_argument("--data-root", required=True)
    source = stimulus_install.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive")
    source.add_argument("--source-dir")
    stimulus_install.add_argument("--password-env", default="BMD_STIMULI_PASSWORD")
    stimulus_install.add_argument("--link", action="store_true")
    stimulus_install.add_argument("--force", action="store_true")
    stimulus_install.add_argument("--accept-stimulus-terms", action="store_true")
    stimulus_install.add_argument("--dry-run", action="store_true")
    stimulus_install.set_defaults(function=_install_stimuli)

    validate = sub.add_parser("validate-data", help="validate the requested four-subject subset")
    validate.add_argument("--data-root", required=True)
    validate.add_argument("--subjects", nargs="+", type=_positive_int, default=[1, 2, 3, 4])
    validate.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    validate.add_argument("--space", choices=("mni", "fsaverage"), default="mni")
    validate.add_argument("--skip-videos", action="store_true")
    validate.add_argument("--deep", action="store_true", help="inspect image headers and confound rows")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(function=_validate_data)

    manifest = sub.add_parser("make-manifest", help="write a portable run-level JSONL manifest")
    manifest.add_argument("--data-root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--subjects", nargs="+", type=_positive_int, default=[1, 2, 3, 4])
    manifest.add_argument("--sessions", nargs="+", type=_positive_int, default=[1, 2, 3])
    manifest.add_argument("--space", choices=("mni", "fsaverage"), default="mni")
    manifest.set_defaults(function=_make_manifest)

    audit = sub.add_parser(
        "audit-pseudostudies", help="report A/B overlap and train/evaluation leakage"
    )
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--subjects", nargs="+", type=_positive_int, default=[1, 2, 3, 4])
    audit.add_argument("--sessions", nargs="+", type=_positive_int, default=[1, 2, 3])
    audit.add_argument("--space", choices=("mni", "fsaverage"), default="mni")
    audit.set_defaults(function=_audit_pseudostudies)

    atlas = sub.add_parser("prepare-atlas", help="compile offline Glasser-360 assets")
    atlas.add_argument("--output-dir", required=True)
    atlas.add_argument("--n-modes", type=_positive_int, default=16)
    atlas.add_argument("--accept-hcp-license", action="store_true")
    atlas.set_defaults(function=_prepare_atlas)

    nuisance = sub.add_parser("nuisance-preview", help="generate a small deterministic audit NPZ")
    nuisance.add_argument("--config", required=True)
    nuisance.add_argument("--output", required=True)
    nuisance.add_argument("--parcels", type=_positive_int, default=360)
    nuisance.set_defaults(function=_nuisance_preview)

    smoke = sub.add_parser("model-smoke", help="run the official model on cached synthetic features")
    smoke.add_argument("--config", required=True)
    smoke.set_defaults(function=_model_smoke)

    vjepa = sub.add_parser(
        "download-vjepa", help="fetch and verify the pinned V-JEPA2 inference snapshot"
    )
    vjepa.add_argument("--config", required=True)
    vjepa.add_argument("--cache-dir")
    vjepa.add_argument("--provenance")
    vjepa.add_argument("--dry-run", action="store_true")
    vjepa.set_defaults(function=_download_vjepa)

    evaluate = sub.add_parser(
        "evaluate-bundle", help="score exported empirical/predicted arrays without training code"
    )
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(function=_evaluate_bundle)

    train = sub.add_parser("train", help="run or inspect one preregistered matrix cell")
    train.add_argument("--config", required=True)
    train.add_argument("--condition", required=True)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(function=_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (ValueError, TypeError, PermissionError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
