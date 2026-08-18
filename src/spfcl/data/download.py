from __future__ import annotations

import concurrent.futures
import fnmatch
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

OPENNEURO_BUCKET = "openneuro.org"
OPENNEURO_DATASET = "ds005165"
DEFAULT_SNAPSHOT = "1.0.4"

ROOT_FILES = (
    "CHANGES",
    "README",
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "task-localizer_events.json",
    "task-test_events.json",
    "task-train_events.json",
)

STIMULUS_METADATA_PATTERNS = (
    "*.json",
    "*.txt",
    "*.tsv",
)

FSAVERAGE_DERIVATIVE_PATTERNS = (
    "*_space-fsaverage_*_bold.func.gii",
    "*_space-fsaverage_bold.func.gii",
    "*_desc-confounds_timeseries.tsv",
    "*_desc-confounds_regressors.tsv",
)

MNI_DERIVATIVE_PATTERNS = (
    "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    "*_desc-confounds_timeseries.tsv",
    "*_desc-confounds_regressors.tsv",
)

# Backward-compatible alias for callers that explicitly use the surface route.
FMRI_DERIVATIVE_PATTERNS = FSAVERAGE_DERIVATIVE_PATTERNS


@dataclass(frozen=True)
class RemoteObject:
    key: str
    size: int


@dataclass(frozen=True)
class DownloadSummary:
    selected: int
    downloaded: int
    skipped: int
    bytes_selected: int
    destination: Path


def _public_s3_client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - exercised by remote doctor
        raise RuntimeError(
            "BOLD Moments download requires boto3. Run scripts/bootstrap_remote.sh "
            "or install spfcl[data]."
        ) from exc
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _task_allowed(
    key: str,
    session: int,
    task_sessions: dict[str, set[int]] | None = None,
) -> bool:
    if task_sessions is not None:
        return any(
            session in allowed_sessions and f"_task-{task}_" in key
            for task, allowed_sessions in task_sessions.items()
        )
    # Subject 4 localizer run 5 was acquired with session-2 field maps; retaining
    # localizer matches avoids silently dropping that documented exception.
    if session == 1:
        return "_task-localizer_" in key
    return any(token in key for token in ("_task-train_", "_task-test_", "_task-localizer_"))


class OpenNeuroSelectiveDownloader:
    """Anonymous, prefix-filtered downloader for OpenNeuro's public S3 mirror."""

    def __init__(self, client=None, bucket: str = OPENNEURO_BUCKET):
        self.client = client or _public_s3_client()
        self.bucket = bucket

    def list_prefix(self, prefix: str) -> Iterator[RemoteObject]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith("/"):
                    yield RemoteObject(key=key, size=int(item.get("Size", 0)))

    def select_bmd(
        self,
        subjects: Sequence[int],
        sessions: Sequence[int] | None = None,
        *,
        space: str = "mni",
        task_sessions: dict[str, set[int]] | None = None,
    ) -> list[RemoteObject]:
        if space not in {"mni", "fsaverage"}:
            raise ValueError("space must be 'mni' or 'fsaverage'")
        derivative_patterns = (
            MNI_DERIVATIVE_PATTERNS if space == "mni" else FSAVERAGE_DERIVATIVE_PATTERNS
        )
        if task_sessions is not None:
            sessions = sorted(set().union(*task_sessions.values()))
        if not sessions:
            raise ValueError("At least one session must be selected")
        root = f"{OPENNEURO_DATASET}/"
        selected: dict[str, RemoteObject] = {}

        for filename in ROOT_FILES:
            key = root + filename
            try:
                head = self.client.head_object(Bucket=self.bucket, Key=key)
            except Exception as exc:
                raise RuntimeError(f"OpenNeuro object is unavailable: s3://{self.bucket}/{key}") from exc
            selected[key] = RemoteObject(key, int(head.get("ContentLength", 0)))

        metadata_prefix = root + "derivatives/stimuli_metadata/"
        for obj in self.list_prefix(metadata_prefix):
            relative = obj.key[len(metadata_prefix) :]
            if "/" not in relative and _matches(relative, STIMULUS_METADATA_PATTERNS):
                selected[obj.key] = obj

        for subject in subjects:
            for session in sessions:
                sub = f"sub-{subject:02d}"
                ses = f"ses-{session:02d}"

                raw_prefix = f"{root}{sub}/{ses}/func/"
                for obj in self.list_prefix(raw_prefix):
                    name = obj.key.rsplit("/", 1)[-1]
                    if _task_allowed(obj.key, session, task_sessions) and name.endswith(("_events.tsv", "_events.json")):
                        selected[obj.key] = obj

                derivative_prefix = (
                    f"{root}derivatives/versionB/fmriprep/{sub}/{ses}/func/"
                )
                for obj in self.list_prefix(derivative_prefix):
                    name = obj.key.rsplit("/", 1)[-1]
                    if _task_allowed(obj.key, session, task_sessions) and _matches(name, derivative_patterns):
                        selected[obj.key] = obj

        return sorted(selected.values(), key=lambda item: item.key)

    def _download_one(self, obj: RemoteObject, destination: Path) -> bool:
        relative = Path(obj.key).relative_to(OPENNEURO_DATASET)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.is_file() and output.stat().st_size == obj.size:
            return False
        partial = output.with_name(output.name + ".part")
        self.client.download_file(self.bucket, obj.key, os.fspath(partial))
        if partial.stat().st_size != obj.size:
            raise OSError(
                f"Size mismatch for {obj.key}: expected {obj.size}, got {partial.stat().st_size}"
            )
        os.replace(partial, output)
        return True

    def download(
        self,
        objects: Iterable[RemoteObject],
        destination: str | Path,
        *,
        workers: int = 8,
        dry_run: bool = False,
    ) -> DownloadSummary:
        destination = Path(destination)
        object_list = list(objects)
        if dry_run:
            for obj in object_list:
                relative = Path(obj.key).relative_to(OPENNEURO_DATASET)
                print(f"GET s3://{self.bucket}/{obj.key} -> {destination / relative}")
            return DownloadSummary(
                selected=len(object_list),
                downloaded=0,
                skipped=0,
                bytes_selected=sum(item.size for item in object_list),
                destination=destination,
            )

        destination.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        skipped = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(self._download_one, obj, destination) for obj in object_list]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    downloaded += 1
                else:
                    skipped += 1
        return DownloadSummary(
            selected=len(object_list),
            downloaded=downloaded,
            skipped=skipped,
            bytes_selected=sum(item.size for item in object_list),
            destination=destination,
        )


def bmd_study_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    return root if root.name.lower() == "lahner2024bold" else root / "Lahner2024Bold"


def write_download_provenance(
    study_root: Path,
    *,
    subjects: Sequence[int],
    sessions: Sequence[int],
    snapshot: str,
    summary: DownloadSummary,
    backend: str = "snapshot",
    space: str = "mni",
    task_sessions: dict[str, set[int]] | None = None,
) -> Path:
    provenance = {
        "dataset": OPENNEURO_DATASET,
        "expected_snapshot": snapshot,
        "subjects": list(subjects),
        "sessions": list(sessions),
        "space": space,
        "backend": backend,
        "task_sessions": (
            {key: sorted(value) for key, value in task_sessions.items()}
            if task_sessions is not None
            else None
        ),
        "selected_objects": summary.selected,
        "bytes_selected": summary.bytes_selected,
        "note": (
            "snapshot uses immutable OpenNeuro versioned object URLs and verifies object IDs; "
            "s3-latest is an explicitly non-pinned convenience backend."
        ),
    }
    output = study_root / "download" / "spfcl_download.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return output
