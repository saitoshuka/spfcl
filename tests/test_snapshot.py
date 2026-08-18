from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spfcl.data.download import _task_allowed
from spfcl.data.snapshot import OpenNeuroSnapshotDownloader, SnapshotObject


def test_task_session_filter_keeps_only_preregistered_assets() -> None:
    mapping = {"train": {2, 3}, "test": {2, 3, 4, 5}, "localizer": {1}}
    assert _task_allowed("sub-01_ses-02_task-train_run-1", 2, mapping)
    assert not _task_allowed("sub-01_ses-04_task-train_run-1", 4, mapping)
    assert _task_allowed("sub-01_ses-04_task-test_run-1", 4, mapping)
    assert _task_allowed("sub-01_ses-01_task-localizer_run-1", 1, mapping)
    assert not _task_allowed("sub-01_ses-02_task-localizer_run-1", 2, mapping)


def test_snapshot_integrity_supports_annex_sha256_and_git_blob(tmp_path: Path) -> None:
    content = b"pinned snapshot payload"
    path = tmp_path / "object"
    path.write_bytes(content)
    annex = SnapshotObject(
        path="object",
        object_id=f"SHA256E-s{len(content)}--{hashlib.sha256(content).hexdigest()}.nii.gz",
        size=len(content),
        annexed=True,
        url="https://example.invalid/object?versionId=frozen",
    )
    git_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    regular = SnapshotObject(
        path="object",
        object_id=git_sha,
        size=len(content),
        annexed=False,
        url="https://example.invalid/object?versionId=frozen",
    )
    downloader = OpenNeuroSnapshotDownloader()
    assert downloader._valid(annex, path)
    assert downloader._valid(regular, path)
    path.write_bytes(content + b"corrupt")
    assert not downloader._valid(annex, path)
    assert not downloader._valid(regular, path)


def test_inventory_round_trip_and_unsafe_path_rejection(tmp_path: Path) -> None:
    item = SnapshotObject(
        path="sub-01/ses-01/func/events.tsv",
        object_id="0" * 40,
        size=0,
        annexed=False,
        url="https://s3.example/object?versionId=frozen",
    )
    path = OpenNeuroSnapshotDownloader.write_inventory([item], tmp_path / "inventory.json")
    assert OpenNeuroSnapshotDownloader.read_inventory(path) == [item]

    payload = [item.__dict__ | {"path": "../escape"}]
    path.write_text(json.dumps(payload), encoding="utf8")
    with pytest.raises(ValueError, match="Unsafe path"):
        OpenNeuroSnapshotDownloader.read_inventory(path)


def test_inventory_envelope_records_selection_contract(tmp_path: Path) -> None:
    item = SnapshotObject(
        path="sub-01/events.tsv",
        object_id="0" * 40,
        size=0,
        annexed=False,
        url="https://s3.example/object?versionId=frozen",
    )
    metadata = {
        "dataset": "ds005165",
        "snapshot": "1.0.4",
        "subjects": [1],
        "sessions": [1],
        "task_sessions": {"localizer": [1]},
        "space": "mni",
    }
    path = OpenNeuroSnapshotDownloader.write_inventory(
        [item], tmp_path / "inventory.json", metadata=metadata
    )
    assert OpenNeuroSnapshotDownloader.read_inventory_metadata(path) == metadata
    assert OpenNeuroSnapshotDownloader.read_inventory(path) == [item]


def test_inventory_rejects_nonversioned_latest_url(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            [
                {
                    "path": "README",
                    "object_id": "0" * 40,
                    "size": 0,
                    "annexed": False,
                    "url": "https://s3.example/latest/README",
                }
            ]
        ),
        encoding="utf8",
    )
    with pytest.raises(ValueError, match="not immutable"):
        OpenNeuroSnapshotDownloader.read_inventory(path)
