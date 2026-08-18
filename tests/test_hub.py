from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from spfcl.model.hub import (
    VJEPA_FILES,
    download_vjepa_snapshot,
    validate_vjepa_snapshot,
)

REVISION = "1" * 40


def _snapshot(tmp_path: Path) -> tuple[Path, str]:
    snapshot = tmp_path / "hub" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for name in VJEPA_FILES:
        (snapshot / name).write_bytes((name + "\n").encode())
    digest = hashlib.sha256((snapshot / "model.safetensors").read_bytes()).hexdigest()
    return snapshot, digest


def test_download_is_revision_pinned_and_writes_verified_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, digest = _snapshot(tmp_path)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    provenance = tmp_path / "provenance.json"
    result = download_vjepa_snapshot(
        repo_id="facebook/vjepa2-vitg-fpc64-256",
        revision=REVISION,
        expected_model_sha256=digest,
        cache_dir=tmp_path / "cache",
        provenance_path=provenance,
    )
    assert result["revision"] == REVISION
    assert calls[0]["revision"] == REVISION
    assert set(calls[0]["allow_patterns"]) == set(VJEPA_FILES)
    assert json.loads(provenance.read_text())["files"]["model.safetensors"]["sha256"] == digest
    assert validate_vjepa_snapshot(
        provenance,
        repo_id="facebook/vjepa2-vitg-fpc64-256",
        revision=REVISION,
        expected_model_sha256=digest,
    ) == snapshot.resolve()


def test_download_dry_run_does_not_import_or_write(tmp_path: Path) -> None:
    provenance = tmp_path / "provenance.json"
    result = download_vjepa_snapshot(
        repo_id="facebook/vjepa2-vitg-fpc64-256",
        revision=REVISION,
        expected_model_sha256="2" * 64,
        cache_dir=tmp_path / "cache",
        provenance_path=provenance,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert not provenance.exists()


def test_validation_rejects_changed_snapshot_file(tmp_path: Path) -> None:
    snapshot, digest = _snapshot(tmp_path)
    provenance = tmp_path / "provenance.json"
    value = {
        "schema_version": 1,
        "repo_id": "facebook/vjepa2-vitg-fpc64-256",
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "files": {
            name: {"size": (snapshot / name).stat().st_size}
            for name in VJEPA_FILES
        },
    }
    value["files"]["model.safetensors"]["sha256"] = digest
    provenance.write_text(json.dumps(value))
    (snapshot / "config.json").write_bytes(b"changed-size")
    with pytest.raises(RuntimeError, match="changed"):
        validate_vjepa_snapshot(
            provenance,
            repo_id=value["repo_id"],
            revision=REVISION,
            expected_model_sha256=digest,
        )


def test_validation_rehashes_same_size_file_after_mtime_change(tmp_path: Path) -> None:
    snapshot, digest = _snapshot(tmp_path)
    files = {
        name: {
            "size": (snapshot / name).stat().st_size,
            "sha256": hashlib.sha256((snapshot / name).read_bytes()).hexdigest(),
        }
        for name in VJEPA_FILES
    }
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_id": "facebook/vjepa2-vitg-fpc64-256",
                "revision": REVISION,
                "snapshot_path": str(snapshot),
                "files": files,
            }
        )
    )
    validate_vjepa_snapshot(
        provenance,
        repo_id="facebook/vjepa2-vitg-fpc64-256",
        revision=REVISION,
        expected_model_sha256=digest,
    )
    config = snapshot / "config.json"
    original = config.stat()
    config.write_bytes(b"x" * original.st_size)
    os.utime(config, ns=(original.st_atime_ns, original.st_mtime_ns + 1))
    with pytest.raises(RuntimeError, match="hash changed"):
        validate_vjepa_snapshot(
            provenance,
            repo_id="facebook/vjepa2-vitg-fpc64-256",
            revision=REVISION,
            expected_model_sha256=digest,
        )


def test_registered_local_image_resolves_inside_video_extractor(tmp_path: Path) -> None:
    pytest.importorskip("neuralset")
    from neuralset.extractors.video import HuggingFaceVideo

    from spfcl.model.hub import PinnedLocalHuggingFaceImage

    if PinnedLocalHuggingFaceImage is None:
        pytest.skip("full neuralset dependency set is not installed")
    snapshot = tmp_path / "hf" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for name in VJEPA_FILES:
        (snapshot / name).touch()
    video = HuggingFaceVideo(
        frequency=2,
        event_types="Video",
        aggregation="sum",
        use_audio=False,
        image={
            "name": "PinnedLocalHuggingFaceImage",
            "model_name": str(snapshot),
            "infra": {"cluster": None, "folder": None, "keep_in_ram": False},
            "layers": [0.75, 1.0],
            "batch_size": 1,
        },
        clip_duration=4,
        infra={"cluster": None, "folder": None, "keep_in_ram": False},
        allow_missing=True,
    )
    assert isinstance(video.image, PinnedLocalHuggingFaceImage)
