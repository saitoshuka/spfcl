from __future__ import annotations

"""Pinned Hugging Face snapshot acquisition for the frozen V-JEPA2 encoder."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

VJEPA_FILES = ("config.json", "video_preprocessor_config.json", "model.safetensors")


@lru_cache(maxsize=16)
def _sha256_cached(path: Path, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # They are cache-key invalidators, not hash inputs.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(path, stat.st_size, stat.st_mtime_ns)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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


def download_vjepa_snapshot(
    *,
    repo_id: str,
    revision: str,
    expected_model_sha256: str,
    cache_dir: str | Path,
    provenance_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download only inference files at an immutable Hub commit and hash the weights."""

    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("V-JEPA revision must be a full lowercase 40-character Git SHA")
    if len(expected_model_sha256) != 64:
        raise ValueError("V-JEPA model SHA-256 must contain 64 hexadecimal characters")
    cache = Path(cache_dir).expanduser().resolve()
    provenance = Path(provenance_path).expanduser().resolve()
    if dry_run:
        return {
            "repo_id": repo_id,
            "revision": revision,
            "allow_patterns": list(VJEPA_FILES),
            "cache_dir": str(cache),
            "provenance": str(provenance),
            "dry_run": True,
        }

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - remote environment only
        raise RuntimeError("Run scripts/bootstrap_remote.sh before downloading V-JEPA") from exc

    cache.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache,
            allow_patterns=list(VJEPA_FILES),
        )
    ).resolve()
    if snapshot.name != revision:
        raise RuntimeError(f"Hub resolved {revision} to an unexpected snapshot: {snapshot}")
    missing = [name for name in VJEPA_FILES if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"Pinned V-JEPA snapshot is incomplete: {missing}")
    model = snapshot / "model.safetensors"
    actual_sha256 = _sha256(model)
    if actual_sha256 != expected_model_sha256:
        raise RuntimeError(
            "Pinned V-JEPA weight hash mismatch: "
            f"expected={expected_model_sha256}, actual={actual_sha256}"
        )
    files = {
        name: {
            "size": (snapshot / name).stat().st_size,
            "sha256": _sha256(snapshot / name),
        }
        for name in VJEPA_FILES
    }
    value = {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "files": files,
    }
    _write_json_atomic(provenance, value)
    return {**value, "provenance": str(provenance), "dry_run": False}


def validate_vjepa_snapshot(
    provenance_path: str | Path,
    *,
    repo_id: str,
    revision: str,
    expected_model_sha256: str,
) -> Path:
    """Resolve a previously verified local snapshot without contacting the Hub."""

    provenance = Path(provenance_path).expanduser().resolve()
    try:
        value = json.loads(provenance.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Pinned V-JEPA provenance is missing or invalid: {provenance}; "
            "run `spfcl download-vjepa` on the server"
        ) from exc
    expected = {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
    }
    mismatch = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    model_record = value.get("files", {}).get("model.safetensors", {})
    if model_record.get("sha256") != expected_model_sha256:
        mismatch["model_sha256"] = (
            model_record.get("sha256"),
            expected_model_sha256,
        )
    if mismatch:
        raise RuntimeError(f"Pinned V-JEPA provenance differs from config: {mismatch}")
    snapshot = Path(value.get("snapshot_path", "")).expanduser().resolve()
    if snapshot.name != revision:
        raise RuntimeError(f"Pinned V-JEPA snapshot path has the wrong revision: {snapshot}")
    for name in VJEPA_FILES:
        path = snapshot / name
        record = value.get("files", {}).get(name, {})
        expected_size = record.get("size")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Pinned V-JEPA file is missing or changed: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != record.get("sha256"):
            raise RuntimeError(f"Pinned V-JEPA file hash changed: {path}")
    return snapshot


# Neuralset 0.0.2 validates every model_name as a Hub repository ID.  Our
# immutable snapshot is intentionally a local directory, so register a narrow
# subclass that accepts only a complete, already-verified inference snapshot.
try:  # pragma: no cover - exercised with the full remote dependency set
    from neuralset.extractors.image import HuggingFaceImage

    class PinnedLocalHuggingFaceImage(HuggingFaceImage):
        def repo_exists(self) -> bool:
            root = Path(self.model_name)
            return root.is_dir() and all((root / name).is_file() for name in VJEPA_FILES)

except ImportError:
    PinnedLocalHuggingFaceImage = None  # type: ignore[assignment]
