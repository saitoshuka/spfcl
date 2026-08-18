from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .download import bmd_study_root

STIMULUS_ARCHIVE_URL = (
    "https://boldmomentsdataset.csail.mit.edu/stimuli_metadata/stimulus_set.zip"
)
STIMULUS_ARCHIVE_SIZE = 3_261_964_337


def _require_terms(accepted: bool) -> None:
    if not accepted:
        raise PermissionError(
            "BOLD Moments stimuli have separate access terms. Read the official portal and "
            "rerun with --accept-stimulus-terms. Do not redistribute the archive/videos."
        )


def _is_video_member(member: str) -> bool:
    parts = Path(member).parts
    for index in range(max(0, len(parts) - 2)):
        if parts[index : index + 2] == ("stimulus_set", "stimuli"):
            return len(parts) > index + 2 and parts[index + 2] in {
                "train",
                "test",
                "localizer",
            }
    return False


def _safe_destination(root: Path, member: str) -> Path:
    destination = (root / member).resolve()
    if destination != root.resolve() and root.resolve() not in destination.parents:
        raise RuntimeError(f"Archive contains an unsafe path: {member}")
    return destination


def _extract_archive(archive: Path, destination: Path, password: str | None) -> None:
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                _safe_destination(destination, info.filename)
                unix_mode = info.external_attr >> 16
                if unix_mode & 0o170000 == 0o120000:
                    raise RuntimeError("Stimulus archive may not contain symbolic links")
            pwd = password.encode("utf8") if password else None
            for info in bundle.infolist():
                if not _is_video_member(info.filename):
                    continue
                target = _safe_destination(destination, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info, pwd=pwd) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        return
    if suffixes.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(archive, mode="r:*") as bundle:
            for member in bundle.getmembers():
                _safe_destination(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError("Stimulus archive may not contain symbolic or hard links")
            for member in bundle.getmembers():
                if _is_video_member(member.name):
                    bundle.extract(member, destination)
        return
    raise ValueError(f"Unsupported stimulus archive: {archive}")


def _find_stimulus_set(root: Path) -> Path:
    candidates: list[Path] = []
    for path in [root, *root.rglob("stimulus_set")]:
        if path.is_dir() and (path / "stimuli" / "train").is_dir() and (
            path / "stimuli" / "test"
        ).is_dir():
            candidates.append(path)
    if not candidates:
        for stimuli in root.rglob("stimuli"):
            if (stimuli / "train").is_dir() and (stimuli / "test").is_dir():
                return stimuli.parent
        raise RuntimeError(
            "Could not find stimulus_set/stimuli/{train,test} in the supplied archive/source"
        )
    return min(candidates, key=lambda path: len(path.parts))


def install_stimuli(
    data_root: str | Path,
    *,
    archive: str | Path | None = None,
    source_dir: str | Path | None = None,
    password_env: str = "BMD_STIMULI_PASSWORD",
    link: bool = False,
    force: bool = False,
    dry_run: bool = False,
    accept_stimulus_terms: bool = False,
) -> Path:
    _require_terms(accept_stimulus_terms)
    if (archive is None) == (source_dir is None):
        raise ValueError("Provide exactly one of archive or source_dir")

    study_root = bmd_study_root(data_root)
    target = study_root / "stimuli" / "stimulus_set"
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(f"Refusing to overwrite non-empty stimulus directory: {target}")

    if dry_run:
        source = archive if archive is not None else source_dir
        print(f"INSTALL {source} -> {target} (link={link}, force={force})")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        source = _find_stimulus_set(Path(source_dir).expanduser().resolve())
        if link:
            if target.exists():
                raise FileExistsError(f"Cannot create symlink because target exists: {target}")
            target.symlink_to(source, target_is_directory=True)
        else:
            shutil.copytree(source, target, dirs_exist_ok=force)
        return target

    archive_path = Path(archive).expanduser().resolve()  # type: ignore[arg-type]
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    password = os.getenv(password_env) or None
    with tempfile.TemporaryDirectory(prefix="bmd-stimuli-", dir=target.parent) as tmp:
        temporary = Path(tmp)
        _extract_archive(archive_path, temporary, password)
        source = _find_stimulus_set(temporary)
        shutil.copytree(source, target, dirs_exist_ok=force)
    return target


def download_stimulus_archive(
    output: str | Path,
    *,
    accept_stimulus_terms: bool = False,
    url: str = STIMULUS_ARCHIVE_URL,
    expected_size: int = STIMULUS_ARCHIVE_SIZE,
    dry_run: bool = False,
) -> Path:
    """Download the separately licensed archive without logging its password."""

    _require_terms(accept_stimulus_terms)
    destination = Path(output).expanduser().resolve()
    if dry_run:
        print(f"GET {url} -> {destination} ({expected_size} bytes)")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        return destination
    partial = destination.with_name(destination.name + ".part")
    if partial.is_file() and partial.stat().st_size == expected_size:
        os.replace(partial, destination)
        return destination
    if partial.is_file() and partial.stat().st_size > expected_size:
        raise OSError(
            f"Partial archive is larger than expected ({partial.stat().st_size} > "
            f"{expected_size}): {partial}"
        )
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"user-agent": "spfcl/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=300) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        with partial.open("ab" if append else "wb") as stream:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    if partial.stat().st_size != expected_size:
        raise OSError(
            f"Stimulus archive size mismatch: expected {expected_size}, "
            f"got {partial.stat().st_size}; retained {partial} for resume."
        )
    os.replace(partial, destination)
    return destination
