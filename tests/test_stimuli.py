from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from spfcl.data.stimuli import download_stimulus_archive, install_stimuli


def _archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("stimulus_set/stimuli/train/a.mp4", b"train")
        bundle.writestr("stimulus_set/stimuli/test/b.mp4", b"test")
        bundle.writestr("stimulus_set/stimuli/localizer/c.mp4", b"localizer")
        bundle.writestr("stimulus_set/frames/train/a/0001.jpg", b"large-unused-frame")
    return path


def test_terms_must_be_explicitly_accepted(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="access terms"):
        install_stimuli(tmp_path, archive=_archive(tmp_path / "stimuli.zip"))


def test_archive_install_extracts_videos_but_not_frames(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "stimuli.zip")
    target = install_stimuli(
        tmp_path,
        archive=archive,
        accept_stimulus_terms=True,
    )
    assert (target / "stimuli" / "train" / "a.mp4").read_bytes() == b"train"
    assert (target / "stimuli" / "test" / "b.mp4").is_file()
    assert (target / "stimuli" / "localizer" / "c.mp4").is_file()
    assert not list(target.rglob("*.jpg"))


def test_download_dry_run_does_not_touch_network_or_disk(tmp_path: Path) -> None:
    output = tmp_path / "stimulus_set.zip"
    returned = download_stimulus_archive(
        output,
        accept_stimulus_terms=True,
        dry_run=True,
    )
    assert returned == output.resolve()
    assert not output.exists()

