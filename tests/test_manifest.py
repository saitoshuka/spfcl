from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from spfcl.data.manifest import (
    audit_pseudostudies,
    build_manifest,
    iter_run_records,
    validate_bmd_layout,
    write_manifest,
)


def _write_run(
    study_root: Path,
    *,
    subject: int,
    session: int,
    task: str,
    run: int,
    include_right: bool = True,
    include_confounds: bool = True,
) -> None:
    derivative = (
        study_root
        / "download"
        / "derivatives"
        / "versionB"
        / "fmriprep"
        / f"sub-{subject:02d}"
        / f"ses-{session:02d}"
        / "func"
    )
    derivative.mkdir(parents=True, exist_ok=True)
    prefix = f"sub-{subject:02d}_ses-{session:02d}_task-{task}_run-{run}"
    left = derivative / f"{prefix}_hemi-L_space-fsaverage_bold.func.gii"
    right = derivative / f"{prefix}_hemi-R_space-fsaverage_bold.func.gii"
    left.write_bytes(b"fake-left-gifti")
    if include_right:
        right.write_bytes(b"fake-right-gifti")
    if include_confounds:
        (derivative / f"{prefix}_desc-confounds_timeseries.tsv").write_text(
            "trans_x\n0\n", encoding="utf8"
        )

    raw = (
        study_root
        / "download"
        / f"sub-{subject:02d}"
        / f"ses-{session:02d}"
        / "func"
    )
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{prefix}_events.tsv").write_text(
        "onset\tduration\ttrial_type\tstim_file\n"
        f"0\t3\tvideo\t{task}/clip-{run:03d}.mp4\n"
        f"4\t3\toddball\t{task}/ignored-{run:03d}.mp4\n",
        encoding="utf8",
    )


def _write_mni_run(study_root: Path, *, subject: int, session: int, task: str, run: int) -> None:
    derivative = (
        study_root
        / "download"
        / "derivatives"
        / "versionB"
        / "fmriprep"
        / f"sub-{subject:02d}"
        / f"ses-{session:02d}"
        / "func"
    )
    derivative.mkdir(parents=True, exist_ok=True)
    prefix = f"sub-{subject:02d}_ses-{session:02d}_task-{task}_run-{run}"
    (derivative / f"{prefix}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz").write_bytes(
        b"fake-mni-bold"
    )
    (derivative / f"{prefix}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz").write_bytes(
        b"fake-mni-mask"
    )
    (derivative / f"{prefix}_desc-confounds_timeseries.tsv").write_text(
        "trans_x\n0\n", encoding="utf8"
    )
    raw = study_root / "download" / f"sub-{subject:02d}" / f"ses-{session:02d}" / "func"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{prefix}_events.tsv").write_text(
        "onset\tduration\ttrial_type\tstim_file\n"
        f"0\t3\tvideo\t{task}/clip-{run:03d}.mp4\n",
        encoding="utf8",
    )


def _write_complete_one_subject_layout(study_root: Path) -> None:
    for run in range(1, 6):
        _write_run(study_root, subject=1, session=1, task="localizer", run=run)
    for session in (2, 3):
        for run in range(1, 11):
            _write_run(study_root, subject=1, session=session, task="train", run=run)
        for run in range(1, 4):
            _write_run(study_root, subject=1, session=session, task="test", run=run)

    download = study_root / "download"
    (download / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "doi:10.18112/openneuro.ds005165.v1.0.4"}),
        encoding="utf8",
    )
    captions = download / "derivatives" / "stimuli_metadata"
    captions.mkdir(parents=True, exist_ok=True)
    (captions / "llm_frame_annotations.json").write_text("{}\n", encoding="utf8")


def test_manifest_assigns_pseudostudy_and_evaluation_roles(tmp_path: Path) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_run(study_root, subject=1, session=1, task="localizer", run=1)
    _write_run(study_root, subject=1, session=2, task="train", run=1)
    _write_run(study_root, subject=1, session=3, task="train", run=1)
    _write_run(study_root, subject=1, session=2, task="test", run=1)

    records = build_manifest(
        tmp_path, subjects=(1,), sessions=(1, 2, 3), space="fsaverage"
    )
    roles = {(item.session, item.task): item.role for item in records}

    assert roles == {
        (1, "localizer"): "probe",
        (2, "train"): "A",
        (3, "train"): "B",
        (2, "test"): "clean_test",
    }
    assert all(item.stimulus_count == 1 for item in records)
    assert all(item.space == "fsaverage" for item in records)
    assert all(item.left_bold is not None for item in records)
    assert all(item.right_bold is not None for item in records)
    assert all(not Path(item.left_bold).is_absolute() for item in records if item.left_bold)
    assert all(not Path(item.right_bold).is_absolute() for item in records if item.right_bold)
    assert all((study_root / item.left_bold).is_file() for item in records if item.left_bold)
    assert all((study_root / item.right_bold).is_file() for item in records if item.right_bold)


def test_mni_manifest_records_bold_and_mask_as_relative_paths(tmp_path: Path) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_mni_run(study_root, subject=1, session=2, task="train", run=1)
    record = build_manifest(tmp_path, subjects=(1,), sessions=(2,), space="mni")[0]

    assert record.space == "mni"
    assert record.bold is not None and record.mask is not None
    assert record.left_bold is None and record.right_bold is None
    assert not Path(record.bold).is_absolute()
    assert (study_root / record.bold).is_file()
    assert (study_root / record.mask).is_file()


def test_manifest_does_not_treat_localizer_fixation_na_as_a_stimulus(
    tmp_path: Path,
) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_run(study_root, subject=1, session=1, task="localizer", run=1)
    events = next(
        (study_root / "download" / "sub-01" / "ses-01" / "func").glob(
            "*_events.tsv"
        )
    )
    with events.open("a", encoding="utf8") as stream:
        stream.write("8\t18\tfix\tn/a\n")

    record = build_manifest(
        tmp_path, subjects=(1,), sessions=(1,), space="fsaverage"
    )[0]
    assert record.stimulus_count == 1


def test_complete_one_subject_layout_has_expected_run_counts(tmp_path: Path) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_complete_one_subject_layout(study_root)

    records = build_manifest(
        tmp_path, subjects=(1,), sessions=(1, 2, 3), space="fsaverage"
    )
    counts = Counter(item.role for item in records)

    assert len(records) == 31
    assert counts == {"A": 10, "B": 10, "clean_test": 6, "probe": 5}
    assert len({item.paired_key for item in records}) == 31

    report = audit_pseudostudies(tmp_path, records)
    assert report["a_b_overlap"] == 10
    assert report["train_test_overlap"] == 0
    assert report["train_probe_overlap"] == 0
    assert all(report["cross_subject_sets_identical"].values())


def test_layout_validator_accepts_complete_partial_subject_selection(tmp_path: Path) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_complete_one_subject_layout(study_root)

    issues = validate_bmd_layout(
        tmp_path,
        subjects=(1,),
        expected_snapshot="1.0.4",
        check_videos=False,
        space="fsaverage",
    )

    assert issues == []


def test_layout_validator_reports_stimuli_as_a_separate_readiness_error(
    tmp_path: Path,
) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_complete_one_subject_layout(study_root)

    issues = validate_bmd_layout(
        tmp_path,
        subjects=(1,),
        expected_snapshot="1.0.4",
        check_videos=True,
        space="fsaverage",
    )

    messages = [item.message for item in issues]
    assert any("Stimulus videos are absent" in message for message in messages)
    assert all(item.level == "error" for item in issues)


def test_subject_and_session_filters_are_applied(tmp_path: Path) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_run(study_root, subject=1, session=2, task="train", run=1)
    _write_run(study_root, subject=2, session=2, task="train", run=1)
    _write_run(study_root, subject=1, session=3, task="train", run=1)

    records = list(
        iter_run_records(tmp_path, subjects=(1,), sessions=(2,), space="fsaverage")
    )

    assert [(item.subject, item.session, item.task, item.run) for item in records] == [
        (1, 2, "train", 1)
    ]


def test_missing_right_hemisphere_is_a_hard_error(tmp_path: Path) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_run(
        study_root,
        subject=1,
        session=2,
        task="train",
        run=1,
        include_right=False,
    )

    with pytest.raises(FileNotFoundError, match="right-hemisphere pair"):
        build_manifest(tmp_path, subjects=(1,), sessions=(2,), space="fsaverage")


def test_missing_confounds_is_recorded_but_not_fabricated(tmp_path: Path) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_run(
        study_root,
        subject=1,
        session=2,
        task="train",
        run=1,
        include_confounds=False,
    )

    record = build_manifest(
        tmp_path, subjects=(1,), sessions=(2,), space="fsaverage"
    )[0]
    assert record.confounds is None


def test_write_manifest_is_jsonl_with_stable_sorted_keys(tmp_path: Path) -> None:
    study_root = tmp_path / "Lahner2024Bold"
    _write_run(study_root, subject=1, session=2, task="train", run=1)
    records = build_manifest(
        tmp_path, subjects=(1,), sessions=(2,), space="fsaverage"
    )

    first = write_manifest(records, tmp_path / "first.jsonl")
    second = write_manifest(records, tmp_path / "second.jsonl")

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf8"))
    assert payload["role"] == "A"
    assert payload["events_sha256"] == records[0].events_sha256
