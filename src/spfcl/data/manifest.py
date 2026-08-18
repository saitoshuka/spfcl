from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .download import DEFAULT_SNAPSHOT, bmd_study_root

_BIDS_RUN = re.compile(
    r"sub-(?P<subject>\d+)_ses-(?P<session>\d+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)"
)


@dataclass(frozen=True)
class RunRecord:
    subject: int
    session: int
    task: str
    run: int
    role: str
    events: str
    confounds: str | None
    events_sha256: str
    stimulus_count: int
    space: str
    bold: str | None = None
    mask: str | None = None
    left_bold: str | None = None
    right_bold: str | None = None

    @property
    def paired_key(self) -> str:
        return f"sub-{self.subject:02d}/ses-{self.session:02d}/{self.task}/run-{self.run:02d}"


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    message: str


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _glob_one(folder: Path, patterns: Iterable[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(folder.glob(pattern))
    unique = sorted(set(candidates))
    if not unique:
        return None
    if len(unique) > 1:
        raise RuntimeError(f"Ambiguous match in {folder}: {[p.name for p in unique]}")
    return unique[0]


def _stimulus_ids(events: Path) -> set[str]:
    with events.open("r", encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        output = set()
        for row in reader:
            if row.get("trial_type", "").lower() == "oddball":
                continue
            value = row.get("stim_file") or row.get("stimulus") or ""
            if value and value.lower() != "n/a":
                output.add(value)
        return output


def _role(session: int, task: str) -> str:
    if task == "localizer":
        return "probe"
    if task == "test":
        return "clean_test"
    if task == "train" and session == 2:
        return "A"
    if task == "train" and session == 3:
        return "B"
    return "unused"


def iter_run_records(
    data_root: str | Path,
    *,
    subjects: Sequence[int] = (1, 2, 3, 4),
    sessions: Sequence[int] = (1, 2, 3),
    space: str = "mni",
) -> Iterator[RunRecord]:
    if space not in {"mni", "fsaverage"}:
        raise ValueError("space must be 'mni' or 'fsaverage'")
    study_root = bmd_study_root(data_root)
    derivatives = study_root / "download" / "derivatives" / "versionB" / "fmriprep"
    subject_set = set(subjects)
    session_set = set(sessions)

    if space == "mni":
        primary_files = set(
            derivatives.glob(
                "sub-*/ses-*/func/*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            )
        )
    else:
        primary_files: set[Path] = set()
        for pattern in ("*_hemi-L_*bold.func.gii", "*_space-fsaverage_hemi-L_bold.func.gii"):
            primary_files.update(derivatives.glob(f"sub-*/ses-*/func/{pattern}"))

    for primary in sorted(primary_files):
        match = _BIDS_RUN.search(primary.name)
        if match is None:
            continue
        subject = int(match.group("subject"))
        session = int(match.group("session"))
        task = match.group("task").lower()
        run = int(match.group("run"))
        if subject not in subject_set or session not in session_set:
            continue
        if task not in {"train", "test", "localizer"}:
            continue

        bold: Path | None = None
        mask: Path | None = None
        left: Path | None = None
        right: Path | None = None
        if space == "mni":
            bold = primary
            mask = Path(
                str(primary).replace(
                    "_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz"
                )
            )
            if not mask.is_file():
                raise FileNotFoundError(f"Missing MNI brain-mask pair for {bold}: {mask}")
        else:
            left = primary
            right = Path(str(left).replace("hemi-L", "hemi-R"))
            if not right.is_file():
                raise FileNotFoundError(f"Missing right-hemisphere pair for {left}: {right}")

        raw_func = study_root / "download" / f"sub-{subject:02d}" / f"ses-{session:02d}" / "func"
        stem = f"sub-{subject:02d}_ses-{session:02d}_task-{task}_run-"
        events = _glob_one(
            raw_func,
            (
                f"{stem}{run}_events.tsv",
                f"{stem}{run:02d}_events.tsv",
            ),
        )
        if events is None:
            raise FileNotFoundError(f"Missing events TSV for {primary.name} in {raw_func}")

        derivative_func = primary.parent
        confounds = _glob_one(
            derivative_func,
            (
                f"{stem}{run}_desc-confounds_timeseries.tsv",
                f"{stem}{run:02d}_desc-confounds_timeseries.tsv",
                f"{stem}{run}_desc-confounds_regressors.tsv",
                f"{stem}{run:02d}_desc-confounds_regressors.tsv",
            ),
        )
        stimuli = _stimulus_ids(events)
        yield RunRecord(
            subject=subject,
            session=session,
            task=task,
            run=run,
            role=_role(session, task),
            events=str(events.relative_to(study_root)),
            confounds=str(confounds.relative_to(study_root)) if confounds else None,
            events_sha256=_sha256(events),
            stimulus_count=len(stimuli),
            space=space,
            bold=str(bold.relative_to(study_root)) if bold else None,
            mask=str(mask.relative_to(study_root)) if mask else None,
            left_bold=str(left.relative_to(study_root)) if left else None,
            right_bold=str(right.relative_to(study_root)) if right else None,
        )


def build_manifest(
    data_root: str | Path,
    *,
    subjects: Sequence[int] = (1, 2, 3, 4),
    sessions: Sequence[int] = (1, 2, 3),
    space: str = "mni",
) -> list[RunRecord]:
    records = list(
        iter_run_records(data_root, subjects=subjects, sessions=sessions, space=space)
    )
    keys = [(item.subject, item.session, item.task, item.run) for item in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Manifest contains duplicate BIDS run keys")
    return records


def write_manifest(records: Sequence[RunRecord], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return output_path


def audit_pseudostudies(
    data_root: str | Path,
    records: Sequence[RunRecord],
) -> dict:
    """Audit stimulus overlap and cross-subject consistency for A/B/test/probe."""

    root = bmd_study_root(data_root)
    by_subject_role: dict[tuple[int, str], set[str]] = {}
    for record in records:
        by_subject_role.setdefault((record.subject, record.role), set()).update(
            _stimulus_ids(root / record.events)
        )
    subjects = sorted({record.subject for record in records})
    roles = ("A", "B", "clean_test", "probe")
    consistency = {}
    for role in roles:
        sets = [by_subject_role.get((subject, role), set()) for subject in subjects]
        consistency[role] = len({frozenset(value) for value in sets}) <= 1
    a = set().union(*(by_subject_role.get((subject, "A"), set()) for subject in subjects))
    b = set().union(*(by_subject_role.get((subject, "B"), set()) for subject in subjects))
    test = set().union(
        *(by_subject_role.get((subject, "clean_test"), set()) for subject in subjects)
    )
    probe = set().union(
        *(by_subject_role.get((subject, "probe"), set()) for subject in subjects)
    )
    union = a | b
    return {
        "subjects": subjects,
        "stimulus_count": {
            "A": len(a),
            "B": len(b),
            "clean_test": len(test),
            "probe": len(probe),
        },
        "cross_subject_sets_identical": consistency,
        "a_b_overlap": len(a & b),
        "a_b_union": len(union),
        "a_b_jaccard": len(a & b) / len(union) if union else float("nan"),
        "train_test_overlap": len(union & test),
        "train_probe_overlap": len(union & probe),
    }


def validate_bmd_layout(
    data_root: str | Path,
    *,
    subjects: Sequence[int] = (1, 2, 3, 4),
    expected_snapshot: str = DEFAULT_SNAPSHOT,
    check_videos: bool = True,
    space: str = "mni",
    deep: bool = False,
) -> list[ValidationIssue]:
    root = bmd_study_root(data_root)
    issues: list[ValidationIssue] = []

    description = root / "download" / "dataset_description.json"
    if not description.is_file():
        issues.append(ValidationIssue("error", f"Missing {description}"))
    else:
        payload = json.loads(description.read_text(encoding="utf8"))
        doi = str(payload.get("DatasetDOI", ""))
        if expected_snapshot and f".v{expected_snapshot}" not in doi:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Expected OpenNeuro snapshot {expected_snapshot}, dataset DOI is {doi!r}",
                )
            )

    captions = root / "download" / "derivatives" / "stimuli_metadata" / "llm_frame_annotations.json"
    if not captions.is_file():
        issues.append(ValidationIssue("error", f"Missing TRIBE-required metadata: {captions}"))

    partials = list((root / "download").rglob("*.part")) if (root / "download").exists() else []
    if partials:
        issues.append(
            ValidationIssue(
                "error",
                f"Found {len(partials)} incomplete .part downloads; first: {partials[0]}",
            )
        )

    try:
        records = build_manifest(root, subjects=subjects, space=space)
    except (FileNotFoundError, RuntimeError) as exc:
        issues.append(ValidationIssue("error", str(exc)))
        records = []

    expected = {
        "A": 10 * len(subjects),
        "B": 10 * len(subjects),
        "clean_test": 6 * len(subjects),
        "probe": 5 * len(subjects),
    }
    for role, count in expected.items():
        actual = sum(item.role == role for item in records)
        if actual != count:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Expected {count} {role} runs for subjects {list(subjects)}, found {actual}",
                )
            )

    missing_confounds = [item.paired_key for item in records if item.confounds is None]
    if missing_confounds:
        issues.append(
            ValidationIssue(
                "warning",
                f"Missing confounds for {len(missing_confounds)} runs; first: {missing_confounds[0]}",
            )
        )

    if check_videos:
        stimulus_root = root / "stimuli" / "stimulus_set" / "stimuli"
        if any(not (stimulus_root / name).is_dir() for name in ("train", "test", "localizer")):
            issues.append(
                ValidationIssue(
                    "error",
                    "Stimulus videos are absent. Run `spfcl install-stimuli` after accepting "
                    "the BOLD Moments stimulus terms.",
                )
            )
        elif records:
            missing = []
            for item in records:
                with (root / item.events).open("r", encoding="utf8", newline="") as stream:
                    for row in csv.DictReader(stream, delimiter="\t"):
                        if row.get("trial_type", "").lower() == "oddball":
                            continue
                        stim = row.get("stim_file")
                        if stim and stim.lower() != "n/a" and not (stimulus_root / stim).is_file():
                            missing.append(stim)
                            if len(missing) >= 5:
                                break
                if len(missing) >= 5:
                    break
            if missing:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"Stimulus archive is incomplete; missing examples: {sorted(set(missing))}",
                    )
                )
    if deep and records:
        try:
            import nibabel as nib
        except ImportError:
            issues.append(
                ValidationIssue("error", "Deep validation requires nibabel (install spfcl[data])")
            )
        else:
            for item in records:
                try:
                    if item.space == "mni":
                        if item.bold is None or item.mask is None:
                            raise RuntimeError("MNI record lacks bold/mask paths")
                        bold = nib.load(root / item.bold)
                        mask = nib.load(root / item.mask)
                        if tuple(mask.shape) != tuple(bold.shape[:3]):
                            raise RuntimeError(
                                f"bold/mask spatial shape mismatch {bold.shape} vs {mask.shape}"
                            )
                        n_volumes = int(bold.shape[-1])
                    else:
                        if item.left_bold is None or item.right_bold is None:
                            raise RuntimeError("surface record lacks hemisphere paths")
                        left = nib.load(root / item.left_bold)
                        right = nib.load(root / item.right_bold)
                        if len(left.darrays) != len(right.darrays):  # type: ignore[attr-defined]
                            raise RuntimeError("left/right surface time dimensions differ")
                        if any(
                            int(image.darrays[0].data.shape[0]) != 163_842  # type: ignore[attr-defined]
                            for image in (left, right)
                        ):
                            raise RuntimeError("fsaverage surface does not contain 163842 vertices")
                        n_volumes = len(left.darrays)  # type: ignore[attr-defined]
                    expected_volumes = 238 if item.task == "train" else 268 if item.task == "test" else None
                    if expected_volumes is not None and n_volumes != expected_volumes:
                        raise RuntimeError(
                            f"expected {expected_volumes} volumes, found {n_volumes}"
                        )
                    if item.confounds is not None:
                        with (root / item.confounds).open("r", encoding="utf8") as stream:
                            rows = max(0, sum(1 for _ in stream) - 1)
                        if rows != n_volumes:
                            raise RuntimeError(
                                f"confounds has {rows} rows but BOLD has {n_volumes} volumes"
                            )
                # Deep validation crosses nibabel/pandas backends whose parse
                # errors are not normalized; report every per-run failure and
                # continue auditing the remaining immutable inventory.
                except Exception as exc:  # noqa: BLE001
                    issues.append(
                        ValidationIssue("error", f"Deep check failed for {item.paired_key}: {exc}")
                    )
    return issues
