from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .download import (
    DEFAULT_SNAPSHOT,
    FSAVERAGE_DERIVATIVE_PATTERNS,
    MNI_DERIVATIVE_PATTERNS,
    OPENNEURO_DATASET,
    ROOT_FILES,
    STIMULUS_METADATA_PATTERNS,
    DownloadSummary,
    _matches,
    _task_allowed,
)

GRAPHQL_ENDPOINT = "https://openneuro.org/crn/graphql"
_ANNEX_ID = re.compile(r"^SHA256E-s(?P<size>\d+)--(?P<sha256>[0-9a-f]{64})(?:\..*)?$")


@dataclass(frozen=True)
class SnapshotObject:
    path: str
    object_id: str
    size: int
    annexed: bool
    url: str

    @property
    def sha256(self) -> str | None:
        match = _ANNEX_ID.match(self.object_id)
        return match.group("sha256") if match else None


class OpenNeuroGraphQL:
    """Small client for immutable OpenNeuro snapshot inventories."""

    def __init__(self, endpoint: str = GRAPHQL_ENDPOINT, timeout: int = 120):
        self.endpoint = endpoint
        self.timeout = timeout

    def _execute(self, query: str) -> dict:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"query": query}).encode("utf8"),
            headers={"content-type": "application/json", "user-agent": "spfcl/0.1"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if payload.get("errors"):
            raise RuntimeError(f"OpenNeuro GraphQL error: {payload['errors']}")
        return payload["data"]

    @staticmethod
    def _quote(value: str) -> str:
        return json.dumps(value)

    def root(self, dataset: str, snapshot: str) -> list[dict]:
        query = (
            "query { snapshot(datasetId: "
            + self._quote(dataset)
            + ", tag: "
            + self._quote(snapshot)
            + ") { id tag files { id filename size directory annexed urls } } }"
        )
        value = self._execute(query)["snapshot"]
        if value is None or value.get("tag") != snapshot:
            raise RuntimeError(f"OpenNeuro snapshot not found: {dataset} {snapshot}")
        return value["files"]

    def tree(self, dataset: str, snapshot: str, tree_id: str, *, recursive: bool) -> list[dict]:
        query = (
            "query { snapshot(datasetId: "
            + self._quote(dataset)
            + ", tag: "
            + self._quote(snapshot)
            + ") { files(tree: "
            + self._quote(tree_id)
            + ", recursive: "
            + ("true" if recursive else "false")
            + ") { id filename size directory annexed urls } } }"
        )
        return self._execute(query)["snapshot"]["files"]


class OpenNeuroSnapshotDownloader:
    """Download selected files from an immutable, versionId-addressed snapshot."""

    def __init__(
        self,
        *,
        dataset: str = OPENNEURO_DATASET,
        snapshot: str = DEFAULT_SNAPSHOT,
        api: OpenNeuroGraphQL | None = None,
    ):
        self.dataset = dataset
        self.snapshot = snapshot
        self.api = api or OpenNeuroGraphQL()
        self._root: list[dict] | None = None

    def _root_files(self) -> list[dict]:
        if self._root is None:
            self._root = self.api.root(self.dataset, self.snapshot)
        return self._root

    @staticmethod
    def _directory(files: Sequence[dict], name: str) -> dict:
        matches = [item for item in files if item["directory"] and item["filename"] == name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one OpenNeuro directory {name!r}, got {len(matches)}")
        return matches[0]

    def _descend(self, start: Sequence[dict], parts: Sequence[str]) -> tuple[dict, list[dict]]:
        files = list(start)
        current: dict | None = None
        for part in parts:
            current = self._directory(files, part)
            files = self.api.tree(
                self.dataset, self.snapshot, current["id"], recursive=False
            )
        if current is None:
            raise ValueError("parts may not be empty")
        return current, files

    @staticmethod
    def _object(path: str, item: dict) -> SnapshotObject:
        urls = item.get("urls") or []
        if not urls:
            raise RuntimeError(f"OpenNeuro snapshot object has no download URL: {path}")
        return SnapshotObject(
            path=path,
            object_id=item["id"],
            size=int(item["size"]),
            annexed=bool(item["annexed"]),
            url=urls[0],
        )

    def select_bmd(
        self,
        subjects: Sequence[int],
        sessions: Sequence[int] | None = None,
        *,
        space: str = "mni",
        task_sessions: dict[str, set[int]] | None = None,
    ) -> list[SnapshotObject]:
        if space not in {"mni", "fsaverage"}:
            raise ValueError("space must be 'mni' or 'fsaverage'")
        derivative_patterns = (
            MNI_DERIVATIVE_PATTERNS if space == "mni" else FSAVERAGE_DERIVATIVE_PATTERNS
        )
        if task_sessions is not None:
            sessions = sorted(set().union(*task_sessions.values()))
        if not sessions:
            raise ValueError("At least one session must be selected")
        root = self._root_files()
        selected: dict[str, SnapshotObject] = {}
        for item in root:
            if not item["directory"] and item["filename"] in ROOT_FILES:
                selected[item["filename"]] = self._object(item["filename"], item)

        # Raw events: query only each selected subject branch.
        root_directories = {item["filename"]: item for item in root if item["directory"]}
        for subject in subjects:
            sub = f"sub-{subject:02d}"
            if sub not in root_directories:
                raise RuntimeError(f"Subject is absent from snapshot: {sub}")
            items = self.api.tree(
                self.dataset, self.snapshot, root_directories[sub]["id"], recursive=True
            )
            for item in items:
                relative = item["filename"]
                if item["directory"] or not relative.endswith(("_events.tsv", "_events.json")):
                    continue
                for session in sessions:
                    if relative.startswith(f"ses-{session:02d}/func/") and _task_allowed(
                        relative, session, task_sessions
                    ):
                        path = f"{sub}/{relative}"
                        selected[path] = self._object(path, item)
                        break

        # Version-B fMRIPrep: descend without enumerating unrelated derivatives.
        _, fmriprep_children = self._descend(root, ("derivatives", "versionB", "fmriprep"))
        fmriprep_subjects = {
            item["filename"]: item for item in fmriprep_children if item["directory"]
        }
        for subject in subjects:
            sub = f"sub-{subject:02d}"
            if sub not in fmriprep_subjects:
                raise RuntimeError(f"Version-B fMRIPrep is missing subject {sub}")
            items = self.api.tree(
                self.dataset, self.snapshot, fmriprep_subjects[sub]["id"], recursive=True
            )
            for item in items:
                relative = item["filename"]
                if item["directory"]:
                    continue
                name = relative.rsplit("/", 1)[-1]
                if not _matches(name, derivative_patterns):
                    continue
                for session in sessions:
                    if relative.startswith(f"ses-{session:02d}/func/") and _task_allowed(
                        relative, session, task_sessions
                    ):
                        path = f"derivatives/versionB/fmriprep/{sub}/{relative}"
                        selected[path] = self._object(path, item)
                        break

        # Only small JSON/TXT metadata files at the stimuli_metadata root.
        _, metadata_children = self._descend(root, ("derivatives", "stimuli_metadata"))
        for item in metadata_children:
            name = item["filename"]
            if not item["directory"] and "/" not in name and _matches(
                name, STIMULUS_METADATA_PATTERNS
            ):
                path = f"derivatives/stimuli_metadata/{name}"
                selected[path] = self._object(path, item)

        return sorted(selected.values(), key=lambda item: item.path)

    @staticmethod
    def _git_blob_sha1(path: Path, size: int) -> str:
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _valid(self, obj: SnapshotObject, path: Path) -> bool:
        if not path.is_file() or path.stat().st_size != obj.size:
            return False
        annex_sha = obj.sha256
        if annex_sha:
            return self._file_sha256(path) == annex_sha
        return self._git_blob_sha1(path, obj.size) == obj.object_id

    def _download_one(self, obj: SnapshotObject, destination: Path) -> bool:
        output = destination / obj.path
        output.parent.mkdir(parents=True, exist_ok=True)
        if self._valid(obj, output):
            return False

        partial = output.with_name(output.name + ".part")
        # A stale/partial object is recoverable. Range requests avoid retransferring
        # completed bytes when S3 honors them; a 200 response restarts safely.
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(obj.url, headers={"user-agent": "spfcl/0.1"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as exc:
            if offset and exc.code == 416:
                response = None
            else:
                raise
        if response is not None:
            mode = "ab" if offset and getattr(response, "status", None) == 206 else "wb"
            with response, partial.open(mode) as stream:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
        if not self._valid(obj, partial):
            actual_size = partial.stat().st_size if partial.exists() else 0
            raise OSError(
                f"Integrity check failed for {obj.path}; expected {obj.size} bytes, "
                f"found {actual_size}. The .part file was retained for inspection/resume."
            )
        os.replace(partial, output)
        return True

    @staticmethod
    def write_inventory(
        objects: Sequence[SnapshotObject],
        path: str | Path,
        *,
        metadata: dict | None = None,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        serialized = [asdict(item) for item in sorted(objects, key=lambda value: value.path)]
        payload: list[dict] | dict = serialized
        if metadata is not None:
            payload = {
                "schema_version": 1,
                "metadata": metadata,
                "objects": serialized,
            }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")
        return output

    @staticmethod
    def read_inventory_metadata(path: str | Path) -> dict | None:
        payload = json.loads(Path(path).read_text(encoding="utf8"))
        if isinstance(payload, list):
            return None
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise TypeError(f"Snapshot inventory envelope is invalid: {path}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError(f"Snapshot inventory metadata is invalid: {path}")
        return metadata

    @staticmethod
    def read_inventory(path: str | Path) -> list[SnapshotObject]:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf8"))
        if isinstance(payload, dict):
            if payload.get("schema_version") != 1 or not isinstance(
                payload.get("objects"), list
            ):
                raise TypeError(f"Snapshot inventory envelope is invalid: {source}")
            payload = payload["objects"]
        if not isinstance(payload, list):
            raise TypeError(f"Snapshot inventory must be a JSON list: {source}")
        objects = []
        for item in payload:
            obj = SnapshotObject(**item)
            candidate = Path(obj.path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Unsafe path in snapshot inventory: {obj.path}")
            if "versionId=" not in obj.url:
                raise ValueError(f"Inventory URL is not immutable/versioned: {obj.path}")
            objects.append(obj)
        if len({obj.path for obj in objects}) != len(objects):
            raise ValueError("Snapshot inventory contains duplicate paths")
        return sorted(objects, key=lambda item: item.path)

    def download(
        self,
        objects: Iterable[SnapshotObject],
        destination: str | Path,
        *,
        workers: int = 8,
        dry_run: bool = False,
        inventory_metadata: dict | None = None,
    ) -> DownloadSummary:
        destination = Path(destination)
        object_list = list(objects)
        if dry_run:
            for obj in object_list:
                print(f"GET {obj.url} -> {destination / obj.path} [{obj.size} bytes]")
            return DownloadSummary(
                selected=len(object_list),
                downloaded=0,
                skipped=0,
                bytes_selected=sum(item.size for item in object_list),
                destination=destination,
            )
        destination.mkdir(parents=True, exist_ok=True)
        self.write_inventory(
            object_list,
            destination / "spfcl_snapshot_inventory.json",
            metadata=inventory_metadata,
        )
        downloaded = skipped = 0
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
