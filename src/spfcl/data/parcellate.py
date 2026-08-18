from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FSAVERAGE5_VERTICES_PER_HEMISPHERE = 10_242
N_GLASSER_PER_HEMISPHERE = 180
N_GLASSER_TOTAL = 360


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParcelOperator:
    """Dense row-normalized vertex-to-parcel operator for fsaverage5."""

    matrix: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)
        if matrix.shape != (N_GLASSER_TOTAL, 2 * FSAVERAGE5_VERTICES_PER_HEMISPHERE):
            raise ValueError(
                "Expected Glasser operator shape "
                f"(360, 20484), got {matrix.shape}"
            )
        if len(self.names) != N_GLASSER_TOTAL:
            raise ValueError(f"Expected 360 parcel names, got {len(self.names)}")
        if np.any(matrix < 0):
            raise ValueError("Parcel weights must be non-negative")
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Every parcel row must sum to one")

    def transform(self, data: np.ndarray, vertex_axis: int = -2) -> np.ndarray:
        value = np.asarray(data)
        if value.shape[vertex_axis] != self.matrix.shape[1]:
            raise ValueError(
                f"Expected {self.matrix.shape[1]} vertices on axis {vertex_axis}, "
                f"got {value.shape[vertex_axis]}"
            )
        moved = np.moveaxis(value, vertex_axis, -2)
        output = np.einsum("pv,...vt->...pt", self.matrix, moved, optimize=True)
        return np.moveaxis(output, -2, vertex_axis)

    def save(self, output: str | Path, metadata: dict | None = None) -> Path:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            matrix=self.matrix.astype(np.float32),
            names=np.asarray(self.names),
            metadata=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> ParcelOperator:
        with np.load(path, allow_pickle=False) as bundle:
            return cls(
                matrix=np.asarray(bundle["matrix"], dtype=np.float32),
                names=tuple(str(item) for item in bundle["names"].tolist()),
            )


def _labels_to_operator(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_names: Sequence[str],
    right_names: Sequence[str],
) -> ParcelOperator:
    if left_labels.shape != (FSAVERAGE5_VERTICES_PER_HEMISPHERE,):
        raise ValueError(f"Unexpected left label shape: {left_labels.shape}")
    if right_labels.shape != (FSAVERAGE5_VERTICES_PER_HEMISPHERE,):
        raise ValueError(f"Unexpected right label shape: {right_labels.shape}")
    if len(left_names) != N_GLASSER_PER_HEMISPHERE or len(right_names) != N_GLASSER_PER_HEMISPHERE:
        raise ValueError(
            f"Expected 180 labels per hemisphere, got {len(left_names)} and {len(right_names)}"
        )

    matrix = np.zeros(
        (N_GLASSER_TOTAL, 2 * FSAVERAGE5_VERTICES_PER_HEMISPHERE), dtype=np.float32
    )
    names: list[str] = []
    for hemi_index, (labels, label_names, prefix) in enumerate(
        ((left_labels, left_names, "L"), (right_labels, right_names, "R"))
    ):
        vertex_offset = hemi_index * FSAVERAGE5_VERTICES_PER_HEMISPHERE
        parcel_offset = hemi_index * N_GLASSER_PER_HEMISPHERE
        for local_index, name in enumerate(label_names):
            vertices = np.flatnonzero(labels == local_index)
            if not len(vertices):
                raise ValueError(f"Empty {prefix} parcel {local_index}: {name}")
            matrix[parcel_offset + local_index, vertex_offset + vertices] = 1.0 / len(vertices)
            names.append(f"{prefix}_{name}")
    return ParcelOperator(matrix=matrix, names=tuple(names))


def compile_hcp_mmp_operator(
    left_annot: str | Path,
    right_annot: str | Path,
    output: str | Path,
) -> Path:
    """Compile HCP-MMP annotations once; training later performs no atlas download."""

    try:
        from nibabel.freesurfer import read_annot
    except ImportError as exc:  # pragma: no cover - remote dependency
        raise RuntimeError("Atlas compilation requires nibabel") from exc

    left_path = Path(left_annot)
    right_path = Path(right_annot)
    left, _, left_raw_names = read_annot(left_path, orig_ids=False)
    right, _, right_raw_names = read_annot(right_path, orig_ids=False)

    def normalize(labels: np.ndarray, raw_names: Sequence[bytes]) -> tuple[np.ndarray, list[str]]:
        labels = np.asarray(labels[:FSAVERAGE5_VERTICES_PER_HEMISPHERE], dtype=np.int64)
        decoded = [item.decode("utf8") if isinstance(item, bytes) else str(item) for item in raw_names]
        keep = [
            index
            for index, name in enumerate(decoded)
            if name.lower() not in {"???", "unknown", "medial_wall"}
        ]
        if len(keep) != N_GLASSER_PER_HEMISPHERE:
            raise ValueError(f"Expected 180 non-background HCP labels, got {len(keep)}")
        remap = np.full(max(len(decoded), int(labels.max()) + 1), -1, dtype=np.int64)
        for new_index, old_index in enumerate(keep):
            remap[old_index] = new_index
        valid = labels >= 0
        normalized = np.full(labels.shape, -1, dtype=np.int64)
        normalized[valid] = remap[labels[valid]]
        names = [decoded[index].replace("_ROI", "") for index in keep]
        return normalized, names

    left_labels, left_names = normalize(left, left_raw_names)
    right_labels, right_names = normalize(right, right_raw_names)
    operator = _labels_to_operator(left_labels, right_labels, left_names, right_names)
    metadata = {
        "atlas": "HCP-MMP1.0",
        "mesh": "fsaverage5",
        "left_annot_sha256": _sha256(left_path),
        "right_annot_sha256": _sha256(right_path),
        "parcel_order": "left-180 then right-180",
    }
    return operator.save(output, metadata)


def parcel_adjacency_from_faces(
    operator: ParcelOperator,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
) -> np.ndarray:
    """Construct a 360x360 binary adjacency matrix from fsaverage5 triangles."""

    vertex_to_parcel = np.full(operator.matrix.shape[1], -1, dtype=np.int64)
    nonzero_rows, nonzero_columns = np.nonzero(operator.matrix)
    vertex_to_parcel[nonzero_columns] = nonzero_rows
    adjacency = np.zeros((N_GLASSER_TOTAL, N_GLASSER_TOTAL), dtype=np.float32)
    for offset, faces in (
        (0, np.asarray(left_faces)),
        (FSAVERAGE5_VERTICES_PER_HEMISPHERE, np.asarray(right_faces)),
    ):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            p = vertex_to_parcel[offset + faces[:, first]]
            q = vertex_to_parcel[offset + faces[:, second]]
            valid = (p >= 0) & (q >= 0) & (p != q)
            adjacency[p[valid], q[valid]] = 1
            adjacency[q[valid], p[valid]] = 1
    return adjacency


def laplacian_basis(adjacency: np.ndarray, n_modes: int = 16) -> np.ndarray:
    """Return low-frequency non-null graph modes, excluding hemisphere constants."""

    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.shape != (N_GLASSER_TOTAL, N_GLASSER_TOTAL):
        raise ValueError(f"Expected (360, 360) adjacency, got {adjacency.shape}")
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    values, vectors = np.linalg.eigh(laplacian)
    non_null = np.flatnonzero(values > 1e-8)
    if len(non_null) < n_modes:
        raise ValueError("Parcel graph has too few non-null Laplacian modes")
    basis = vectors[:, non_null[:n_modes]]
    basis -= basis.mean(axis=0, keepdims=True)
    basis /= np.sqrt(np.mean(basis**2, axis=0, keepdims=True))
    return basis.astype(np.float32)


# Register an offline projector with neuralset only in the full remote environment.
try:  # pragma: no cover - unavailable in the lightweight local test environment
    import pydantic
    from tribev2.utils_fmri import TribeSurfaceProjector

    class CompiledGlasser360Projector(TribeSurfaceProjector):
        operator_path: Path
        mesh_data_dir: Path | None = None
        _operator: ParcelOperator | None = pydantic.PrivateAttr(default=None)

        def get_operator(self) -> ParcelOperator:
            if self._operator is None:
                self._operator = ParcelOperator.load(self.operator_path)
            return self._operator

        def get_mesh(self):
            if self.mesh_data_dir is None:
                return super().get_mesh()
            if self._mesh is None:
                from nilearn import datasets

                self._mesh = datasets.fetch_surf_fsaverage(
                    self.mesh, data_dir=self.mesh_data_dir
                )
            return self._mesh

        def apply(self, rec):
            vertices = super().apply(rec)
            return self.get_operator().transform(vertices, vertex_axis=0)

except ImportError:  # importing spfcl must remain possible before remote bootstrap
    CompiledGlasser360Projector = None  # type: ignore[assignment]
