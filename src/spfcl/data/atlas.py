from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .parcellate import (
    ParcelOperator,
    _sha256,
    compile_hcp_mmp_operator,
    laplacian_basis,
    parcel_adjacency_from_faces,
)


def _gifti_faces(path: str | Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - remote dependency
        raise RuntimeError("Atlas preparation requires nibabel") from exc
    image = nib.load(str(path))
    arrays = getattr(image, "darrays", None)
    if not arrays or len(arrays) < 2:
        raise RuntimeError(f"Could not read surface triangles from {path}")
    return np.asarray(arrays[1].data, dtype=np.int64)


def prepare_glasser360(
    output_dir: str | Path,
    *,
    accept_hcp_license: bool = False,
    n_modes: int = 16,
) -> dict[str, Path]:
    """Fetch the two annotation files once and compile offline training assets."""

    if not accept_hcp_license:
        raise PermissionError(
            "Read the HCP-MMP parcellation license and rerun with --accept-hcp-license."
        )
    try:
        import mne
        from nilearn import datasets
    except ImportError as exc:  # pragma: no cover - remote dependency
        raise RuntimeError("Atlas preparation requires the spfcl[data] dependencies") from exc

    root = Path(output_dir).expanduser().resolve()
    subjects_dir = root / "freesurfer_subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    # This API downloads only the HCP-MMP annotation assets. It intentionally does
    # not call mne.datasets.sample.data_path(), which can pull the large sample set.
    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=subjects_dir,
        combine=False,
        accept=True,
        verbose=True,
    )
    label_dir = subjects_dir / "fsaverage" / "label"
    left_annot = label_dir / "lh.HCPMMP1.annot"
    right_annot = label_dir / "rh.HCPMMP1.annot"
    if not left_annot.is_file() or not right_annot.is_file():
        raise FileNotFoundError(
            f"MNE did not create the expected HCP annotations under {label_dir}"
        )

    operator_path = root / "glasser360_fsaverage5.npz"
    compile_hcp_mmp_operator(left_annot, right_annot, operator_path)
    operator = ParcelOperator.load(operator_path)

    meshes = datasets.fetch_surf_fsaverage(mesh="fsaverage5", data_dir=root / "nilearn")
    left_faces = _gifti_faces(meshes["pial_left"])
    right_faces = _gifti_faces(meshes["pial_right"])
    adjacency = parcel_adjacency_from_faces(operator, left_faces, right_faces)
    basis = laplacian_basis(adjacency, n_modes=n_modes)
    basis_path = root / "glasser360_laplacian_basis.npz"
    np.savez_compressed(
        basis_path,
        basis=basis,
        adjacency=adjacency,
        parcel_names=np.asarray(operator.names),
    )

    manifest_path = root / "atlas_manifest.json"
    manifest = {
        "atlas": "HCP-MMP1.0",
        "mesh": "fsaverage5",
        "parcel_order": "left-180 then right-180",
        "n_parcels": 360,
        "n_vertices": 20_484,
        "operator": operator_path.name,
        "operator_sha256": _sha256(operator_path),
        "basis": basis_path.name,
        "basis_sha256": _sha256(basis_path),
        "left_annot_sha256": _sha256(left_annot),
        "right_annot_sha256": _sha256(right_annot),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    return {
        "operator": operator_path,
        "basis": basis_path,
        "manifest": manifest_path,
    }

