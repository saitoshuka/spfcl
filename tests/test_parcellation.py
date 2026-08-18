from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spfcl.data.parcellate import (
    FSAVERAGE5_VERTICES_PER_HEMISPHERE,
    N_GLASSER_PER_HEMISPHERE,
    N_GLASSER_TOTAL,
    ParcelOperator,
    _labels_to_operator,
    laplacian_basis,
    parcel_adjacency_from_faces,
)


@pytest.fixture(scope="module")
def operator() -> ParcelOperator:
    vertices = FSAVERAGE5_VERTICES_PER_HEMISPHERE
    left_labels = np.arange(vertices, dtype=np.int64) % N_GLASSER_PER_HEMISPHERE
    right_labels = np.arange(vertices, dtype=np.int64) % N_GLASSER_PER_HEMISPHERE
    left_names = [f"area-{index:03d}" for index in range(N_GLASSER_PER_HEMISPHERE)]
    right_names = [f"area-{index:03d}" for index in range(N_GLASSER_PER_HEMISPHERE)]
    return _labels_to_operator(left_labels, right_labels, left_names, right_names)


def test_operator_has_360_distinct_hemisphere_specific_rows(operator: ParcelOperator) -> None:
    assert operator.matrix.shape == (N_GLASSER_TOTAL, 20_484)
    assert len(operator.names) == N_GLASSER_TOTAL
    assert operator.names[0] == "L_area-000"
    assert operator.names[179] == "L_area-179"
    assert operator.names[180] == "R_area-000"
    assert operator.names[-1] == "R_area-179"
    np.testing.assert_allclose(operator.matrix.sum(axis=1), 1.0, atol=1e-6)


def test_constant_vertex_field_is_preserved(operator: ParcelOperator) -> None:
    vertices = np.ones((20_484, 5), dtype=np.float32)
    parcels = operator.transform(vertices, vertex_axis=0)

    assert parcels.shape == (360, 5)
    np.testing.assert_allclose(parcels, 1.0, atol=1e-6)


def test_left_and_right_homologs_are_not_merged(operator: ParcelOperator) -> None:
    vertices = np.empty((20_484, 3), dtype=np.float32)
    vertices[:FSAVERAGE5_VERTICES_PER_HEMISPHERE] = 1.0
    vertices[FSAVERAGE5_VERTICES_PER_HEMISPHERE:] = 3.0
    parcels = operator.transform(vertices, vertex_axis=0)

    np.testing.assert_allclose(parcels[:180], 1.0, atol=1e-6)
    np.testing.assert_allclose(parcels[180:], 3.0, atol=1e-6)


def test_transform_preserves_batch_and_time_axes(operator: ParcelOperator) -> None:
    rng = np.random.default_rng(9)
    vertices = rng.standard_normal((2, 20_484, 4), dtype=np.float32)
    parcels = operator.transform(vertices, vertex_axis=1)

    assert parcels.shape == (2, 360, 4)
    manual = operator.matrix @ vertices[0, :, 0]
    np.testing.assert_allclose(parcels[0, :, 0], manual, atol=1e-5, rtol=1e-5)


def test_operator_round_trip_is_exact_within_float32(
    operator: ParcelOperator, tmp_path: Path
) -> None:
    path = operator.save(tmp_path / "glasser360.npz", metadata={"source": "test"})
    restored = ParcelOperator.load(path)

    assert restored.names == operator.names
    assert np.array_equal(restored.matrix, operator.matrix)


def test_constructor_rejects_wrong_shape_or_row_normalization() -> None:
    with pytest.raises(ValueError, match="shape"):
        ParcelOperator(matrix=np.ones((360, 10), dtype=np.float32), names=("x",) * 360)

    matrix = np.zeros((360, 20_484), dtype=np.float32)
    with pytest.raises(ValueError, match="sum to one"):
        ParcelOperator(matrix=matrix, names=tuple(f"p{i}" for i in range(360)))


def test_empty_parcel_is_rejected() -> None:
    vertices = FSAVERAGE5_VERTICES_PER_HEMISPHERE
    left = np.arange(vertices, dtype=np.int64) % N_GLASSER_PER_HEMISPHERE
    right = np.arange(vertices, dtype=np.int64) % N_GLASSER_PER_HEMISPHERE
    left[left == 179] = 178
    names = [f"area-{index:03d}" for index in range(180)]

    with pytest.raises(ValueError, match="Empty L parcel 179"):
        _labels_to_operator(left, right, names, names)


def test_bad_vertex_axis_is_rejected(operator: ParcelOperator) -> None:
    with pytest.raises(ValueError, match="Expected 20484 vertices"):
        operator.transform(np.zeros((360, 10), dtype=np.float32), vertex_axis=0)


def test_face_adjacency_is_symmetric_and_does_not_cross_hemispheres(
    operator: ParcelOperator,
) -> None:
    left_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    right_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    adjacency = parcel_adjacency_from_faces(operator, left_faces, right_faces)

    assert adjacency.shape == (360, 360)
    assert np.array_equal(adjacency, adjacency.T)
    assert np.count_nonzero(np.diag(adjacency)) == 0
    assert adjacency[0, 1] == adjacency[1, 0] == 1
    assert adjacency[180, 181] == adjacency[181, 180] == 1
    assert np.count_nonzero(adjacency[:180, 180:]) == 0


def test_laplacian_basis_excludes_null_mode_and_normalizes_columns() -> None:
    adjacency = np.zeros((360, 360), dtype=np.float32)
    indices = np.arange(359)
    adjacency[indices, indices + 1] = 1
    adjacency[indices + 1, indices] = 1

    basis = laplacian_basis(adjacency, n_modes=4)

    assert basis.shape == (360, 4)
    np.testing.assert_allclose(basis.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.mean(basis**2, axis=0)), 1.0, atol=1e-6)

