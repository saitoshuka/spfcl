from __future__ import annotations

import numpy as np
import pytest

from spfcl.eval import (
    conditional_nuisance_probe,
    dose_response_slope,
    nuisance_projection_strength,
    orthogonal_subspace_ablation,
    paired_nuisance_swap_effect,
)


def _conditional_probe_fixture() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(71)
    n_samples = 240
    subject_ids = np.repeat(np.arange(4), n_samples // 4)
    fold_ids = np.tile(np.repeat(np.arange(12), 5), 4)
    stimulus = rng.standard_normal((n_samples, 3))
    latent_nuisance = rng.standard_normal(n_samples)
    subject_effect = np.asarray([-1.0, -0.2, 0.4, 1.1])[subject_ids]
    nuisance = (
        subject_effect
        + stimulus @ np.asarray([0.7, -0.4, 0.2])
        + 1.5 * latent_nuisance
        + 0.05 * rng.standard_normal(n_samples)
    )
    representation = np.column_stack(
        [
            latent_nuisance + 0.03 * rng.standard_normal(n_samples),
            rng.standard_normal((n_samples, 5)),
        ]
    )
    return representation, nuisance, subject_ids, fold_ids, stimulus


def test_conditional_probe_measures_incremental_nuisance_information() -> None:
    representation, nuisance, subjects, folds, stimulus = _conditional_probe_fixture()

    result = conditional_nuisance_probe(
        representation,
        nuisance,
        subject_ids=subjects,
        fold_ids=folds,
        stimulus_covariates=stimulus,
        alpha=0.1,
        metric="r2",
    )

    assert result.full_score > 0.95
    assert result.incremental_score > 0.3
    assert result.full_score > result.baseline_score
    assert result.baseline_prediction.shape == nuisance.shape
    assert result.full_prediction.shape == nuisance.shape
    assert result.n_subjects == 4
    assert result.n_folds == 12
    assert result.n_seeds is None


def test_conditional_probe_supports_linear_fit_and_correlation() -> None:
    representation, nuisance, subjects, folds, stimulus = _conditional_probe_fixture()

    result = conditional_nuisance_probe(
        representation,
        nuisance,
        subject_ids=subjects,
        fold_ids=folds,
        stimulus_covariates=stimulus,
        alpha=0.0,
        metric="correlation",
    )

    assert result.metric == "correlation"
    assert result.full_score > 0.98
    assert result.full_score > result.baseline_score


def test_conditional_correlation_treats_constant_baseline_as_no_information() -> None:
    nuisance = np.tile(np.asarray([-1.0, 1.0]), 6)
    representation = nuisance[:, None]
    folds = np.repeat(np.arange(6), 2)

    result = conditional_nuisance_probe(
        representation,
        nuisance,
        subject_ids=np.repeat("subject-1", nuisance.size),
        fold_ids=folds,
        alpha=0.0,
        metric="correlation",
    )

    assert result.baseline_score == 0.0
    assert result.full_score == pytest.approx(1.0, abs=1e-12)


def test_conditional_probe_rejects_bad_shapes_groups_and_values() -> None:
    representation, nuisance, subjects, folds, stimulus = _conditional_probe_fixture()

    with pytest.raises(ValueError, match="nuisance has"):
        conditional_nuisance_probe(
            representation,
            nuisance[:-1],
            subject_ids=subjects,
            fold_ids=folds,
        )
    with pytest.raises(ValueError, match="at least two"):
        conditional_nuisance_probe(
            representation,
            nuisance,
            subject_ids=subjects,
            fold_ids=np.zeros_like(folds),
        )
    with pytest.raises(ValueError, match="stimulus_covariates has"):
        conditional_nuisance_probe(
            representation,
            nuisance,
            subject_ids=subjects,
            fold_ids=folds,
            stimulus_covariates=stimulus[:-1],
        )
    bad = representation.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        conditional_nuisance_probe(
            bad,
            nuisance,
            subject_ids=subjects,
            fold_ids=folds,
        )
    with pytest.raises(ValueError, match="alpha"):
        conditional_nuisance_probe(
            representation,
            nuisance,
            subject_ids=subjects,
            fold_ids=folds,
            alpha=-1,
        )


def test_seed_ids_cannot_be_a_relabeling_of_subject_ids() -> None:
    representation = np.arange(12, dtype=np.float64).reshape(4, 3)
    nuisance = np.asarray([0.0, 1.0, 2.0, 3.0])
    subjects = np.asarray(["sub-a", "sub-a", "sub-b", "sub-b"])
    relabeled_seeds = np.asarray(["seed-1", "seed-1", "seed-2", "seed-2"])

    with pytest.raises(ValueError, match="cannot be treated as biological subjects"):
        conditional_nuisance_probe(
            representation,
            nuisance,
            subject_ids=subjects,
            seed_ids=relabeled_seeds,
            fold_ids=np.asarray([0, 1, 0, 1]),
        )


def test_nuisance_projection_strength_recovers_known_coefficient() -> None:
    rng = np.random.default_rng(2)
    canonical = rng.standard_normal((4, 12, 20))
    nuisance = rng.standard_normal((4, 12, 20))
    predicted = canonical + 0.7 * nuisance

    global_strength = nuisance_projection_strength(predicted, canonical, nuisance)
    subject_strength = nuisance_projection_strength(
        predicted, canonical, nuisance, axis=(-2, -1)
    )

    assert global_strength == pytest.approx(0.7, abs=1e-12)
    np.testing.assert_allclose(subject_strength, 0.7, atol=1e-12)


def test_nuisance_projection_strength_rejects_mismatch_and_zero_direction() -> None:
    value = np.ones((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="identical shapes"):
        nuisance_projection_strength(value, value[:, :2], value)
    with pytest.raises(ValueError, match="zero norm"):
        nuisance_projection_strength(value, value, np.zeros_like(value))
    with pytest.raises(ValueError, match="axis tuple"):
        nuisance_projection_strength(value, value, value, axis=())


def test_orthogonal_subspace_ablation_removes_only_basis_span() -> None:
    rng = np.random.default_rng(8)
    representation = rng.standard_normal((30, 6))
    basis = np.zeros((6, 3), dtype=np.float64)
    basis[0, 0] = 1.0
    basis[1, 1] = 2.0
    basis[0, 2] = 1.0  # duplicate direction should not change the span

    ablated = orthogonal_subspace_ablation(representation, basis)
    repeated = orthogonal_subspace_ablation(ablated, basis)

    np.testing.assert_allclose(ablated[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(ablated[:, 2:], representation[:, 2:], atol=1e-12)
    np.testing.assert_allclose(repeated, ablated, atol=1e-12)


def test_orthogonal_subspace_ablation_supports_nonfinal_feature_axis() -> None:
    rng = np.random.default_rng(19)
    representation = rng.standard_normal((2, 5, 7))
    basis = np.eye(5, dtype=np.float64)[:, :2]

    ablated = orthogonal_subspace_ablation(
        representation, basis, feature_axis=1
    )

    assert ablated.shape == representation.shape
    np.testing.assert_allclose(ablated[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(ablated[:, 2:], representation[:, 2:], atol=1e-12)


def test_orthogonal_subspace_ablation_rejects_bad_basis() -> None:
    representation = np.ones((3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="basis has 3 features"):
        orthogonal_subspace_ablation(representation, np.ones((3, 1)))
    with pytest.raises(ValueError, match="zero numerical rank"):
        orthogonal_subspace_ablation(representation, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="out of bounds"):
        orthogonal_subspace_ablation(representation, np.ones((4, 1)), feature_axis=3)


def test_paired_swap_effect_recovers_alignment_and_equal_subject_aggregation() -> None:
    rng = np.random.default_rng(29)
    subjects = np.repeat(np.arange(4), 3)
    seeds = np.tile(np.arange(3), 4)
    nuisance_change = rng.standard_normal((12, 20))
    original_nuisance = np.zeros_like(nuisance_change)
    swapped_nuisance = nuisance_change
    original_prediction = rng.standard_normal((12, 20))
    swapped_prediction = original_prediction + 0.6 * nuisance_change

    result = paired_nuisance_swap_effect(
        original_prediction,
        swapped_prediction,
        original_nuisance,
        swapped_nuisance,
        subject_ids=subjects,
        seed_ids=seeds,
    )

    np.testing.assert_allclose(result.per_pair_projection, 0.6, atol=1e-12)
    np.testing.assert_allclose(result.per_pair_cosine, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.subject_projection, 0.6, atol=1e-12)
    assert result.mean_projection == pytest.approx(0.6, abs=1e-12)
    assert result.mean_cosine == pytest.approx(1.0, abs=1e-12)
    assert result.n_subjects == 4
    assert result.n_seeds == 3


def test_paired_swap_effect_treats_no_model_response_as_zero_alignment() -> None:
    nuisance = np.asarray([[1.0, -1.0], [2.0, 1.0]])
    prediction = np.zeros_like(nuisance)
    result = paired_nuisance_swap_effect(
        prediction,
        prediction,
        np.zeros_like(nuisance),
        nuisance,
        subject_ids=np.asarray(["s1", "s2"]),
    )

    np.testing.assert_allclose(result.per_pair_projection, 0.0)
    np.testing.assert_allclose(result.per_pair_cosine, 0.0)


def test_paired_swap_effect_rejects_zero_or_unpaired_intervention() -> None:
    value = np.ones((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="share non-empty shape"):
        paired_nuisance_swap_effect(
            value,
            value[:-1],
            value,
            value,
            subject_ids=np.arange(4),
        )
    with pytest.raises(ValueError, match="non-zero intervention"):
        paired_nuisance_swap_effect(
            value,
            value,
            value,
            value,
            subject_ids=np.arange(4),
        )


def test_dose_response_fits_within_seed_then_counts_four_subjects() -> None:
    subject_slopes = np.asarray([0.5, 1.0, 1.5, 2.0])
    records = []
    for subject, slope in enumerate(subject_slopes):
        for seed in range(3):
            for dose in (0.0, 0.2, 0.4):
                effect = 10.0 * subject + seed + slope * dose
                records.append((dose, effect, subject, seed))
    values = np.asarray(records, dtype=np.float64)

    result = dose_response_slope(
        values[:, 0],
        values[:, 1],
        subject_ids=values[:, 2].astype(int),
        seed_ids=values[:, 3].astype(int),
    )

    np.testing.assert_allclose(result.subject_slopes, subject_slopes, atol=1e-12)
    assert result.mean_slope == pytest.approx(1.25, abs=1e-12)
    assert result.n_subjects == 4
    assert result.n_seeds == 3


def test_dose_response_rejects_alias_and_single_dose_cells() -> None:
    with pytest.raises(ValueError, match="cannot be treated as biological subjects"):
        dose_response_slope(
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            subject_ids=["a", "a", "b", "b"],
            seed_ids=["x", "x", "y", "y"],
        )
    with pytest.raises(ValueError, match="at least two distinct doses"):
        dose_response_slope(
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 2.0, 3.0],
            subject_ids=["a", "a", "b", "b"],
        )
