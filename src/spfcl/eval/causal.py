from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

ScoreMetric = Literal["r2", "correlation"]


@dataclass(frozen=True)
class ConditionalProbeResult:
    """Out-of-fold nuisance prediction with and without representation features."""

    baseline_score: float
    full_score: float
    incremental_score: float
    baseline_prediction: np.ndarray
    full_prediction: np.ndarray
    n_subjects: int
    n_folds: int
    n_seeds: int | None
    metric: ScoreMetric


@dataclass(frozen=True)
class PairedSwapResult:
    """Alignment of a paired prediction change with a nuisance intervention."""

    per_pair_projection: np.ndarray
    per_pair_cosine: np.ndarray
    subject_labels: np.ndarray
    subject_projection: np.ndarray
    subject_cosine: np.ndarray
    mean_projection: float
    mean_cosine: float
    n_subjects: int
    n_seeds: int | None


@dataclass(frozen=True)
class DoseResponseResult:
    """Within-subject dose slopes, averaged over seeds before subjects."""

    subject_labels: np.ndarray
    subject_slopes: np.ndarray
    mean_slope: float
    n_subjects: int
    n_seeds: int | None


def _finite_vector(name: str, value: np.ndarray | Sequence[float]) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.ndim != 1 or output.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} must contain only finite values")
    return output


def _finite_matrix(
    name: str,
    value: np.ndarray | Sequence[Sequence[float]],
    *,
    n_samples: int | None = None,
) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.ndim != 2 or output.shape[0] == 0 or output.shape[1] == 0:
        raise ValueError(f"{name} must have shape (sample, feature) with non-zero dimensions")
    if n_samples is not None and output.shape[0] != n_samples:
        raise ValueError(
            f"{name} has {output.shape[0]} samples, expected {n_samples}"
        )
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} must contain only finite values")
    return output


def _id_vector(name: str, value: np.ndarray | Sequence[object], n_samples: int) -> np.ndarray:
    output = np.asarray(value)
    if output.ndim != 1 or output.shape[0] != n_samples:
        raise ValueError(f"{name} must have shape ({n_samples},)")
    labels = np.asarray([str(item) for item in output.tolist()], dtype=object)
    if any(label in {"", "None", "nan"} for label in labels):
        raise ValueError(f"{name} contains a missing identifier")
    return labels


def _validate_subject_and_seed_ids(
    subject_ids: np.ndarray | Sequence[object],
    seed_ids: np.ndarray | Sequence[object] | None,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    subjects = _id_vector("subject_ids", subject_ids, n_samples)
    seeds = None if seed_ids is None else _id_vector("seed_ids", seed_ids, n_samples)
    if seeds is None:
        return subjects, None

    subject_levels = np.unique(subjects)
    seed_levels = np.unique(seeds)
    subject_to_seed = [np.unique(seeds[subjects == subject]).size for subject in subject_levels]
    seed_to_subject = [np.unique(subjects[seeds == seed]).size for seed in seed_levels]
    if all(count == 1 for count in subject_to_seed) and all(
        count == 1 for count in seed_to_subject
    ):
        raise ValueError(
            "seed_ids are a one-to-one alias of subject_ids; random seeds cannot be "
            "treated as biological subjects"
        )
    return subjects, seeds


def _subject_one_hot(subject_ids: np.ndarray) -> np.ndarray:
    levels = np.unique(subject_ids)
    return (subject_ids[:, None] == levels[None, :]).astype(np.float64)


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    mean_x = train_x.mean(axis=0)
    scale_x = train_x.std(axis=0)
    active = scale_x > 1e-12
    mean_y = float(train_y.mean())
    if not np.any(active):
        return np.full(test_x.shape[0], mean_y, dtype=np.float64)

    train_z = (train_x[:, active] - mean_x[active]) / scale_x[active]
    test_z = (test_x[:, active] - mean_x[active]) / scale_x[active]
    centered_y = train_y - mean_y
    if alpha == 0:
        coefficients = np.linalg.lstsq(train_z, centered_y, rcond=None)[0]
    else:
        gram = train_z.T @ train_z
        regularized = gram + alpha * np.eye(gram.shape[0], dtype=np.float64)
        try:
            coefficients = np.linalg.solve(regularized, train_z.T @ centered_y)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(regularized, train_z.T @ centered_y, rcond=None)[0]
    return mean_y + test_z @ coefficients


def _prediction_score(target: np.ndarray, prediction: np.ndarray, metric: ScoreMetric) -> float:
    centered_target = target - target.mean()
    if metric == "r2":
        denominator = float(centered_target @ centered_target)
        if denominator <= 1e-12:
            raise ValueError("nuisance target is constant, so cross-validated R² is undefined")
        residual = target - prediction
        return float(1.0 - (residual @ residual) / denominator)
    if metric == "correlation":
        centered_prediction = prediction - prediction.mean()
        target_energy = centered_target @ centered_target
        prediction_energy = centered_prediction @ centered_prediction
        if target_energy <= 1e-12:
            raise ValueError("nuisance target is constant, so correlation is undefined")
        if prediction_energy <= 1e-12:
            return 0.0
        denominator = float(np.sqrt(target_energy * prediction_energy))
        return float((centered_target @ centered_prediction) / denominator)
    raise ValueError("metric must be 'r2' or 'correlation'")


def conditional_nuisance_probe(
    representation: np.ndarray,
    nuisance: np.ndarray,
    *,
    subject_ids: np.ndarray | Sequence[object],
    fold_ids: np.ndarray | Sequence[object],
    stimulus_covariates: np.ndarray | None = None,
    seed_ids: np.ndarray | Sequence[object] | None = None,
    alpha: float = 1.0,
    metric: ScoreMetric = "r2",
) -> ConditionalProbeResult:
    """Measure incremental nuisance information using grouped cross-validation.

    ``representation`` has shape ``(sample, feature)`` and ``nuisance`` has shape
    ``(sample,)``. Subject fixed effects and optional numeric stimulus covariates form
    the baseline model. The full model adds representation features. Each unique
    ``fold_ids`` value is held out once; run IDs are a suitable Phase-1 choice.

    Seeds are accepted only as an explicit, separate identifier. A one-to-one
    subject/seed alias is rejected so algorithmic repetitions cannot silently become
    biological samples.
    """

    features = _finite_matrix("representation", representation)
    target = _finite_vector("nuisance", nuisance)
    if target.shape[0] != features.shape[0]:
        raise ValueError(
            f"nuisance has {target.shape[0]} samples, expected {features.shape[0]}"
        )
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative scalar")
    if metric not in {"r2", "correlation"}:
        raise ValueError("metric must be 'r2' or 'correlation'")

    n_samples = features.shape[0]
    subjects, seeds = _validate_subject_and_seed_ids(subject_ids, seed_ids, n_samples)
    folds = _id_vector("fold_ids", fold_ids, n_samples)
    fold_levels = np.unique(folds)
    if fold_levels.size < 2:
        raise ValueError("fold_ids must contain at least two held-out groups")

    subject_covariates = _subject_one_hot(subjects)
    if stimulus_covariates is None:
        baseline = subject_covariates
    else:
        stimulus = _finite_matrix(
            "stimulus_covariates", stimulus_covariates, n_samples=n_samples
        )
        baseline = np.concatenate([subject_covariates, stimulus], axis=1)
    full = np.concatenate([baseline, features], axis=1)

    baseline_prediction = np.empty(n_samples, dtype=np.float64)
    full_prediction = np.empty(n_samples, dtype=np.float64)
    for fold in fold_levels:
        test = folds == fold
        train = ~test
        if train.sum() < 2 or test.sum() == 0:
            raise ValueError(f"fold {fold!r} does not have a usable train/test partition")
        baseline_prediction[test] = _ridge_predict(
            baseline[train], target[train], baseline[test], alpha=alpha
        )
        full_prediction[test] = _ridge_predict(
            full[train], target[train], full[test], alpha=alpha
        )

    baseline_score = _prediction_score(target, baseline_prediction, metric)
    full_score = _prediction_score(target, full_prediction, metric)
    return ConditionalProbeResult(
        baseline_score=baseline_score,
        full_score=full_score,
        incremental_score=full_score - baseline_score,
        baseline_prediction=baseline_prediction,
        full_prediction=full_prediction,
        n_subjects=int(np.unique(subjects).size),
        n_folds=int(fold_levels.size),
        n_seeds=None if seeds is None else int(np.unique(seeds).size),
        metric=metric,
    )


def nuisance_projection_strength(
    predicted: np.ndarray,
    canonical: np.ndarray,
    nuisance: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> float | np.ndarray:
    """Return ``<predicted-canonical, nuisance> / <nuisance, nuisance>``.

    With ``axis=None`` all elements form one projection. Passing axes, for example
    ``axis=(-2, -1)``, preserves leading subject/checkpoint dimensions.
    """

    prediction = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(canonical, dtype=np.float64)
    direction = np.asarray(nuisance, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.shape != direction.shape:
        raise ValueError(
            "predicted, canonical, and nuisance must have identical shapes; got "
            f"{prediction.shape}, {reference.shape}, and {direction.shape}"
        )
    if prediction.size == 0 or not all(
        np.all(np.isfinite(value)) for value in (prediction, reference, direction)
    ):
        raise ValueError("projection inputs must be non-empty and finite")
    if isinstance(axis, tuple) and not axis:
        raise ValueError("axis tuple may not be empty")

    residual = prediction - reference
    numerator = np.sum(residual * direction, axis=axis)
    denominator = np.sum(direction**2, axis=axis)
    if np.any(denominator <= 1e-12):
        raise ValueError("nuisance direction has zero norm along a requested projection")
    output = numerator / denominator
    return float(output) if np.ndim(output) == 0 else np.asarray(output)


def orthogonal_subspace_ablation(
    representation: np.ndarray,
    basis: np.ndarray,
    *,
    feature_axis: int = -1,
    rcond: float = 1e-10,
) -> np.ndarray:
    """Remove the span of ``basis`` from a representation feature axis.

    ``basis`` is shaped ``(feature, direction)`` or ``(feature,)``. SVD makes
    the operation invariant to direction scaling and safely removes duplicate
    directions. No probe or head is refit after ablation.
    """

    value = np.asarray(representation, dtype=np.float64)
    directions = np.asarray(basis, dtype=np.float64)
    if value.ndim == 0 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("representation must be a non-empty finite array")
    if directions.ndim == 1:
        directions = directions[:, None]
    if directions.ndim != 2 or directions.size == 0 or not np.all(np.isfinite(directions)):
        raise ValueError("basis must be a finite (feature, direction) matrix")
    if not np.isfinite(rcond) or rcond <= 0:
        raise ValueError("rcond must be a finite positive scalar")

    axis = int(feature_axis)
    if axis < 0:
        axis += value.ndim
    if axis < 0 or axis >= value.ndim:
        raise ValueError(
            f"feature_axis {feature_axis} is out of bounds for {value.ndim} dimensions"
        )
    if value.shape[axis] != directions.shape[0]:
        raise ValueError(
            f"basis has {directions.shape[0]} features, but representation axis "
            f"{axis} has {value.shape[axis]}"
        )
    left, singular_values, _ = np.linalg.svd(directions, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-12:
        raise ValueError("basis has zero numerical rank")
    keep = singular_values > rcond * singular_values[0]
    orthonormal = left[:, keep]
    if orthonormal.shape[1] == 0:
        raise ValueError("basis has zero numerical rank at the requested rcond")

    moved = np.moveaxis(value, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    ablated = flat - (flat @ orthonormal) @ orthonormal.T
    restored = ablated.reshape(moved.shape)
    return np.moveaxis(restored, -1, axis)


def _subject_equal_means(
    values: np.ndarray,
    subjects: np.ndarray,
    seeds: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    subject_labels = np.unique(subjects)
    output = np.empty(subject_labels.size, dtype=np.float64)
    for index, subject in enumerate(subject_labels):
        selected_subject = subjects == subject
        if seeds is None:
            output[index] = float(values[selected_subject].mean())
            continue
        per_seed = []
        for seed in np.unique(seeds[selected_subject]):
            cell = selected_subject & (seeds == seed)
            per_seed.append(float(values[cell].mean()))
        output[index] = float(np.mean(per_seed))
    return subject_labels, output


def paired_nuisance_swap_effect(
    original_prediction: np.ndarray,
    swapped_prediction: np.ndarray,
    original_nuisance: np.ndarray,
    swapped_nuisance: np.ndarray,
    *,
    subject_ids: np.ndarray | Sequence[object],
    seed_ids: np.ndarray | Sequence[object] | None = None,
) -> PairedSwapResult:
    """Measure whether paired prediction changes follow a nuisance swap.

    Inputs have shape ``(pair, ...)``. Projection and cosine are first computed
    within each exact pair. Replicates are then averaged within seed, then within
    biological subject, and finally equally across subjects.
    """

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            original_prediction,
            swapped_prediction,
            original_nuisance,
            swapped_nuisance,
        )
    ]
    shapes = {value.shape for value in arrays}
    if len(shapes) != 1 or arrays[0].ndim < 2 or arrays[0].shape[0] == 0:
        raise ValueError("all swap inputs must share non-empty shape (pair, ...)")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("swap inputs must contain only finite values")

    n_pairs = arrays[0].shape[0]
    subjects, seeds = _validate_subject_and_seed_ids(subject_ids, seed_ids, n_pairs)
    prediction_change = (arrays[1] - arrays[0]).reshape(n_pairs, -1)
    nuisance_change = (arrays[3] - arrays[2]).reshape(n_pairs, -1)
    nuisance_energy = np.sum(nuisance_change**2, axis=1)
    if np.any(nuisance_energy <= 1e-12):
        raise ValueError("every nuisance swap pair must have a non-zero intervention")

    dot = np.sum(prediction_change * nuisance_change, axis=1)
    per_pair_projection = dot / nuisance_energy
    prediction_energy = np.sum(prediction_change**2, axis=1)
    cosine_denominator = np.sqrt(prediction_energy * nuisance_energy)
    per_pair_cosine = np.divide(
        dot,
        cosine_denominator,
        out=np.zeros_like(dot),
        where=cosine_denominator > 1e-12,
    )
    subject_labels, subject_projection = _subject_equal_means(
        per_pair_projection, subjects, seeds
    )
    _, subject_cosine = _subject_equal_means(per_pair_cosine, subjects, seeds)
    return PairedSwapResult(
        per_pair_projection=per_pair_projection,
        per_pair_cosine=per_pair_cosine,
        subject_labels=subject_labels,
        subject_projection=subject_projection,
        subject_cosine=subject_cosine,
        mean_projection=float(subject_projection.mean()),
        mean_cosine=float(subject_cosine.mean()),
        n_subjects=int(subject_labels.size),
        n_seeds=None if seeds is None else int(np.unique(seeds).size),
    )


def _simple_slope(dose: np.ndarray, effect: np.ndarray) -> float:
    centered = dose - dose.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-12:
        raise ValueError("each subject/seed cell must contain at least two distinct doses")
    return float((centered @ (effect - effect.mean())) / denominator)


def dose_response_slope(
    dose: np.ndarray | Sequence[float],
    effect: np.ndarray | Sequence[float],
    *,
    subject_ids: np.ndarray | Sequence[object],
    seed_ids: np.ndarray | Sequence[object] | None = None,
) -> DoseResponseResult:
    """Estimate within-subject dose slopes without counting seeds as subjects.

    When seeds are supplied, a slope is fit separately for every subject/seed
    cell. Seed slopes are averaged within subject, and subject slopes are then
    equally averaged. Thus three seeds for four subjects still yield four
    biological units, not twelve.
    """

    doses = _finite_vector("dose", dose)
    effects = _finite_vector("effect", effect)
    if doses.shape != effects.shape:
        raise ValueError(f"dose and effect shapes differ: {doses.shape} versus {effects.shape}")
    subjects, seeds = _validate_subject_and_seed_ids(subject_ids, seed_ids, doses.size)
    subject_labels = np.unique(subjects)
    subject_slopes = np.empty(subject_labels.size, dtype=np.float64)
    for index, subject in enumerate(subject_labels):
        selected_subject = subjects == subject
        if seeds is None:
            subject_slopes[index] = _simple_slope(
                doses[selected_subject], effects[selected_subject]
            )
            continue
        seed_slopes = []
        for seed in np.unique(seeds[selected_subject]):
            cell = selected_subject & (seeds == seed)
            seed_slopes.append(_simple_slope(doses[cell], effects[cell]))
        subject_slopes[index] = float(np.mean(seed_slopes))

    return DoseResponseResult(
        subject_labels=subject_labels,
        subject_slopes=subject_slopes,
        mean_slope=float(subject_slopes.mean()),
        n_subjects=int(subject_labels.size),
        n_seeds=None if seeds is None else int(np.unique(seeds).size),
    )


__all__ = [
    "ConditionalProbeResult",
    "DoseResponseResult",
    "PairedSwapResult",
    "conditional_nuisance_probe",
    "dose_response_slope",
    "nuisance_projection_strength",
    "orthogonal_subspace_ablation",
    "paired_nuisance_swap_effect",
]
