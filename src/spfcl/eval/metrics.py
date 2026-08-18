from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _as_float_pair(empirical: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(empirical, dtype=np.float64)
    second = np.asarray(predicted, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: empirical={first.shape}, predicted={second.shape}")
    return first, second


def pearson_last_axis(empirical: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    empirical, predicted = _as_float_pair(empirical, predicted)
    first = empirical - empirical.mean(axis=-1, keepdims=True)
    second = predicted - predicted.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(first**2, axis=-1) * np.sum(second**2, axis=-1))
    numerator = np.sum(first * second, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def encoding_metrics(empirical: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray | float]:
    """Parcel-wise time-series metrics for arrays shaped ``(..., parcel, time)``."""

    empirical, predicted = _as_float_pair(empirical, predicted)
    if empirical.ndim < 2:
        raise ValueError("Encoding arrays must include parcel and time axes")
    correlation = pearson_last_axis(empirical, predicted)
    clipped = np.clip(correlation, -1 + 1e-7, 1 - 1e-7)
    fisher = np.arctanh(clipped)
    squared_error = (predicted - empirical) ** 2
    mse = squared_error.mean(axis=-1)
    residual = squared_error.sum(axis=-1)
    total = ((empirical - empirical.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    r2 = 1 - np.divide(
        residual,
        total,
        out=np.full_like(residual, np.nan),
        where=total > 0,
    )
    return {
        "parcel_pearson": correlation,
        "parcel_fisher_z": fisher,
        "parcel_mse": mse,
        "parcel_r2": r2,
        "mean_fisher_z": float(np.nanmean(fisher)),
        "mean_pearson_from_z": float(np.tanh(np.nanmean(fisher))),
        "mean_mse": float(np.nanmean(mse)),
        "mean_r2": float(np.nanmean(r2)),
    }


def forgetting(scores: Iterable[float]) -> float:
    """``max_t score(t) - score(T)`` for a fixed task/probe."""

    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite one-dimensional sequence")
    return float(values.max() - values[-1])


def _cohen_dz(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    standard_deviation = value.std(ddof=1)
    return float(value.mean() / standard_deviation) if standard_deviation > 0 else float("nan")


def localizer_contrast_metrics(
    empirical_conditions: np.ndarray,
    predicted_conditions: np.ndarray,
    *,
    positive: int,
    negative: int,
    left_roi: np.ndarray | None = None,
    right_roi: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """Compare empirical/predicted localizer contrasts.

    Inputs are ``(..., condition, parcel)``. The leading axes are normally subject.
    No training data or IBC map is used: the primary Phase-1 probe is the same-subject
    BOLD Moments silent-video localizer.
    """

    empirical, predicted = _as_float_pair(empirical_conditions, predicted_conditions)
    if empirical.ndim < 2:
        raise ValueError("Localizer arrays must include condition and parcel axes")
    empirical_contrast = empirical[..., positive, :] - empirical[..., negative, :]
    predicted_contrast = predicted[..., positive, :] - predicted[..., negative, :]
    map_correlation = pearson_last_axis(empirical_contrast, predicted_contrast)
    result: dict[str, float | np.ndarray] = {
        "contrast_map_correlation": map_correlation,
        "mean_contrast_map_correlation": float(np.nanmean(map_correlation)),
        "contrast_effect_mae": float(np.mean(np.abs(predicted_contrast - empirical_contrast))),
        "contrast_effect_rmse": float(
            np.sqrt(np.mean((predicted_contrast - empirical_contrast) ** 2))
        ),
    }
    if left_roi is not None or right_roi is not None:
        if left_roi is None or right_roi is None:
            raise ValueError("Provide both left_roi and right_roi")
        left = np.asarray(left_roi, dtype=bool)
        right = np.asarray(right_roi, dtype=bool)
        if left.shape != (empirical.shape[-1],) or right.shape != left.shape:
            raise ValueError("ROI masks must match the parcel axis")
        empirical_lateralization = empirical_contrast[..., left].mean(axis=-1) - empirical_contrast[
            ..., right
        ].mean(axis=-1)
        predicted_lateralization = predicted_contrast[..., left].mean(axis=-1) - predicted_contrast[
            ..., right
        ].mean(axis=-1)
        result["lateralization_error"] = float(
            np.mean(np.abs(predicted_lateralization - empirical_lateralization))
        )
        result["predicted_lateralization"] = predicted_lateralization
        result["empirical_lateralization"] = empirical_lateralization
    return result


def exact_two_sided_sign_flip_paired(differences: np.ndarray) -> float:
    """Exact subject-level paired randomization p-value (seeds are not samples)."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a non-empty finite vector")
    observed = abs(float(values.mean()))
    statistics = []
    for code in range(1 << len(values)):
        signs = np.asarray([1 if code & (1 << index) else -1 for index in range(len(values))])
        statistics.append(abs(float(np.mean(signs * values))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))

