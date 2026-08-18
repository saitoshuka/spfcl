from __future__ import annotations

import numpy as np
import pytest

from spfcl.eval.metrics import (
    encoding_metrics,
    exact_two_sided_sign_flip_paired,
    forgetting,
    localizer_contrast_metrics,
)


def test_encoding_metrics_and_forgetting() -> None:
    empirical = np.asarray([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]])
    predicted = empirical.copy()
    result = encoding_metrics(empirical, predicted)
    np.testing.assert_allclose(result["parcel_pearson"], 1.0)
    assert result["mean_mse"] == pytest.approx(0.0)
    assert result["mean_r2"] == pytest.approx(1.0)
    assert forgetting([0.2, 0.5, 0.3]) == pytest.approx(0.2)


def test_localizer_contrast_map_fidelity() -> None:
    conditions = np.asarray(
        [
            [[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 2.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    result = localizer_contrast_metrics(
        conditions,
        conditions,
        positive=0,
        negative=1,
        left_roi=np.asarray([True, True, False, False]),
        right_roi=np.asarray([False, False, True, True]),
    )
    np.testing.assert_allclose(result["contrast_map_correlation"], 1.0)
    assert result["contrast_effect_mae"] == pytest.approx(0.0)
    assert result["lateralization_error"] == pytest.approx(0.0)


def test_four_subject_exact_sign_flip_resolution_is_one_eighth() -> None:
    assert exact_two_sided_sign_flip_paired(np.ones(4)) == pytest.approx(0.125)

