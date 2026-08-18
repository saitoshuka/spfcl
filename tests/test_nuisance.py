from __future__ import annotations

import numpy as np
import pytest

from spfcl.data.nuisance import DriftConfig, LowFrequencyDriftInjector


@pytest.fixture(scope="module")
def spatial_basis() -> np.ndarray:
    parcels = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    return np.stack(
        [
            np.sin(parcels),
            np.cos(parcels),
            np.sin(2.0 * parcels),
            np.cos(2.0 * parcels),
        ],
        axis=1,
    ).astype(np.float32)


@pytest.fixture(scope="module")
def injector(spatial_basis: np.ndarray) -> LowFrequencyDriftInjector:
    return LowFrequencyDriftInjector(
        DriftConfig(
            amplitude=0.23,
            chunk_seconds=100,
            sample_hz=1.0,
            cycles=(1, 2),
            temporal_jitter=0.3,
            spatial_jitter=0.25,
            seed=17,
        ),
        spatial_basis,
    )


def test_injection_is_deterministic_per_sample_and_order_independent(
    injector: LowFrequencyDriftInjector,
) -> None:
    clean = np.zeros((24, 100), dtype=np.float32)
    first_a = injector.apply(clean, sample_uid="subject-01/run-01/window-000")
    first_b = injector.apply(clean, sample_uid="subject-02/run-01/window-000")

    # Generate the same samples in the opposite order. The implementation must
    # derive each RNG stream from the stable sample key, not from global RNG order.
    second_b = injector.apply(clean, sample_uid="subject-02/run-01/window-000")
    second_a = injector.apply(clean, sample_uid="subject-01/run-01/window-000")

    assert np.array_equal(first_a.target, second_a.target)
    assert np.array_equal(first_b.target, second_b.target)
    assert not np.array_equal(first_a.nuisance, first_b.nuisance)


def test_lambda_zero_is_bitwise_clean_and_does_not_mutate_source(
    spatial_basis: np.ndarray,
) -> None:
    rng = np.random.default_rng(3)
    clean = rng.standard_normal((24, 100), dtype=np.float32)
    original_bytes = clean.tobytes()
    zero = LowFrequencyDriftInjector(
        DriftConfig(amplitude=0.0, seed=3), spatial_basis
    ).apply(clean, sample_uid="B0")

    assert zero.target.tobytes() == original_bytes
    assert clean.tobytes() == original_bytes
    assert zero.target is not clean
    assert np.count_nonzero(zero.nuisance) == 0
    assert np.count_nonzero(zero.temporal) == 0
    assert np.count_nonzero(zero.spatial) == 0


def test_nuisance_is_the_documented_spatial_temporal_outer_product(
    injector: LowFrequencyDriftInjector,
) -> None:
    clean = np.zeros((24, 100), dtype=np.float32)
    result = injector.apply(clean, sample_uid="outer-product")

    expected = result.spatial[:, None] * result.temporal[None, :]
    np.testing.assert_allclose(result.nuisance, expected, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(
        result.target - clean,
        injector.config.amplitude * result.nuisance,
        atol=1e-7,
        rtol=1e-6,
    )
    assert result.metadata["sample_uid"] == "outer-product"
    assert result.metadata["target_shape"] == [24, 100]
    assert result.metadata["realized_nuisance_rms"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("component_name", ["temporal", "spatial"])
def test_components_are_centered_and_unit_rms(
    injector: LowFrequencyDriftInjector, component_name: str
) -> None:
    result = injector.apply(
        np.zeros((24, 100), dtype=np.float32), sample_uid="normalization"
    )
    component = np.asarray(getattr(result, component_name), dtype=np.float64)
    time = np.linspace(-1.0, 1.0, component.size)

    assert component.mean() == pytest.approx(0.0, abs=1e-6)
    assert np.sqrt(np.mean(component**2)) == pytest.approx(1.0, abs=1e-6)
    # Linear trend is meaningful for time, but parcel ordering is arbitrary;
    # spatial graph modes should therefore only be centered/RMS-normalized.
    if component_name == "temporal":
        assert float(component @ time) == pytest.approx(0.0, abs=1e-5)


def test_temporal_component_contains_the_preregistered_low_frequency_peak(
    injector: LowFrequencyDriftInjector,
) -> None:
    result = injector.apply(
        np.zeros((24, 100), dtype=np.float32), sample_uid="spectrum"
    )
    frequencies = np.fft.rfftfreq(result.temporal.size, d=1.0)
    power = np.abs(np.fft.rfft(result.temporal)) ** 2
    dominant_non_dc = frequencies[1 + int(np.argmax(power[1:]))]

    # Linear detrending leaks some energy into neighboring bins, so the robust
    # contract is that the dominant peak remains one of the configured cycles.
    assert any(abs(dominant_non_dc - expected) <= 1e-12 for expected in (0.01, 0.02))


def test_partial_final_chunk_has_the_requested_shape(
    injector: LowFrequencyDriftInjector,
) -> None:
    result = injector.apply(
        np.zeros((24, 250), dtype=np.float32), sample_uid="two-and-a-half-chunks"
    )
    assert result.target.shape == (24, 250)
    assert result.temporal.shape == (250,)
    assert result.spatial.shape == (24,)
    assert np.isfinite(result.target).all()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"amplitude": -0.1}, "amplitude"),
        ({"chunk_seconds": 0}, "positive"),
        ({"sample_hz": 0}, "positive"),
        ({"cycles": ()}, "cycles"),
        ({"cycles": (0, 1)}, "cycles"),
        ({"temporal_jitter": 1.0}, "jitter"),
        ({"spatial_jitter": -0.1}, "jitter"),
        ({"dose_levels": (0.1, -0.2)}, "dose_levels"),
        ({"phase_offset_seconds": -1.0}, "phase_offset_seconds"),
    ],
)
def test_invalid_drift_config_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DriftConfig(**kwargs)


def test_non_integral_samples_per_chunk_is_rejected() -> None:
    config = DriftConfig(chunk_seconds=3, sample_hz=0.5)
    with pytest.raises(ValueError, match="must be an integer"):
        _ = config.samples_per_chunk

    phase = DriftConfig(sample_hz=2.0, phase_offset_seconds=0.25)
    with pytest.raises(ValueError, match="phase_offset_seconds"):
        _ = phase.phase_offset_samples


def test_bad_target_or_basis_shape_is_rejected(spatial_basis: np.ndarray) -> None:
    with pytest.raises(ValueError, match="spatial_basis"):
        LowFrequencyDriftInjector(DriftConfig(), np.ones(24, dtype=np.float32))

    injector = LowFrequencyDriftInjector(DriftConfig(), spatial_basis)
    with pytest.raises(ValueError, match="parcel, time"):
        injector.apply(np.zeros((2, 24, 100), dtype=np.float32), sample_uid="bad")
    with pytest.raises(ValueError, match="basis has 24"):
        injector.apply(np.zeros((23, 100), dtype=np.float32), sample_uid="bad")


def test_preregistered_dose_levels_are_balanced_within_each_chunk(
    spatial_basis: np.ndarray,
) -> None:
    injector = LowFrequencyDriftInjector(
        DriftConfig(amplitude=0.23, dose_levels=(0.0, 0.1, 0.3), seed=21),
        spatial_basis,
    )
    clean = np.zeros((24, 300), dtype=np.float32)
    result = injector.apply(clean, sample_uid="dose-schedule")
    for start in (0, 100, 200):
        values, counts = np.unique(
            result.amplitude_trace[start : start + 100], return_counts=True
        )
        np.testing.assert_allclose(values, [0.0, 0.1, 0.3], atol=1e-7)
        assert counts.max() - counts.min() <= 1
    np.testing.assert_allclose(
        result.target,
        result.nuisance * result.amplitude_trace[None, :],
        atol=1e-7,
        rtol=1e-6,
    )


def test_hrf_offset_windows_are_complete_phase_aligned_chunks(
    spatial_basis: np.ndarray,
) -> None:
    injector = LowFrequencyDriftInjector(
        DriftConfig(
            amplitude=0.23,
            dose_levels=(0.10, 0.23, 0.36),
            phase_offset_seconds=5.0,
            seed=22,
        ),
        spatial_basis,
    )
    result = injector.apply(
        np.zeros((24, 417), dtype=np.float32), sample_uid="run-with-five-second-offset"
    )
    for start in (5, 105, 205, 305):
        window = result.amplitude_trace[start : start + 100]
        values, counts = np.unique(window, return_counts=True)
        np.testing.assert_allclose(values, [0.10, 0.23, 0.36], atol=1e-7)
        assert counts.max() - counts.min() <= 1
        # Canonical chunks are independently generated but each has unit RMS;
        # no short-tail renormalisation is present in a training window.
        assert np.sqrt(np.mean(result.temporal[start : start + 100] ** 2)) == pytest.approx(
            1.0, abs=1e-6
        )
