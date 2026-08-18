from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _stable_seed(base_seed: int, sample_uid: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_uid}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _detrend_and_rms(value: np.ndarray, axis: int = -1) -> np.ndarray:
    data = np.asarray(value, dtype=np.float64)
    moved = np.moveaxis(data, axis, -1)
    time = np.linspace(-1.0, 1.0, moved.shape[-1])
    design = np.stack([np.ones_like(time), time], axis=1)
    flat = moved.reshape(-1, moved.shape[-1])
    coefficients = np.linalg.lstsq(design, flat.T, rcond=None)[0]
    flat = flat - (design @ coefficients).T
    rms = np.sqrt(np.mean(flat**2, axis=1, keepdims=True))
    if np.any(rms < 1e-12):
        raise ValueError("Cannot normalize a constant nuisance component")
    normalized = (flat / rms).reshape(moved.shape)
    return np.moveaxis(normalized, -1, axis).astype(np.float32)


def _center_and_rms(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float64)
    data = data - data.mean()
    rms = float(np.sqrt(np.mean(data**2)))
    if rms < 1e-12:
        raise ValueError("Cannot normalize a constant nuisance component")
    return (data / rms).astype(np.float32)


@dataclass(frozen=True)
class DriftConfig:
    amplitude: float = 0.23
    chunk_seconds: int = 100
    sample_hz: float = 1.0
    cycles: tuple[int, ...] = (1, 2)
    temporal_jitter: float = 0.3
    spatial_jitter: float = 0.25
    seed: int = 20260818
    dose_levels: tuple[float, ...] = ()
    phase_offset_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.amplitude < 0:
            raise ValueError("Nuisance amplitude must be non-negative")
        if self.chunk_seconds <= 0 or self.sample_hz <= 0:
            raise ValueError("chunk_seconds and sample_hz must be positive")
        if not self.cycles or any(cycle <= 0 for cycle in self.cycles):
            raise ValueError("cycles must contain positive integers")
        if not 0 <= self.temporal_jitter < 1 or not 0 <= self.spatial_jitter < 1:
            raise ValueError("jitter fractions must lie in [0, 1)")
        if any(level < 0 for level in self.dose_levels):
            raise ValueError("dose_levels must be non-negative")
        if self.phase_offset_seconds < 0:
            raise ValueError("phase_offset_seconds must be non-negative")

    @property
    def samples_per_chunk(self) -> int:
        value = self.chunk_seconds * self.sample_hz
        if not float(value).is_integer():
            raise ValueError("chunk_seconds * sample_hz must be an integer")
        return int(value)

    @property
    def phase_offset_samples(self) -> int:
        value = self.phase_offset_seconds * self.sample_hz
        if not float(value).is_integer():
            raise ValueError("phase_offset_seconds * sample_hz must be an integer")
        return int(value) % self.samples_per_chunk


@dataclass(frozen=True)
class InjectionResult:
    target: np.ndarray
    nuisance: np.ndarray
    temporal: np.ndarray
    spatial: np.ndarray
    amplitude_trace: np.ndarray
    metadata: dict


class LowFrequencyDriftInjector:
    """Deterministic shared-plus-jitter drift for parcel x time targets."""

    def __init__(self, config: DriftConfig, spatial_basis: np.ndarray):
        self.config = config
        basis = np.asarray(spatial_basis, dtype=np.float32)
        if basis.ndim != 2 or basis.shape[1] < 2:
            raise ValueError("spatial_basis must have shape (n_parcels, at least 2 modes)")
        basis = basis - basis.mean(axis=0, keepdims=True)
        basis = basis / np.sqrt(np.mean(basis**2, axis=0, keepdims=True))
        self.spatial_basis = basis

        shared_coefficients = np.linspace(1.0, 0.35, min(4, basis.shape[1]))
        self.shared_spatial = _center_and_rms(
            basis[:, : len(shared_coefficients)] @ shared_coefficients
        )

    def _temporal(self, n_time: int, rng: np.random.Generator) -> np.ndarray:
        n_chunk = self.config.samples_per_chunk
        shared_coefficients = np.linspace(1.0, 0.5, len(self.config.cycles))
        phase = self.config.phase_offset_samples
        output = np.empty(n_time, dtype=np.float32)
        # Generate full canonical chunks even for the leading/trailing overlap.
        # This avoids renormalising a short run tail and makes each extracted
        # 100-second segment identical in phase when its target starts at `phase`.
        first_start = phase if phase == 0 else phase - n_chunk
        for start in range(first_start, n_time, n_chunk):
            tau = np.arange(n_chunk, dtype=np.float64) / self.config.sample_hz
            shared = np.zeros(n_chunk, dtype=np.float64)
            random_component = np.zeros(n_chunk, dtype=np.float64)
            for coefficient, cycle in zip(shared_coefficients, self.config.cycles):
                frequency = cycle / self.config.chunk_seconds
                shared += coefficient * np.sin(2 * np.pi * frequency * tau)
                phase = rng.uniform(0, 2 * np.pi)
                random_component += rng.normal() * np.sin(2 * np.pi * frequency * tau + phase)
            shared = _detrend_and_rms(shared, axis=0)
            random_component = _detrend_and_rms(random_component, axis=0)
            mixed = (
                np.sqrt(1 - self.config.temporal_jitter**2) * shared
                + self.config.temporal_jitter * random_component
            )
            canonical = _detrend_and_rms(mixed, axis=0)
            source_start = max(0, -start)
            source_stop = min(n_chunk, n_time - start)
            if source_stop > source_start:
                destination_start = start + source_start
                destination_stop = start + source_stop
                output[destination_start:destination_stop] = canonical[
                    source_start:source_stop
                ]
        return output

    def _spatial(self, rng: np.random.Generator) -> np.ndarray:
        random_coefficients = rng.normal(size=self.spatial_basis.shape[1])
        random_component = _center_and_rms(self.spatial_basis @ random_coefficients)
        mixed = (
            np.sqrt(1 - self.config.spatial_jitter**2) * self.shared_spatial
            + self.config.spatial_jitter * random_component
        )
        return _center_and_rms(mixed)

    def _amplitude_trace(self, n_time: int, sample_uid: str) -> np.ndarray:
        levels = self.config.dose_levels or (self.config.amplitude,)
        chunk = self.config.samples_per_chunk
        phase = self.config.phase_offset_samples
        relative = (np.arange(n_time, dtype=np.int64) - phase) % chunk
        boundaries = np.linspace(0, chunk, len(levels) + 1, dtype=int)
        indices = np.searchsorted(boundaries[1:], relative, side="right")
        return np.asarray(levels, dtype=np.float32)[indices]

    def apply(self, clean: np.ndarray, *, sample_uid: str) -> InjectionResult:
        clean_value = np.asarray(clean)
        if clean_value.ndim != 2:
            raise ValueError(f"Expected clean target shaped (parcel, time), got {clean_value.shape}")
        if clean_value.shape[0] != self.spatial_basis.shape[0]:
            raise ValueError(
                f"Target has {clean_value.shape[0]} parcels but basis has {self.spatial_basis.shape[0]}"
            )
        amplitude_trace = self._amplitude_trace(clean_value.shape[1], sample_uid)
        if not np.any(amplitude_trace):
            zero_temporal = np.zeros(clean_value.shape[1], dtype=np.float32)
            zero_spatial = np.zeros(clean_value.shape[0], dtype=np.float32)
            zero_nuisance = np.zeros_like(clean_value, dtype=np.float32)
            return InjectionResult(
                target=clean_value.copy(),
                nuisance=zero_nuisance,
                temporal=zero_temporal,
                spatial=zero_spatial,
                amplitude_trace=amplitude_trace,
                metadata={"sample_uid": sample_uid, **asdict(self.config)},
            )

        rng = np.random.default_rng(_stable_seed(self.config.seed, sample_uid))
        temporal = self._temporal(clean_value.shape[1], rng)
        spatial = self._spatial(rng)
        nuisance = spatial[:, None] * temporal[None, :]
        target = clean_value + (
            nuisance * amplitude_trace[None, :]
        ).astype(clean_value.dtype)
        metadata = {
            "sample_uid": sample_uid,
            "config": asdict(self.config),
            "realized_nuisance_rms": float(np.sqrt(np.mean(nuisance**2))),
            "target_shape": list(target.shape),
            "realized_dose_levels": sorted({float(value) for value in amplitude_trace}),
        }
        return InjectionResult(
            target=target,
            nuisance=nuisance.astype(np.float32),
            temporal=temporal,
            spatial=spatial,
            amplitude_trace=amplitude_trace,
            metadata=metadata,
        )


def load_basis(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.load(source, allow_pickle=False)
    with np.load(source, allow_pickle=False) as bundle:
        key = "basis" if "basis" in bundle.files else bundle.files[0]
        return np.asarray(bundle[key])


def save_preview(
    output: str | Path,
    *,
    config: DriftConfig,
    spatial_basis: np.ndarray,
    n_time: int = 100,
    sample_uid: str = "preview/sub-01/ses-03/run-01/window-000",
) -> Path:
    clean = np.zeros((spatial_basis.shape[0], n_time), dtype=np.float32)
    result = LowFrequencyDriftInjector(config, spatial_basis).apply(clean, sample_uid=sample_uid)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        target=result.target,
        nuisance=result.nuisance,
        temporal=result.temporal,
        spatial=result.spatial,
        amplitude_trace=result.amplitude_trace,
        metadata=np.asarray(json.dumps(result.metadata, sort_keys=True)),
    )
    return output_path


# Register the post-cleaning injector in the full TRIBE environment.
try:  # pragma: no cover - remote integration path
    import pydantic
    from neuralset.base import TimedArray
    from neuralset.extractors.neuro import FmriExtractor

    class NuisanceFmriExtractor(FmriExtractor):
        nuisance_amplitude: float = 0.0
        nuisance_seed: int = 20260818
        nuisance_basis_path: Path | None = None
        nuisance_chunk_seconds: int = 100
        nuisance_cycles: tuple[int, ...] = (1, 2)
        nuisance_temporal_jitter: float = 0.3
        nuisance_spatial_jitter: float = 0.25
        nuisance_dose_levels: tuple[float, ...] = ()
        nuisance_phase_offset_seconds: float | None = None
        nuisance_train_only: bool = True
        nuisance_sessions: tuple[int, ...] = ()
        _injector: LowFrequencyDriftInjector | None = pydantic.PrivateAttr(default=None)

        def _get_injector(self) -> LowFrequencyDriftInjector | None:
            if self.nuisance_amplitude == 0:
                return None
            if self.nuisance_basis_path is None:
                raise ValueError("nuisance_basis_path is required when nuisance_amplitude > 0")
            if self._injector is None:
                config = DriftConfig(
                    amplitude=self.nuisance_amplitude,
                    chunk_seconds=self.nuisance_chunk_seconds,
                    sample_hz=float(self.frequency),
                    cycles=self.nuisance_cycles,
                    temporal_jitter=self.nuisance_temporal_jitter,
                    spatial_jitter=self.nuisance_spatial_jitter,
                    seed=self.nuisance_seed,
                    dose_levels=self.nuisance_dose_levels,
                    phase_offset_seconds=(
                        float(self.offset)
                        if self.nuisance_phase_offset_seconds is None
                        else self.nuisance_phase_offset_seconds
                    ),
                )
                self._injector = LowFrequencyDriftInjector(
                    config, load_basis(self.nuisance_basis_path)
                )
            return self._injector

        def _preprocess_event(self, event):
            clean = super()._preprocess_event(event)
            split = getattr(event, "split", None)
            if split is None and hasattr(event, "extra"):
                split = event.extra.get("split")
            if self.nuisance_train_only and split != "train":
                return clean
            session = getattr(event, "session", None)
            if session is None and hasattr(event, "extra"):
                session = event.extra.get("session")
            if self.nuisance_sessions and int(session) not in self.nuisance_sessions:
                return clean
            injector = self._get_injector()
            if injector is None:
                return clean
            sample_uid = str(event.study_relative_path())
            injected = injector.apply(clean.data, sample_uid=sample_uid)
            return TimedArray(
                data=injected.target.astype(np.float32),
                frequency=clean.frequency,
                start=clean.start,
                duration=clean.duration,
            )

except ImportError:
    NuisanceFmriExtractor = None  # type: ignore[assignment]
