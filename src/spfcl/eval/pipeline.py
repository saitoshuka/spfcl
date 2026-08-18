from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import encoding_metrics, localizer_contrast_metrics


def evaluate_npz_bundle(bundle_path: str | Path, output: str | Path) -> Path:
    """Evaluate a checkpoint-export bundle without loading the training framework.

    A bundle may contain encoding arrays, localizer arrays, or both. Localizer
    stage artifacts with named conditions score both preregistered contrasts.
    """

    with np.load(bundle_path, allow_pickle=False) as bundle:
        has_encoding = {"encoding_empirical", "encoding_predicted"}.issubset(bundle.files)
        has_localizer = {"localizer_empirical", "localizer_predicted"}.issubset(
            bundle.files
        )
        if not has_encoding and not has_localizer:
            raise ValueError("Bundle contains neither encoding nor localizer prediction arrays")
        result: dict[str, Any] = {}
        if has_encoding:
            metrics = encoding_metrics(
                bundle["encoding_empirical"], bundle["encoding_predicted"]
            )
            result["encoding"] = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in metrics.items()
            }
            # Retain the original top-level metrics for older analysis scripts.
            result.update(result["encoding"])
            if "group_predicted" in bundle.files:
                group_metrics = encoding_metrics(
                    bundle["encoding_empirical"], bundle["group_predicted"]
                )
                result["group_head"] = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in group_metrics.items()
                }
        if has_localizer:
            if "conditions" in bundle.files:
                names = [str(value) for value in bundle["conditions"].tolist()]
                requested = (
                    ("faces_vs_bodies", "faces", "bodies"),
                    ("scenes_vs_objects", "scenes", "objects"),
                )
                contrasts = [
                    (name, names.index(positive), names.index(negative))
                    for name, positive, negative in requested
                ]
            else:
                contrasts = [
                    (
                        "configured_contrast",
                        int(bundle["positive_condition"]),
                        int(bundle["negative_condition"]),
                    )
                ]

            def score(predicted: np.ndarray) -> dict[str, Any]:
                output_metrics = {}
                for name, positive, negative in contrasts:
                    values = localizer_contrast_metrics(
                        bundle["localizer_empirical"],
                        predicted,
                        positive=positive,
                        negative=negative,
                    )
                    output_metrics[name] = {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in values.items()
                    }
                return output_metrics

            result["localizer_contrasts"] = score(bundle["localizer_predicted"])
            # Backward-compatible alias for the first requested contrast.
            first_name = contrasts[0][0]
            result["localizer"] = result["localizer_contrasts"][first_name]
            if "localizer_group_predicted" in bundle.files:
                result["localizer_group_head"] = score(
                    bundle["localizer_group_predicted"]
                )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return path
