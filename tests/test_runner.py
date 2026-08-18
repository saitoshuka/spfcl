from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spfcl.train.runner import REPO_ROOT, _inspect_prediction_archive, run_condition
from spfcl.train.tribe_experiment import _session_by_timeline


@pytest.fixture
def remote_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAPATH", str(tmp_path / "data"))
    monkeypatch.setenv("SPFCL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SPFCL_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("SAVEPATH", str(tmp_path / "outputs"))


@pytest.mark.parametrize(
    "condition,stage_ids",
    [
        ("a_to_b_naive", ["A", "B0", "B_lambda"]),
        ("b_to_a_naive", ["B0", "B0_to_A", "B_lambda", "B_lambda_to_A"]),
        ("offline_joint", ["joint_B0", "joint_B_lambda"]),
        ("a_to_b_replay_1pct", ["A", "B0_replay", "B_lambda_replay"]),
    ],
)
def test_training_matrix_dry_run_needs_no_gpu_or_data(
    remote_env: None, condition: str, stage_ids: list[str]
) -> None:
    result = run_condition(
        REPO_ROOT / "configs" / "training" / "phase1_causal12.yaml",
        condition=condition,
        seed=17,
        dry_run=True,
    )
    assert [stage["id"] for stage in result["stages"]] == stage_ids
    assert result["dry_run"] is True
    assert not any(item["exists"] for item in result["required_paths"])


def test_unregistered_seed_or_condition_is_rejected(remote_env: None) -> None:
    config = REPO_ROOT / "configs" / "training" / "phase1_causal12.yaml"
    with pytest.raises(ValueError, match="not preregistered"):
        run_condition(config, condition="a_to_b_naive", seed=999, dry_run=True)
    with pytest.raises(ValueError, match="absent"):
        run_condition(config, condition="invented", seed=17, dry_run=True)


def test_reuse_rejects_prediction_shape_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "clean_test_predictions.npz"
    np.savez_compressed(
        archive,
        encoding_empirical=np.zeros((2, 360, 100)),
        encoding_predicted=np.zeros((2, 360, 99)),
        group_predicted=np.zeros((2, 360, 100)),
        subject_id=np.asarray([0, 1]),
        segment_uid=np.asarray(["one", "two"]),
    )
    with pytest.raises(RuntimeError, match="must share shape"):
        _inspect_prediction_archive(
            archive,
            {
                "encoding_empirical",
                "encoding_predicted",
                "group_predicted",
                "subject_id",
                "segment_uid",
            },
        )


def test_replay_session_mapping_ignores_dummy_event() -> None:
    pd = pytest.importorskip("pandas")
    events = pd.DataFrame(
        {
            "timeline": ["old", "old", "new"],
            "session": ["", 2, 3],
            "type": ["CategoricalEvent", "Fmri", "Fmri"],
        }
    )
    assert _session_by_timeline(events) == {"old": 2, "new": 3}
