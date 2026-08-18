from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from spfcl.config import load_yaml


def test_load_yaml_expands_nested_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "remote-data"
    save_root = tmp_path / "remote-output"
    monkeypatch.setenv("SPFCL_TEST_DATA", os.fspath(data_root))
    monkeypatch.setenv("SPFCL_TEST_SAVE", os.fspath(save_root))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  root: ${SPFCL_TEST_DATA}
  subjects: [1, 2, 3, 4]
outputs:
  roots:
    - ${SPFCL_TEST_SAVE}/runs
nuisance:
  amplitude: 0.23
""".lstrip(),
        encoding="utf8",
    )

    config = load_yaml(config_path)

    assert config["data"]["root"] == os.fspath(data_root)
    assert config["outputs"]["roots"] == [os.fspath(save_root / "runs")]
    assert config["data"]["subjects"] == [1, 2, 3, 4]
    assert config["nuisance"]["amplitude"] == pytest.approx(0.23)


def test_load_yaml_rejects_missing_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SPFCL_CONFIG_VALUE_THAT_MUST_NOT_EXIST", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "root: ${SPFCL_CONFIG_VALUE_THAT_MUST_NOT_EXIST}\n", encoding="utf8"
    )

    with pytest.raises(RuntimeError, match="SPFCL_CONFIG_VALUE_THAT_MUST_NOT_EXIST"):
        load_yaml(config_path)


@pytest.mark.parametrize("contents", ["- one\n- two\n", "null\n", "plain-string\n"])
def test_load_yaml_requires_a_top_level_mapping(tmp_path: Path, contents: str) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(contents, encoding="utf8")

    with pytest.raises(TypeError, match="Expected a mapping"):
        load_yaml(config_path)


def test_load_yaml_uses_safe_loader(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-be-created"
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(
        "value: !!python/object/apply:pathlib.Path.touch\n"
        f"  - {marker}\n",
        encoding="utf8",
    )

    with pytest.raises(yaml.constructor.ConstructorError):
        load_yaml(config_path)
    assert not marker.exists()


def test_import_has_no_environment_or_filesystem_side_effects(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("DATAPATH", None)
    env.pop("SAVEPATH", None)
    source_root = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = os.fspath(source_root)

    completed = subprocess.run(
        [sys.executable, "-c", "import spfcl; import spfcl.config"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []

