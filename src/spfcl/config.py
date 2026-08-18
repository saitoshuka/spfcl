from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_UNEXPANDED_ENV = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        missing = _UNEXPANDED_ENV.findall(expanded)
        if missing:
            names = ", ".join(sorted(set(missing)))
            raise RuntimeError(f"Missing environment variable(s) required by config: {names}")
        return expanded
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and expand ``${ENV}`` placeholders strictly."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {config_path}, got {type(value).__name__}")
    return _expand(value)

