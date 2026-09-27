"""Small JSON configuration loader for SMAT experiments.

Configurations remain ordinary JSON files.  ``extends`` is intentionally the
only composition feature: it is enough for the benchmark/run split without
adding a second configuration language or a runtime dependency.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings and replace all other values.

    Lists are deliberately replaced rather than appended.  A run config should
    be able to select its task suite exactly, rather than inherit part of one.
    """

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a resolved SMAT config, recursively applying relative ``extends``."""

    return _load(Path(os.path.expandvars(str(path))).expanduser().resolve(), seen=())


def _expand_values(value: Any) -> Any:
    """Expand portable path notation without inventing a shell language."""

    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if re.search(r"\$\{?[A-Za-z_][A-Za-z_0-9]*", expanded):
            raise ValueError(f"unresolved environment variable in config: {value}")
        return expanded
    if isinstance(value, Mapping):
        return {key: _expand_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_values(item) for item in value]
    return value


def _load(path: Path, *, seen: tuple[Path, ...]) -> dict[str, Any]:
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"cyclic config extends: {chain}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"configuration not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration: {path}: {error}") from error
    if not isinstance(data, Mapping):
        raise ValueError(f"configuration must be a JSON object: {path}")

    extends = data.get("extends")
    if extends is None:
        result = _merge({}, data)
    elif isinstance(extends, str):
        parent = Path(os.path.expanduser(os.path.expandvars(extends)))
        result = _merge(
            _load((path.parent / parent).resolve(), seen=(*seen, path)), data
        )
    else:
        raise ValueError(f"config extends must be a path string: {path}")
    result = _expand_values(result)
    if not isinstance(result.get("experiment"), str) or not result["experiment"]:
        raise ValueError(f"configuration must declare a non-empty experiment: {path}")
    result["_config_path"] = str(path)
    return result
