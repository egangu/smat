"""Small shared run, adapter and checkpoint-location helpers."""

from __future__ import annotations

import importlib
import json
import os
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

_EXPERIMENT_MODULES = {
    "trace_llm": "smat.experiments.trace_llm",
    "clip8_vit": "smat.experiments.clip8_vit",
}


def experiment_name(config: Mapping[str, Any]) -> str:
    requested = config.get("experiment")
    if requested not in _EXPERIMENT_MODULES:
        raise ValueError(
            f"experiment must be one of {', '.join(sorted(_EXPERIMENT_MODULES))}; got {requested!r}"
        )
    return str(requested)


def experiment_module(config: Mapping[str, Any]) -> ModuleType:
    """Load one of the two supported experiment adapters lazily."""

    requested = experiment_name(config)
    module_name = _EXPERIMENT_MODULES[requested]
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise ValueError(
                f"experiment backend is unavailable: {requested}"
            ) from error
        raise


def require(module: ModuleType, name: str) -> Callable[..., Any]:
    function = getattr(module, name, None)
    if not callable(function):
        raise TypeError(f"{module.__name__} must define callable {name}(config, ...)")
    return function


def task_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(require(experiment_module(config), "task_names")(config))
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(
            "experiment task_names(config) must return nonempty task names"
        )
    return names


def run_name(config: Mapping[str, Any]) -> str:
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("config needs a nonempty name")
    return name


def work_root(config: Mapping[str, Any]) -> Path:
    value = config.get("work_root")
    if not isinstance(value, str) or not value:
        raise ValueError("config needs a nonempty work_root")
    return Path(value).expanduser()


def expert_checkpoint(config: Mapping[str, Any], task: str) -> Path:
    """Return the one canonical expert location for a task."""
    if "expert_overrides" in config:
        sources = config["expert_overrides"]
        if not isinstance(sources, Mapping) or not isinstance(sources.get(task), str):
            raise ValueError(f"expert_overrides needs an explicit path for {task}")
        return Path(sources[task]).expanduser()
    return work_root(config) / "experts" / run_name(config) / task


def checkpoint_path(
    module: ModuleType, config: Mapping[str, Any], directory: Path
) -> Path:
    """Resolve an adapter's one checkpoint payload inside its run directory."""

    selector = getattr(module, "checkpoint_path", None)
    if callable(selector):
        return Path(selector(config, directory))
    return directory


def checkpoint_adapter(module: ModuleType, config: Mapping[str, Any]):
    factory = getattr(module, "checkpoint_adapter", None)
    if callable(factory):
        return factory(config)
    from .checkpoints import HuggingFaceSafetensorsAdapter

    return HuggingFaceSafetensorsAdapter()


def base_checkpoint(config: Mapping[str, Any]) -> Path:
    base = config.get("base_model")
    if not isinstance(base, str) or not base:
        raise ValueError("a task-vector merge needs a nonempty base_model")
    return Path(base).expanduser()


def checkpoint_exists(path: Path) -> bool:
    return path.exists() and (path.is_file() or any(path.iterdir()))


def atomic_write_json(path: Path, payload: Any, *, default: Any = None) -> None:
    """Atomically replace one JSON result without exposing a partial payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_file_complete(path: Path) -> bool:
    """Return whether a result path contains one complete JSON object."""

    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            return isinstance(json.load(handle), Mapping)
    except (OSError, json.JSONDecodeError):
        return False


@contextmanager
def exclusive_directory_lock(
    directory: Path, name: str = ".eval-ta.lock"
) -> Iterator[None]:
    """Reject concurrent writers for one evaluation result directory."""

    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"evaluation output is already locked: {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_metadata(directory: Path, payload: Mapping[str, Any]) -> None:
    """Keep manifests beside, rather than inside, backend checkpoint files."""

    path = directory / "metadata.json"
    atomic_write_json(path, payload, default=str)


def partial_directory(directory: Path) -> Path:
    return directory.with_name(f".{directory.name}.partial-{uuid.uuid4().hex[:8]}")
