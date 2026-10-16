"""Configuration file support.

Loads and validates ``skala.toml``. Precedence, lowest to highest:

1. built-in defaults,
2. the config file (``skala.toml`` in the current directory, or an explicit ``--config PATH``),
3. CLI options that were explicitly passed.

Every field is optional and ``None`` means "not set", so ``cli.py`` can merge with
``cli_value if cli_value is not None else config_value``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

from skala.utils import InputError, get_logger

__all__ = [
    "AUDIO_CHOICES",
    "CONFIG_FILENAME",
    "SkalaConfig",
    "find_config_file",
    "load_config",
]

logger = get_logger("config")

CONFIG_FILENAME = "skala.toml"

#: Accepted values for the ``audio`` key (how the video pipeline handles audio).
AUDIO_CHOICES = ("copy", "drop", "aac")

_INT_FIELDS = frozenset({"crf", "gpu_id", "threads", "tile"})
_BOOL_FIELDS = frozenset({"keep_temp", "overwrite", "verbose"})
_STR_FIELDS = frozenset(
    {"audio", "engine_path", "ffmpeg_path", "ffprobe_path", "model", "video_codec", "workdir"}
)


@dataclass(frozen=True)
class SkalaConfig:
    """Validated config-file contents. ``None`` means "fall back to default"."""

    model: str | None = None
    engine_path: str | None = None
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    tile: int | None = None
    gpu_id: int | None = None
    threads: int | None = None
    workdir: str | None = None
    keep_temp: bool | None = None
    overwrite: bool | None = None
    verbose: bool | None = None
    crf: int | None = None
    video_codec: str | None = None
    audio: str | None = None


def find_config_file(cwd: Path | None = None) -> Path | None:
    """Return ``<cwd>/skala.toml`` when it exists, otherwise ``None``."""
    candidate = (cwd if cwd is not None else Path.cwd()) / CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def load_config(path: Path | None = None, *, cwd: Path | None = None) -> SkalaConfig:
    """Load a config file, or return an all-default config when none is found.

    An explicit ``path`` must exist (otherwise :class:`~skala.utils.InputError`).
    Otherwise ``skala.toml`` is auto-discovered in ``cwd`` (default: the current
    directory); when absent, defaults are returned.
    """
    source: Path
    if path is not None:
        if not path.is_file():
            raise InputError(f"Config file not found: {path}")
        source = path
    else:
        found = find_config_file(cwd)
        if found is None:
            logger.debug("no %s found; using built-in defaults", CONFIG_FILENAME)
            return SkalaConfig()
        source = found

    try:
        with source.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise InputError(f"Invalid TOML in {source}: {exc}") from exc
    except OSError as exc:
        raise InputError(f"Cannot read config file {source}: {exc}") from exc

    return _from_mapping(data, source=source)


def _from_mapping(data: dict[str, Any], *, source: Path) -> SkalaConfig:
    known = {field.name for field in fields(SkalaConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise InputError(
            f"Unknown key(s) in {source}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(known))}."
        )

    values: dict[str, Any] = {}
    for key, value in data.items():
        _validate(key, value, source=source)
        values[key] = value
    return SkalaConfig(**values)


def _validate(key: str, value: object, *, source: Path) -> None:
    if key in _BOOL_FIELDS:
        if not isinstance(value, bool):
            raise _type_error(source, key, "a boolean", value)
    elif key in _INT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _type_error(source, key, "an integer", value)
        _validate_int_range(source, key, value)
    elif key in _STR_FIELDS:
        if not isinstance(value, str) or not value.strip():
            raise _type_error(source, key, "a non-empty string", value)
        if key == "audio" and value not in AUDIO_CHOICES:
            raise InputError(
                f"{source}: 'audio' must be one of {', '.join(AUDIO_CHOICES)}, got {value!r}."
            )
    else:  # pragma: no cover - unknown keys are rejected before validation
        raise InputError(f"{source}: Unknown key: {key!r}.")


def _validate_int_range(source: Path, key: str, value: int) -> None:
    if key == "crf" and not 0 <= value <= 51:
        raise InputError(f"{source}: 'crf' must be between 0 and 51, got {value}.")
    if key in {"gpu_id", "tile"} and value < 0:
        raise InputError(f"{source}: '{key}' must be 0 or greater, got {value}.")
    if key == "threads" and value < 1:
        raise InputError(f"{source}: 'threads' must be at least 1, got {value}.")


def _type_error(source: Path, key: str, expected: str, value: object) -> InputError:
    return InputError(f"{source}: '{key}' must be {expected}, got {type(value).__name__}.")
