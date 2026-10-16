"""Locate and drive the ``realesrgan-ncnn-vulkan`` engine binary.

The flag surface is verified against the canonical upstream source
(``xinntao/Real-ESRGAN-ncnn-vulkan``, ``src/main.cpp``; getopt string
``"i:o:s:t:m:n:g:j:f:vxh"``) rather than guessed::

    -i input-path        input image path or directory (required)
    -o output-path       output image path or directory (required)
    -s scale             upscale ratio (2, 3 or 4; default 4)
    -t tile-size         tile size (>=32, 0=auto; default 0)
    -m model-path        folder holding the models; must contain "models"
    -n model-name        realesr-animevideov3 | realesrgan-x4plus |
                         realesrgan-x4plus-anime | realesrnet-x4plus
    -g gpu-id            gpu device id (default auto)
    -j load:proc:save    thread counts (default 1:2:2)
    -f format            output image format (png, webp or jpg; default png)
    -x                   enable tta mode (slow; deliberately not exposed)
    -v                   verbose output
    -h                   help

This binary has no fp16 flag and no ``-y`` flag; ``-x`` means TTA, not fp16.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from .utils import DependencyError, EngineError, InputError, get_logger, skala_home

__all__ = [
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_VIDEO_MODEL",
    "ENGINE_BINARY",
    "MIN_TILE",
    "MODEL_SPECS",
    "TILE_LADDER",
    "VALID_FORMATS",
    "ModelSpec",
    "build_command",
    "discover_engine",
    "managed_model_dir",
    "model_spec",
    "run_engine",
    "tile_retry_sequence",
]

_log = get_logger("engine")

ENGINE_BINARY = "realesrgan-ncnn-vulkan"
ENV_ENGINE = "SKALA_ENGINE"

#: Tile sizes tried, largest first, when the engine exhausts GPU memory. The tail
#: mirrors the engine's own heap-budget auto-tile policy (200/100/64/32).
TILE_LADDER = (400, 200, 100, 64, 32)

#: The engine rejects non-zero tiles below this value (verified in main.cpp).
MIN_TILE = 32

#: Output formats accepted by ``-f`` (verified in main.cpp).
VALID_FORMATS = ("png", "jpg", "webp")


@dataclass(frozen=True)
class ModelSpec:
    """A model distributed with the realesrgan-ncnn-vulkan binary."""

    name: str
    native_scales: tuple[int, ...]
    kind: str


#: Model names accepted by ``-n`` and the scales each one serves natively.
#: ``realesr-animevideov3`` loads per-scale weights (``-x2``/``-x3``/``-x4``); every
#: other model ships a single fixed 4x network.
MODEL_SPECS: dict[str, ModelSpec] = {
    "realesrgan-x4plus": ModelSpec("realesrgan-x4plus", (4,), "photo"),
    "realesrgan-x4plus-anime": ModelSpec("realesrgan-x4plus-anime", (4,), "anime"),
    "realesrnet-x4plus": ModelSpec("realesrnet-x4plus", (4,), "photo"),
    "realesr-animevideov3": ModelSpec("realesr-animevideov3", (2, 3, 4), "video"),
}

DEFAULT_IMAGE_MODEL = "realesrgan-x4plus"
DEFAULT_VIDEO_MODEL = "realesr-animevideov3"


def _candidate_names() -> tuple[str, ...]:
    """Binary filenames to look for, in priority order, for this platform."""
    if os.name == "nt":
        return (f"{ENGINE_BINARY}.exe", f"{ENGINE_BINARY}.bat", f"{ENGINE_BINARY}.cmd")
    return (ENGINE_BINARY,)


def _resolve_engine(path: str | os.PathLike[str]) -> Path | None:
    """Resolve ``path`` to an existing engine file, or ``None``.

    Handles an explicit file, a Windows executable name without the ``.exe`` suffix,
    and a bare command name that should be looked up on ``PATH``.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if os.name == "nt" and candidate.suffix == "":
        for suffix in (".exe", ".bat", ".cmd"):
            alternative = candidate.with_name(candidate.name + suffix)
            if alternative.is_file():
                return alternative.resolve()
    if candidate.parent == Path():
        located = shutil.which(candidate.name)
        if located:
            return Path(located).resolve()
    return None


def _first_in_dir(directory: Path) -> Path | None:
    """Return the first engine binary present in ``directory``, or ``None``."""
    for name in _candidate_names():
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def discover_engine(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the engine binary path, or raise if none can be found.

    Searched in order: ``explicit``, ``$SKALA_ENGINE``, the managed install
    created by ``skala setup``, ``./bin``, then ``PATH``.

    Raises:
        DependencyError: when no usable binary can be found.
    """
    if explicit is not None:
        found = _resolve_engine(explicit)
        if found is None:
            raise DependencyError(f"engine binary not found at --engine-path {str(explicit)!r}")
        return found

    from_env = os.environ.get(ENV_ENGINE)
    if from_env:
        found = _resolve_engine(from_env)
        if found is None:
            raise DependencyError(f"{ENV_ENGINE} is set to {from_env!r}, but no such file exists")
        return found

    for directory in (skala_home() / "engine", Path.cwd() / "bin"):
        found = _first_in_dir(directory)
        if found is not None:
            return found

    located = shutil.which(ENGINE_BINARY)
    if located:
        return Path(located).resolve()

    raise DependencyError(
        f"could not find the {ENGINE_BINARY} binary. Run 'skala setup' to install it, "
        f"add it to PATH, place it in ./bin/, or pass --engine-path / set {ENV_ENGINE}."
    )


def managed_model_dir(engine: str | os.PathLike[str]) -> Path | None:
    """Return the ``models`` folder shipped next to ``engine``, or ``None``.

    ``skala setup`` installs the bundled models beside the binary; this detects
    that layout so ``--model-dir`` can stay optional for managed installs.
    """
    models = Path(engine).resolve().parent / "models"
    return models if models.is_dir() else None


def model_spec(model: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``model``.

    Raises:
        InputError: for an unknown model name.
    """
    spec = MODEL_SPECS.get(model)
    if spec is None:
        known = ", ".join(sorted(MODEL_SPECS))
        raise InputError(f"unknown model {model!r}; supported models: {known}")
    return spec


def build_command(
    engine: str | os.PathLike[str],
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    model: str,
    scale: int,
    tile: int,
    model_dir: str | os.PathLike[str] | None = None,
    gpu_id: int | None = None,
    threads: int | None = None,
    fmt: str | None = None,
    verbose: bool = False,
) -> list[str]:
    """Build the engine argv list. Pure function: validates ranges, touches nothing.

    Raises:
        InputError: for an unknown model, an unsupported scale, or out-of-range options.
    """
    spec = model_spec(model)
    if scale not in spec.native_scales:
        supported = ", ".join(str(item) for item in spec.native_scales)
        raise InputError(
            f"model {model!r} does not support scale {scale}x natively "
            f"(supported: {supported}); split the request with plan_scale() first"
        )
    if tile != 0 and tile < MIN_TILE:
        raise InputError(f"tile size must be 0 (auto) or >= {MIN_TILE}, got {tile}")
    if gpu_id is not None and gpu_id < 0:
        raise InputError(f"gpu id must be >= 0, got {gpu_id}")
    if threads is not None and threads < 1:
        raise InputError(f"thread count must be >= 1, got {threads}")
    if fmt is not None and fmt not in VALID_FORMATS:
        valid = ", ".join(VALID_FORMATS)
        raise InputError(f"format must be one of {valid}, got {fmt!r}")
    if model_dir is not None and "models" not in str(model_dir).lower():
        raise InputError(
            f"model directory {str(model_dir)!r} must contain 'models' in its path; "
            "the engine rejects any other layout"
        )

    argv = [
        str(engine),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        model,
        "-s",
        str(scale),
        "-t",
        str(tile),
    ]
    if model_dir is not None:
        argv += ["-m", str(model_dir)]
    if gpu_id is not None:
        argv += ["-g", str(gpu_id)]
    if threads is not None:
        argv += ["-j", f"{threads}:{threads}:{threads}"]
    if fmt is not None:
        argv += ["-f", fmt]
    if verbose:
        argv.append("-v")
    return argv


def tile_retry_sequence(tile: int) -> tuple[int, ...]:
    """Return the tile sizes to try, in order, starting from ``tile``.

    ``0`` means the engine's automatic sizing, which is tried first and then narrowed
    through the fixed ladder because auto always picks the largest tile that fits.
    """
    if tile != 0 and tile < MIN_TILE:
        raise InputError(f"tile size must be 0 (auto) or >= {MIN_TILE}, got {tile}")
    if tile == 0:
        return (0, *TILE_LADDER)
    return (tile, *[step for step in TILE_LADDER if step < tile])


_OOM_PATTERN = re.compile(
    r"(?i)vkallocatememory|out of (?:device |gpu )?memory|failed to allocate|bad_alloc"
)


def _replace_tile(argv: Sequence[str], tile: int) -> list[str]:
    args = list(argv)
    try:
        index = args.index("-t")
    except ValueError:
        return args
    args[index + 1] = str(tile)
    return args


def _tail(text: str, limit: int = 800) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return "..." + stripped[-limit:]


def _failure_message(engine: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = _tail(result.stderr) or _tail(result.stdout) or "(no output)"
    return f"{engine} exited with code {result.returncode}:\n{detail}"


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    _log.debug("running: %s", " ".join(argv))
    return subprocess.run(list(argv), capture_output=True, text=True, check=False)


def run_engine(
    argv: Sequence[str],
    *,
    tile_candidates: Sequence[int] = (0,),
    log: Logger | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the engine, shrinking the tile size whenever it exhausts GPU memory.

    Args:
        argv: Command built by :func:`build_command`; ``-t`` is rewritten per attempt.
        tile_candidates: Tile sizes to try in order.
        log: Logger for retry warnings; defaults to the module logger.

    Returns:
        The successful process result.

    Raises:
        ValueError: if ``tile_candidates`` is empty.
        EngineError: for a non-memory failure, or when every tile size fails.
    """
    logger = log or _log
    candidates = tuple(tile_candidates)
    if not candidates:
        raise ValueError("tile_candidates must not be empty")

    failures: list[subprocess.CompletedProcess[str]] = []
    for tile in candidates:
        attempt = _replace_tile(argv, tile)
        result = _run(attempt)
        if result.returncode == 0:
            return result
        failures.append(result)
        if not _OOM_PATTERN.search(f"{result.stdout}\n{result.stderr}"):
            raise EngineError(_failure_message(attempt[0], result))
        remaining = len(candidates) - len(failures)
        logger.warning("out of GPU memory at tile %s; retries left: %s", tile, remaining)

    tried = ", ".join(str(tile) for tile in candidates)
    raise EngineError(
        f"engine ran out of GPU memory for every tile size tried ({tried}).\n"
        f"Last attempt:\n{_failure_message(str(argv[0]), failures[-1])}"
    )
