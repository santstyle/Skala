"""Single-image upscaling built on top of the engine wrapper.

The pipeline is deliberately boring:

1. Read the source with Pillow and record its mode, size and whether it carries an
   alpha channel.
2. Ask :func:`skala.scaling.plan_scale` how many engine passes the requested scale
   needs (2x/3x/4x are usually one pass; 8x may be two).
3. Feed the engine a PNG in a scratch directory, chaining pass outputs, and finally
   resize with LANCZOS only when the engine passes cannot land on the exact target
   size (e.g. 2x via a model that only ships a 4x network).
4. Re-attach transparency, convert to the requested output format and save.

Alpha is handled with a cheap-first strategy: the engine reads and writes 4-channel
PNG, so the common path is a single pass that keeps alpha intact. If a build ever
returns an opaque image for a transparent source, the alpha channel is upscaled in a
second pass and re-composited; LANCZOS is reserved for the exact-size fixup only.

There is no ``--resume`` for images: they are single, small jobs, so the scratch
directory is created per run and removed afterwards even on failure.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from PIL import Image

from .engine import (
    DEFAULT_IMAGE_MODEL,
    build_command,
    discover_engine,
    managed_model_dir,
    model_spec,
    run_engine,
    tile_retry_sequence,
)
from .scaling import VALID_SCALES, plan_scale
from .utils import (
    EngineError,
    InputError,
    default_output_path,
    default_work_base,
    ensure_output_path,
    format_bytes,
    get_logger,
    require_free_space,
    validate_input_path,
)

__all__ = [
    "ImageInfo",
    "ImageResult",
    "inspect_image",
    "upscale_image",
]

_log = get_logger("image")

#: Pillow modes that carry an alpha channel directly.
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA"})

#: Pillow modes that are single-channel (luminance) rather than colour.
_GRAYSCALE_MODES = frozenset({"1", "L", "I", "F", "I;16", "I;16B", "I;16L", "I;16N"})

#: Pillow modes whose samples are wider than 8 bits; the engine is 8-bit only.
_HIGH_DEPTH_MODES = frozenset({"I", "F", "I;16", "I;16B", "I;16L", "I;16N"})

_EXT_TO_FORMAT = {"png": "png", "jpg": "jpg", "jpeg": "jpg", "webp": "webp"}

_PIL_FORMAT = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP"}


@dataclass(frozen=True)
class ImageInfo:
    """What Skala needs to know about a source image before planning work."""

    path: Path
    size: tuple[int, int]
    mode: str
    has_alpha: bool

    @property
    def pixels(self) -> int:
        """Total pixel count of the source."""
        return self.size[0] * self.size[1]


@dataclass(frozen=True)
class ImageResult:
    """Outcome of a successful :func:`upscale_image` call."""

    source: Path
    output: Path
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    scale: int
    model: str
    passes: tuple[int, ...]
    format: str
    resized: bool
    alpha_handled: str
    warnings: tuple[str, ...]
    elapsed: float

    def describe(self) -> str:
        """Return a one-line summary suitable for logging."""
        chain = " -> ".join(f"{factor}x" for factor in self.passes)
        width, height = self.output_size
        return (
            f"{self.source.name}: {self.input_size[0]}x{self.input_size[1]} -> "
            f"{width}x{height} ({self.scale}x via {chain}, {self.format}"
            f"{', LANCZOS fixup' if self.resized else ''})"
        )


def _has_alpha(image: Image.Image) -> bool:
    """Return True when ``image`` stores transparency that must survive the run."""
    if image.mode in _ALPHA_MODES:
        return True
    return image.mode == "P" and "transparency" in image.info


def _target_mode(original_mode: str, has_alpha: bool) -> str:
    """Return the Pillow mode the output should be saved in."""
    if has_alpha:
        return "RGBA"
    if original_mode in _GRAYSCALE_MODES:
        return "L"
    return "RGB"


def _depth_warning(mode: str) -> str | None:
    """Return a warning when ``mode`` is wider than the engine's 8-bit pipeline."""
    if mode in _HIGH_DEPTH_MODES:
        return (
            f"the source is {mode} (higher than 8 bits per channel); the engine works "
            "in 8-bit, so the result is written at 8 bits per channel"
        )
    return None


def _validate_format(fmt: str) -> str:
    if fmt not in _PIL_FORMAT:
        valid = ", ".join(sorted(_PIL_FORMAT))
        raise InputError(f"format must be one of {valid}, got {fmt!r}")
    return fmt


def _resolve_format(fmt: str | None, output_path: Path | None, source: Path) -> str:
    """Decide the output format from the explicit flag, the output path, or the source.

    An explicit ``fmt`` always wins. Otherwise a known output extension is honoured;
    an unknown one is rejected rather than silently writing a different container.
    With neither signal, the source extension is kept when Skala can write it.
    """
    if fmt is not None:
        return _validate_format(fmt)
    if output_path is not None and output_path.suffix:
        candidate = output_path.suffix.lstrip(".").lower()
        mapped = _EXT_TO_FORMAT.get(candidate)
        if mapped is None:
            valid = ", ".join(sorted(_PIL_FORMAT))
            raise InputError(
                f"unsupported output extension {output_path.suffix!r}; use one of {valid} "
                f"or pass an explicit format"
            )
        return mapped
    return _EXT_TO_FORMAT.get(source.suffix.lstrip(".").lower(), "png")


def _save_options(fmt: str) -> dict[str, int]:
    if fmt in {"jpg", "webp"}:
        return {"quality": 95}
    return {}


def _engine_input(source: Image.Image, has_alpha: bool) -> Image.Image:
    """Convert the source into a 3- or 4-channel image the engine accepts.

    Grayscale and high-bit-depth sources are widened to RGB before the engine runs
    (1-channel PNG is ambiguous for the ncnn loader) and narrowed back afterwards.
    """
    if has_alpha:
        return source.convert("RGBA")
    if source.mode in _HIGH_DEPTH_MODES:
        return source.convert("L").convert("RGB")
    return source.convert("RGB")


def inspect_image(path: str | os.PathLike[str]) -> ImageInfo:
    """Read image metadata, validating that ``path`` is a readable image file.

    Raises:
        InputError: when the path is missing, a directory, unreadable or undecodable.
    """
    source = Path(path)
    validate_input_path(source)
    if source.is_dir():
        raise InputError(f"expected an image file, got a directory: {source}")
    try:
        with Image.open(source) as handle:
            handle.load()
            return ImageInfo(
                path=source,
                size=(handle.width, handle.height),
                mode=handle.mode,
                has_alpha=_has_alpha(handle),
            )
    except InputError:
        raise
    except Exception as exc:  # Pillow raises a wide range of decode/IO errors
        raise InputError(f"cannot read image {source}: {exc}") from exc


def _run_passes(
    engine_bin: Path,
    source_png: Path,
    destination_png: Path,
    passes: tuple[int, ...],
    *,
    model: str,
    tile: int,
    model_dir: str | os.PathLike[str] | None,
    gpu_id: int | None,
    threads: int | None,
    log: Logger,
) -> None:
    """Run one engine invocation per pass, chaining intermediate PNGs."""
    current = source_png
    for index, factor in enumerate(passes):
        is_last = index == len(passes) - 1
        target = (
            destination_png
            if is_last
            else destination_png.with_name(f"{destination_png.stem}_pass{index}.png")
        )
        argv = build_command(
            engine_bin,
            current,
            target,
            model=model,
            scale=factor,
            tile=tile,
            model_dir=model_dir,
            gpu_id=gpu_id,
            threads=threads,
        )
        run_engine(argv, tile_candidates=tile_retry_sequence(tile), log=log)
        if not target.is_file():
            raise EngineError(
                f"the engine reported success but produced no file for pass "
                f"{index + 1} ({factor}x): {target}. This normally means the input "
                f"image could not be decoded."
            )
        current = target


def _estimate_workspace(plan_size: tuple[int, int], *, channels: int) -> int:
    """Estimate peak bytes for the engine input/output plus headroom."""
    pixels = plan_size[0] * plan_size[1]
    return int(pixels * channels * 2 * 1.1) + 8 * 1024 * 1024


def upscale_image(
    input_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    scale: int = 4,
    model: str = DEFAULT_IMAGE_MODEL,
    tile: int = 0,
    model_dir: str | os.PathLike[str] | None = None,
    gpu_id: int | None = None,
    threads: int | None = None,
    fmt: str | None = None,
    overwrite: bool = False,
    engine_path: str | os.PathLike[str] | None = None,
    work_base: str | os.PathLike[str] | None = None,
    log: Logger | None = None,
) -> ImageResult:
    """Upscale a single image by ``scale`` (2, 3, 4 or 8).

    Args:
        input_path: Source image file.
        output_path: Destination file; defaults to ``<name>_skala_<scale>x.<ext>``
            next to the source.
        scale: Requested overall factor, one of :data:`skala.scaling.VALID_SCALES`.
        model: Engine model name (see :data:`skala.engine.MODEL_SPECS`).
        tile: Engine tile size; ``0`` selects the engine's automatic sizing.
        model_dir: Folder holding the engine models (must contain ``models``).
        gpu_id: GPU device index; ``None`` lets the engine choose.
        threads: ``-j`` load/proc/save thread count.
        fmt: Output format (``png``, ``jpg`` or ``webp``).
        overwrite: Allow replacing an existing output file.
        engine_path: Explicit engine binary, overriding discovery.
        work_base: Parent directory for the scratch directory.
        log: Logger override; defaults to ``skala.image``.

    Returns:
        An :class:`ImageResult` describing what was written.

    Raises:
        InputError: bad source, unknown model, invalid scale/format, or output == input.
        OutputExistsError: output exists and ``overwrite`` is False.
        DependencyError: the engine binary could not be found.
        DiskSpaceError: the destination volume lacks room for the workspace.
        EngineError: the engine failed or produced no output.
    """
    logger = log or _log
    source = Path(input_path)
    validate_input_path(source)
    if source.is_dir():
        raise InputError(f"expected an image file, got a directory: {source}")

    spec = model_spec(model)
    if scale not in VALID_SCALES:
        valid = ", ".join(str(item) for item in VALID_SCALES)
        raise InputError(f"scale must be one of {valid}, got {scale}")

    try:
        with Image.open(source) as handle:
            handle.load()
            original_mode = handle.mode
            width, height = handle.width, handle.height
            has_alpha = _has_alpha(handle)
            source_image = handle.copy()
    except InputError:
        raise
    except Exception as exc:  # Pillow raises a wide range of decode/IO errors
        raise InputError(f"cannot read image {source}: {exc}") from exc

    plan = plan_scale(width, height, scale, native_scales=spec.native_scales)

    requested_output = Path(output_path) if output_path is not None else None
    resolved_format = _resolve_format(fmt, requested_output, source)
    if requested_output is not None:
        destination = requested_output
    else:
        extension = ".jpg" if resolved_format == "jpg" else f".{resolved_format}"
        destination = default_output_path(source, scale, extension)
    if destination.resolve() == source.resolve():
        raise InputError(f"output path must differ from the input path: {destination}")
    ensure_output_path(destination, overwrite=overwrite)

    channels = 4 if has_alpha else 3
    peak = max(plan.intermediate_size, plan.final_size, (width, height))
    needed = _estimate_workspace(peak, channels=channels)
    require_free_space(destination, needed, what=f"upscaling {source.name} to {scale}x")
    logger.debug("estimated scratch requirement: %s", format_bytes(needed))

    engine_bin = discover_engine(engine_path)
    if model_dir is None:
        model_dir = managed_model_dir(engine_bin)

    base = Path(work_base) if work_base is not None else default_work_base()
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="skala_img_", dir=base))

    warnings = list(plan.warnings)
    depth_warning = _depth_warning(original_mode)
    if depth_warning is not None:
        warnings.append(depth_warning)

    alpha_handled = "none"
    started = time.monotonic()
    try:
        color_source = work / "source.png"
        _engine_input(source_image, has_alpha).save(color_source, format="PNG")

        color_result = work / "upscaled.png"
        _run_passes(
            engine_bin,
            color_source,
            color_result,
            plan.passes,
            model=model,
            tile=tile,
            model_dir=model_dir,
            gpu_id=gpu_id,
            threads=threads,
            log=logger,
        )

        with Image.open(color_result) as handle:
            handle.load()
            upscaled = handle.copy()

        if has_alpha:
            if _has_alpha(upscaled):
                alpha_handled = "engine"
            else:
                logger.warning(
                    "the engine returned an opaque image for a transparent source; "
                    "upscaling the alpha channel separately"
                )
                alpha_handled = "fallback"
                alpha_source = work / "alpha.png"
                source_image.convert("RGBA").getchannel("A").convert("RGB").save(
                    alpha_source, format="PNG"
                )
                alpha_result = work / "alpha_upscaled.png"
                _run_passes(
                    engine_bin,
                    alpha_source,
                    alpha_result,
                    plan.passes,
                    model=model,
                    tile=tile,
                    model_dir=model_dir,
                    gpu_id=gpu_id,
                    threads=threads,
                    log=logger,
                )
                with Image.open(alpha_result) as handle:
                    handle.load()
                    alpha_channel = handle.convert("L")
                if plan.needs_resize:
                    alpha_channel = alpha_channel.resize(plan.final_size, Image.Resampling.LANCZOS)
                upscaled = upscaled.convert("RGB")
                upscaled.putalpha(alpha_channel)

        if plan.needs_resize:
            upscaled = upscaled.resize(plan.final_size, Image.Resampling.LANCZOS)

        target_mode = _target_mode(original_mode, has_alpha)
        result_image = upscaled.convert(target_mode)
        if resolved_format == "jpg" and "A" in result_image.getbands():
            background = Image.new("RGB", result_image.size, (255, 255, 255))
            background.paste(result_image, mask=result_image.getchannel("A"))
            result_image = background
            warnings.append(
                "JPEG cannot store transparency; the alpha channel was flattened onto white"
            )

        result_image.save(
            destination,
            format=_PIL_FORMAT[resolved_format],
            **_save_options(resolved_format),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    elapsed = time.monotonic() - started
    logger.info("%s -> %s in %.1fs", source.name, destination.name, elapsed)
    return ImageResult(
        source=source,
        output=destination,
        input_size=(width, height),
        output_size=result_image.size,
        scale=scale,
        model=model,
        passes=plan.passes,
        format=resolved_format,
        resized=plan.needs_resize,
        alpha_handled=alpha_handled,
        warnings=tuple(warnings),
        elapsed=elapsed,
    )
