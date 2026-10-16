"""Command-line interface for Skala.

``skala <input>`` upscales a single image, a single video, or every image/video
directly inside a folder. Options not given on the command line fall back to
``skala.toml`` and then to built-in defaults (see :mod:`skala.config`).

``skala <input>`` is shorthand for the ``run`` command, which is the default:
any first argument that is not a known subcommand (``run``, ``doctor``) is
treated as the path to process. ``skala doctor`` reports whether the external
tools are available. A bare ``skala`` in a terminal starts a short guided
prompt (pick a file, then a scale) and runs the job; without a terminal it
prints the help text.

Exit codes: ``0`` success, ``1`` a :class:`~skala.utils.SkalaError` (bad input,
missing tool, engine failure), ``2`` usage error (raised by Typer), ``130``
interrupted (Ctrl+C).
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Annotated, NoReturn, TypeVar

import click
import typer
from typer.core import TyperGroup

from . import __version__
from .config import AUDIO_CHOICES, load_config
from .engine import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_VIDEO_MODEL,
    MODEL_SPECS,
    DependencyError,
    discover_engine,
    model_spec,
)
from .image import ImageResult, inspect_image, upscale_image
from .scaling import VALID_SCALES, plan_scale
from .utils import (
    InputError,
    SkalaError,
    default_output_path,
    detect_hardware,
    format_duration,
    get_logger,
    missing_ffmpeg,
    setup_logging,
    validate_input_path,
)
from .video import (
    DEFAULT_CRF,
    DEFAULT_VIDEO_CODEC,
    VideoResult,
    probe_video,
    upscale_video,
)

__all__ = ["app", "main"]

logger = get_logger("cli")

#: Extensions the image pipeline can read (and, for png/jpg/webp, write).
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})

#: Image extensions Skala can write back out (see ``skala.image``).
_IMAGE_OUTPUT_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

#: Extensions the video pipeline treats as video containers.
_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts"}
)

#: Name of the implicit default command (see :class:`_DefaultGroup`).
_DEFAULT_COMMAND = "run"

_T = TypeVar("_T")


class _DefaultGroup(TyperGroup):
    """Typer group that treats an unrecognised first argument as ``run``.

    Typer/Click parse a group callback's positional arguments *before*
    resolving subcommands, so a ``source`` argument on the callback would shadow
    ``doctor``. Keeping the callback positional-free and rerouting unknown
    first tokens to the default command lets ``skala <path>`` and
    ``skala doctor`` coexist.
    """

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        name = args[0] if args else None
        if name is not None and not name.startswith("-") and name not in self.commands:
            args = [_DEFAULT_COMMAND, *args]
        return super().resolve_command(ctx, args)


app = typer.Typer(
    cls=_DefaultGroup,
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    help=(
        "Upscale images and videos 2x-8x with Real-ESRGAN (ncnn/Vulkan). "
        "Run a job with 'skala PATH [OPTIONS]'; the 'run' command is the default."
    ),
)


def _pick(cli_value: _T | None, config_value: _T | None, default: _T) -> _T:
    """Merge a CLI value over a config value over a built-in default."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _to_str(value: Path | None) -> str | None:
    """Return ``str(value)`` for a path, or ``None`` when unset."""
    return str(value) if value is not None else None


def _fail(message: str, code: int = 1) -> NoReturn:
    """Print an error and exit with ``code``."""
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _kind_for_suffix(path: Path) -> str | None:
    """Classify a file as ``"image"``/``"video"`` by extension, or ``None``."""
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    return None


def _resolve_targets(source: Path, kind: str) -> list[tuple[str, Path]]:
    """Return the ``(kind, path)`` pairs to process for ``source``."""
    if source.is_dir():
        wanted = None if kind == "auto" else kind
        found: list[tuple[str, Path]] = []
        for child in sorted(source.iterdir()):
            if not child.is_file():
                continue
            detected = _kind_for_suffix(child)
            if detected is None or (wanted is not None and detected != wanted):
                continue
            found.append((detected, child))
        if not found:
            raise InputError(f"no image or video files found in {source}")
        return found

    if kind == "auto":
        detected = _kind_for_suffix(source)
        if detected is None:
            known = ", ".join(sorted(_IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS))
            raise InputError(
                f"cannot tell whether {source} is an image or a video from its "
                f"extension; pass --type (known extensions: {known})"
            )
        return [(detected, source)]
    return [(kind, source)]


def _destination_for(
    output_dir: Path | None,
    output_file: Path | None,
    source: Path,
    scale: int,
    kind: str,
) -> Path | None:
    """Resolve the output path for one target, or ``None`` for the default."""
    if output_dir is None:
        return output_file
    if kind == "video":
        ext = source.suffix or ".mp4"
    else:
        suffix = source.suffix.lower()
        ext = suffix if suffix in _IMAGE_OUTPUT_EXTENSIONS else ".png"
    return output_dir / default_output_path(source, scale, ext).name


def _report(result: ImageResult | VideoResult) -> None:
    """Print a one-line summary plus any warnings for a finished job."""
    typer.echo(result.describe())
    for warning in result.warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)


def _dry_run(kind: str, source: Path, scale: int, model: str, destination: Path | None) -> None:
    """Print what a real run would do, without touching the engine or ffmpeg."""
    spec = model_spec(model)
    size: tuple[int, int] | None = None
    try:
        if kind == "image":
            size = inspect_image(source).size
        else:
            info = probe_video(source)
            size = (info.width, info.height)
    except SkalaError:
        size = None

    if size is not None:
        detail = plan_scale(size[0], size[1], scale, native_scales=spec.native_scales).describe()
    else:
        preview = plan_scale(1, 1, scale, native_scales=spec.native_scales)
        detail = " -> ".join(f"engine {factor}x" for factor in preview.passes)

    target = str(destination) if destination is not None else "(default beside source)"
    typer.echo(f"[dry-run] {kind} {source.name} -> {target} ({scale}x via {detail}, model={model})")


#: Prompt shown while asking for the upscale factor in interactive mode.
_INTERACTIVE_SCALE_PROMPT = "Upscale factor (2, 3, 4 or 8)"


def _wants_interactive() -> bool:
    """Return True when a bare ``skala`` should start the guided prompt flow.

    The prompts need a real terminal. When stdin is a pipe or a file (scripts,
    CI, :class:`~typer.testing.CliRunner`) Skala prints help instead of waiting
    on an answer that will never arrive.
    """
    return sys.stdin.isatty()


def _prompt_source() -> Path:
    """Prompt until the user names an existing file or folder."""
    while True:
        raw = typer.prompt("File or folder to upscale")
        text = raw.strip().strip("\"'")
        if not text:
            typer.secho("  please type a path", fg=typer.colors.RED, err=True)
            continue
        candidate = Path(text).expanduser()
        try:
            validate_input_path(candidate)
        except SkalaError as exc:
            typer.secho(f"  {exc}", fg=typer.colors.RED, err=True)
            continue
        return candidate


def _prompt_scale() -> int:
    """Prompt until the user picks one of :data:`~skala.scaling.VALID_SCALES`."""
    choices = ", ".join(str(item) for item in VALID_SCALES)
    while True:
        raw = typer.prompt(_INTERACTIVE_SCALE_PROMPT, default="4")
        text = raw.strip()
        if text.isdigit() and int(text) in VALID_SCALES:
            return int(text)
        typer.secho(f"  choose one of: {choices}", fg=typer.colors.RED, err=True)


def _interactive(ctx: typer.Context) -> None:
    """Run the guided prompts, then dispatch to the default ``run`` command."""
    typer.secho("Skala interactive mode", fg=typer.colors.CYAN, bold=True)
    typer.echo("Answer a few prompts; press Ctrl+C to cancel.\n")

    try:
        source = _prompt_source()
        scale = _prompt_scale()
    except click.Abort:
        typer.secho("\naborted", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(130) from None

    typer.echo()
    group = ctx.command
    if not isinstance(group, click.Group):  # pragma: no cover - the root is always a group
        _fail("internal error: the 'run' command is unavailable")
    command = group.get_command(ctx, _DEFAULT_COMMAND)
    if command is None:  # pragma: no cover - the run command is always registered
        _fail("internal error: the 'run' command is unavailable")

    with command.make_context(
        _DEFAULT_COMMAND, [str(source), "--scale", str(scale)], parent=ctx
    ) as sub:
        command.invoke(sub)


@app.callback()
def root(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show the version and exit.")] = False,
) -> None:
    """Upscale an image, a video, or every image/video in a folder."""
    if version:
        typer.echo(f"skala {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    if _wants_interactive():
        _interactive(ctx)
        raise typer.Exit()

    typer.echo(ctx.get_help())
    raise typer.Exit()


@app.command()
def run(
    ctx: typer.Context,
    source: Annotated[
        Path | None, typer.Argument(help="Image, video, or folder to upscale.")
    ] = None,
    scale: Annotated[
        int | None, typer.Option("--scale", "-s", help="Upscale factor: 2, 3, 4 or 8.")
    ] = None,
    kind: Annotated[
        str, typer.Option("--type", help="Input type: auto (default), image, or video.")
    ] = "auto",
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Engine model name (see --help/doctor).")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file, or directory for a folder.")
    ] = None,
    fmt: Annotated[
        str | None, typer.Option("--format", "-f", help="Image output format: png, jpg or webp.")
    ] = None,
    tile: Annotated[int | None, typer.Option("--tile", help="Engine tile size (0 = auto).")] = None,
    model_dir: Annotated[
        Path | None, typer.Option("--model-dir", help="Folder holding the engine models.")
    ] = None,
    gpu_id: Annotated[int | None, typer.Option("--gpu-id", "-g", help="GPU device index.")] = None,
    threads: Annotated[
        int | None, typer.Option("--threads", "-j", help="Engine thread count (>= 1).")
    ] = None,
    crf: Annotated[
        int | None, typer.Option("--crf", help="Video quality factor 0-51 (lower is better).")
    ] = None,
    video_codec: Annotated[
        str | None, typer.Option("--video-codec", help="ffmpeg video encoder (default libx264).")
    ] = None,
    audio: Annotated[
        str | None, typer.Option("--audio", help="Audio handling: copy, drop or aac.")
    ] = None,
    engine_path: Annotated[
        Path | None, typer.Option("--engine-path", help="Explicit engine binary.")
    ] = None,
    ffmpeg_path: Annotated[
        Path | None, typer.Option("--ffmpeg-path", help="Explicit ffmpeg executable.")
    ] = None,
    ffprobe_path: Annotated[
        Path | None, typer.Option("--ffprobe-path", help="Explicit ffprobe executable.")
    ] = None,
    workdir: Annotated[
        Path | None, typer.Option("--workdir", help="Parent directory for scratch files.")
    ] = None,
    overwrite: Annotated[
        bool | None, typer.Option("--overwrite/--no-overwrite", help="Replace existing output.")
    ] = None,
    keep_temp: Annotated[
        bool | None, typer.Option("--keep-temp/--no-keep-temp", help="Keep scratch files.")
    ] = None,
    verbose: Annotated[
        bool | None, typer.Option("--verbose/--no-verbose", "-v", help="Debug logging.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would happen, then exit.")
    ] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Explicit skala.toml path.")
    ] = None,
) -> None:
    """Upscale an image, a video, or every image/video in a folder."""
    if source is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    if kind not in {"auto", "image", "video"}:
        raise typer.BadParameter("must be one of auto, image, video", param_hint="--type")

    try:
        config = load_config(Path(config_path)) if config_path is not None else load_config()
    except SkalaError as exc:
        _fail(str(exc))

    resolved_verbose = _pick(verbose, config.verbose, False)
    setup_logging(resolved_verbose)

    if scale is not None and scale not in VALID_SCALES:
        valid = ", ".join(str(item) for item in VALID_SCALES)
        _fail(f"scale must be one of {valid}, got {scale}")
    resolved_scale = 4 if scale is None else scale

    resolved_audio = _pick(audio, config.audio, "copy")
    if resolved_audio not in AUDIO_CHOICES:
        _fail(f"audio must be one of {', '.join(AUDIO_CHOICES)}, got {resolved_audio!r}")

    resolved_tile = _pick(tile, config.tile, 0)
    resolved_gpu = _pick(gpu_id, config.gpu_id, None)
    resolved_threads = _pick(threads, config.threads, None)
    resolved_overwrite = _pick(overwrite, config.overwrite, False)
    resolved_keep = _pick(keep_temp, config.keep_temp, False)
    resolved_crf = _pick(crf, config.crf, DEFAULT_CRF)
    resolved_codec = _pick(video_codec, config.video_codec, DEFAULT_VIDEO_CODEC)
    resolved_engine = _pick(_to_str(engine_path), config.engine_path, None)
    resolved_ffmpeg = _pick(_to_str(ffmpeg_path), config.ffmpeg_path, None)
    resolved_ffprobe = _pick(_to_str(ffprobe_path), config.ffprobe_path, None)
    resolved_workdir = _pick(_to_str(workdir), config.workdir, None)
    resolved_model_dir = _to_str(model_dir)
    model_image = _pick(model, config.model, DEFAULT_IMAGE_MODEL)
    model_video = _pick(model, config.model, DEFAULT_VIDEO_MODEL)

    src = Path(source)
    try:
        validate_input_path(src)
        targets = _resolve_targets(src, kind)
    except SkalaError as exc:
        _fail(str(exc))

    is_multi = src.is_dir() or len(targets) > 1
    output_dir: Path | None = None
    output_file: Path | None = None
    if output is not None:
        if is_multi:
            if output.suffix:
                _fail("--output must be a directory when processing multiple files")
            output_dir = output
        elif output.suffix:
            output_file = output
        else:
            output_dir = output
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("processing %s target(s) at scale %sx", len(targets), resolved_scale)
    failures = 0
    started = time.monotonic()
    for target_kind, path in targets:
        destination = _destination_for(output_dir, output_file, path, resolved_scale, target_kind)
        if dry_run:
            _dry_run(
                target_kind,
                path,
                resolved_scale,
                model_image if target_kind == "image" else model_video,
                destination,
            )
            continue
        try:
            if target_kind == "image":
                result: ImageResult | VideoResult = upscale_image(
                    path,
                    output_path=destination,
                    scale=resolved_scale,
                    model=model_image,
                    tile=resolved_tile,
                    model_dir=resolved_model_dir,
                    gpu_id=resolved_gpu,
                    threads=resolved_threads,
                    fmt=fmt,
                    overwrite=resolved_overwrite,
                    engine_path=resolved_engine,
                    work_base=resolved_workdir,
                )
            else:
                result = upscale_video(
                    path,
                    output_path=destination,
                    scale=resolved_scale,
                    model=model_video,
                    tile=resolved_tile,
                    model_dir=resolved_model_dir,
                    gpu_id=resolved_gpu,
                    threads=resolved_threads,
                    overwrite=resolved_overwrite,
                    engine_path=resolved_engine,
                    ffmpeg_path=resolved_ffmpeg,
                    ffprobe_path=resolved_ffprobe,
                    keep_temp=resolved_keep,
                    crf=resolved_crf,
                    video_codec=resolved_codec,
                    audio=resolved_audio,
                    work_base=resolved_workdir,
                )
        except KeyboardInterrupt:
            typer.secho("interrupted", fg=typer.colors.RED, err=True)
            raise typer.Exit(130) from None
        except SkalaError as exc:
            failures += 1
            if is_multi:
                typer.secho(f"error: {path}: {exc}", fg=typer.colors.RED, err=True)
            else:
                _fail(str(exc))
        else:
            _report(result)

    if is_multi:
        typer.echo(
            f"upscaled {len(targets) - failures} of {len(targets)} file(s) "
            f"in {format_duration(time.monotonic() - started)}"
        )
    if failures:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Report whether the external tools Skala needs are available."""
    typer.echo(f"skala {__version__}")
    typer.echo(f"python {platform.python_version()} on {platform.system()} {platform.release()}")

    hardware = detect_hardware()
    state = {True: "yes", False: "no", None: "unknown"}[hardware.vulkan_available]
    typer.echo(f"vulkan: {state} ({hardware.vulkan_detail})")
    typer.echo(f"cpu cores: {hardware.cpu_count}")

    missing = missing_ffmpeg()
    if missing:
        for dependency in missing:
            typer.secho(f"ffmpeg tool missing: {dependency.name}", fg=typer.colors.YELLOW, err=True)
            typer.echo(f"  {dependency.hint()}")
    else:
        typer.echo("ffmpeg/ffprobe: found")

    try:
        engine = discover_engine()
    except DependencyError as exc:
        typer.secho(f"engine: not found ({exc})", fg=typer.colors.YELLOW, err=True)
    else:
        typer.echo(f"engine: {engine}")

    typer.echo(f"models: {', '.join(sorted(MODEL_SPECS))}")


def main() -> None:
    """Console-script entry point (``skala = skala.cli:main``)."""
    app()
