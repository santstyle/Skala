"""Video upscaling: probe, extract frames, upscale them, re-encode.

The engine only understands still images, so a video is decomposed into its frames,
each frame is pushed through the same engine wrapper :mod:`skala.image` uses, and the
upscaled frames are re-encoded with the original audio track.

The pipeline has four stages, all driven by ffmpeg/ffprobe plus the engine:

1. ``ffprobe`` (JSON) reports the frame size, frame rate, duration, codec and whether
   an audio stream exists.
2. ``ffmpeg`` extracts the frames as a numbered PNG sequence.
3. The engine upscales the sequence, once per :func:`skala.scaling.plan_scale` pass.
4. ``ffmpeg`` re-encodes the upscaled sequence, copying the source audio across (or
   dropping it, or re-encoding to AAC) and resizing only when the engine passes
   cannot land on the exact target size.

Unlike images, videos run long enough that a failure mid-way is worth resuming, so
the working directory is deterministic for a given (input, scale, model, tile) and is
kept on failure. A rerun reuses completed frames and only redoes the stages that are
incomplete. On success the directory is removed unless ``keep_temp`` is set.

The two ffmpeg stages report their frame counts through ``-progress pipe:1``, so each
drives a progress bar. The engine passes are logged rather than counted: a single
invocation owns the whole sequence, so per-frame progress is not available from it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from logging import Logger
from pathlib import Path

from tqdm import tqdm

from .config import AUDIO_CHOICES
from .engine import (
    DEFAULT_VIDEO_MODEL,
    build_command,
    discover_engine,
    managed_model_dir,
    model_spec,
    run_engine,
    tile_retry_sequence,
)
from .scaling import VALID_SCALES, ScalePlan, plan_scale
from .utils import (
    DependencyError,
    EngineError,
    InputError,
    SkalaError,
    default_output_path,
    ensure_output_path,
    format_bytes,
    format_duration,
    get_logger,
    missing_ffmpeg,
    require_free_space,
    validate_input_path,
    work_dir,
)

__all__ = [
    "DEFAULT_CRF",
    "DEFAULT_VIDEO_CODEC",
    "VideoInfo",
    "VideoResult",
    "probe_video",
    "upscale_video",
]

_log = get_logger("video")

#: Default H.264 quality factor; lower is better and larger (0-51).
DEFAULT_CRF = 18

#: Default video encoder; widely available in ffmpeg builds.
DEFAULT_VIDEO_CODEC = "libx264"

#: ffmpeg's printf-style pattern for a numbered PNG sequence.
_FRAME_GLOB = "%08d.png"

#: Frame rate assumed when a container does not advertise a usable one.
_FALLBACK_FPS = Fraction(25, 1)


@dataclass(frozen=True)
class VideoInfo:
    """What Skala needs to know about a source video before planning work."""

    path: Path
    width: int
    height: int
    fps: Fraction
    duration: float
    codec: str
    pix_fmt: str
    has_audio: bool
    nb_frames: int | None

    @property
    def fps_text(self) -> str:
        """Frame rate as an ffmpeg ``num/den`` string."""
        return f"{self.fps.numerator}/{self.fps.denominator}"

    def expected_frames(self) -> int:
        """Best-effort frame count, used to size the workspace before extraction."""
        if self.nb_frames:
            return self.nb_frames
        if self.duration > 0:
            return max(1, round(self.duration * float(self.fps)))
        return 1


@dataclass(frozen=True)
class VideoResult:
    """Outcome of a successful :func:`upscale_video` call."""

    source: Path
    output: Path
    info: VideoInfo
    scale: int
    model: str
    passes: tuple[int, ...]
    output_size: tuple[int, int]
    frames: int
    audio: str
    resized: bool
    warnings: tuple[str, ...]
    elapsed: float

    def describe(self) -> str:
        """Return a one-line summary suitable for logging."""
        chain = " -> ".join(f"{factor}x" for factor in self.passes)
        width, height = self.output_size
        return (
            f"{self.source.name}: {self.info.width}x{self.info.height} -> "
            f"{width}x{height} ({self.scale}x via {chain}, {self.frames} frames, "
            f"audio: {self.audio}{', LANCZOS fixup' if self.resized else ''})"
        )


def _tail(text: str, limit: int = 800) -> str:
    stripped = (text or "").strip()
    if len(stripped) <= limit:
        return stripped
    return "..." + stripped[-limit:]


def _parse_progress_line(line: str) -> tuple[str, str] | None:
    """Split an ffmpeg ``-progress`` line into ``(key, value)``.

    Returns ``None`` for anything that is not a ``key=value`` line.
    """
    key, separator, value = line.strip().partition("=")
    if not separator or not key:
        return None
    return key, value


def _run_streaming(
    argv: Sequence[str],
    *,
    total: int,
    desc: str,
    log: Logger,
) -> subprocess.CompletedProcess[str]:
    """Run a command while reading ffmpeg ``-progress`` output into a tqdm bar.

    stdout carries the ``-progress`` key/value lines and is consumed line by line;
    stderr is diverted to a temporary file so a verbose command cannot fill a pipe and
    deadlock. The captured stderr is replayed in the returned result, and the tqdm bar
    disables itself when stderr is not a TTY, keeping captured logs clean.
    """
    command = list(argv)
    log.debug("running: %s", " ".join(command))
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errors:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise DependencyError(f"could not execute {argv[0]!r}: {exc}") from exc
        seen = 0
        stdout = process.stdout
        try:
            if stdout is not None:
                with tqdm(total=total, desc=desc, unit="frame", leave=False) as bar:
                    for line in stdout:
                        parsed = _parse_progress_line(line)
                        if parsed is None or parsed[0] != "frame":
                            continue
                        try:
                            count = int(parsed[1])
                        except ValueError:
                            continue
                        bar.update(max(0, count - seen))
                        seen = count
        except BaseException:
            process.kill()
            process.wait()
            raise
        code = process.wait()
        errors.seek(0)
        stderr_text = errors.read()
    return subprocess.CompletedProcess(command, code, "", stderr_text)


def _run(
    argv: Sequence[str],
    log: Logger,
    *,
    progress_total: int | None = None,
    progress_desc: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run ``argv``, optionally while streaming an ffmpeg progress bar.

    With ``progress_total`` set the command is expected to emit ffmpeg ``-progress``
    output on stdout and :func:`_run_streaming` drives a bar; otherwise the output is
    captured wholesale.
    """
    if progress_total is not None:
        return _run_streaming(argv, total=progress_total, desc=progress_desc, log=log)
    log.debug("running: %s", " ".join(argv))
    try:
        return subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"could not execute {argv[0]!r}: {exc}") from exc


def _resolve_tool(
    name: str,
    explicit: str | os.PathLike[str] | None,
    log: Logger,
) -> str:
    """Resolve an external tool to an executable path.

    An explicit path (``--ffmpeg-path``/``--ffprobe-path``) is honoured when it points
    at a real file or a name on ``PATH``; otherwise the tool is looked up normally.

    Raises:
        DependencyError: when the tool cannot be found, with an install hint.
    """
    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        located = shutil.which(str(explicit))
        if located is not None:
            return located
        raise DependencyError(f"{name} was set to {str(explicit)!r}, but no such executable exists")
    located = shutil.which(name)
    if located is not None:
        return located
    hint = next((item.hint() for item in missing_ffmpeg() if item.name == name), "")
    message = f"could not find {name} on PATH"
    if hint:
        message = f"{message}. {hint}"
    raise DependencyError(message)


def _parse_rate(value: object) -> Fraction:
    """Parse an ffprobe frame rate (``"30000/1001"`` or ``"25"``) into a Fraction."""
    if not value or value == "N/A":
        return _FALLBACK_FPS
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return _FALLBACK_FPS
    return rate if rate > 0 else _FALLBACK_FPS


def _parse_float(value: object) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _probe(tool: str, source: Path, log: Logger) -> VideoInfo:
    """Run ffprobe against ``source`` and translate the JSON into a :class:`VideoInfo`."""
    argv = [
        tool,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(source),
    ]
    result = _run(argv, log)
    if result.returncode != 0:
        raise SkalaError(
            f"ffprobe could not read {source} (exit {result.returncode}):\n{_tail(result.stderr)}"
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise SkalaError(f"ffprobe returned malformed JSON for {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SkalaError(f"ffprobe returned an unexpected payload for {source}")

    streams = payload.get("streams") or []
    video = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise InputError(f"no video stream found in {source}")
    has_audio = any(
        isinstance(item, dict) and item.get("codec_type") == "audio" for item in streams
    )

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise InputError(f"could not determine the frame size of {source}")

    container = payload.get("format")
    container = container if isinstance(container, dict) else {}
    duration = _parse_float(container.get("duration")) or _parse_float(video.get("duration")) or 0.0

    raw_frames = video.get("nb_frames")
    nb_frames: int | None = None
    if raw_frames not in (None, "N/A"):
        try:
            parsed = int(raw_frames)
        except (TypeError, ValueError):
            parsed = 0
        nb_frames = parsed if parsed > 0 else None

    return VideoInfo(
        path=source,
        width=width,
        height=height,
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        duration=duration,
        codec=str(video.get("codec_name") or "unknown"),
        pix_fmt=str(video.get("pix_fmt") or "unknown"),
        has_audio=has_audio,
        nb_frames=nb_frames,
    )


def probe_video(
    path: str | os.PathLike[str],
    *,
    ffprobe: str | os.PathLike[str] | None = None,
    log: Logger | None = None,
) -> VideoInfo:
    """Read video metadata, validating that ``path`` is a readable video file.

    Args:
        path: Source video file.
        ffprobe: Explicit ffprobe executable; ``None`` searches ``PATH``.
        log: Logger override; defaults to ``skala.video``.

    Raises:
        InputError: when the path is missing, a directory, or has no video stream.
        DependencyError: when ffprobe cannot be found.
        SkalaError: when ffprobe fails or returns malformed output.
    """
    logger = log or _log
    source = Path(path)
    validate_input_path(source)
    if source.is_dir():
        raise InputError(f"expected a video file, got a directory: {source}")
    tool = _resolve_tool("ffprobe", ffprobe, logger)
    return _probe(tool, source, logger)


def _frame_pattern(directory: Path) -> str:
    return str(directory / _FRAME_GLOB)


def _count_frames(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob("*.png"))


def _extract(ffmpeg: str, source: Path, frames_dir: Path, log: Logger, *, expected: int) -> int:
    """Extract every frame as a numbered PNG and return the count."""
    argv = [
        ffmpeg,
        "-y",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        str(source),
        "-vsync",
        "0",
        _frame_pattern(frames_dir),
    ]
    result = _run(argv, log, progress_total=expected, progress_desc="extracting frames")
    if result.returncode != 0:
        raise SkalaError(
            f"ffmpeg failed to extract frames from {source} (exit {result.returncode}):\n"
            f"{_tail(result.stderr)}"
        )
    count = _count_frames(frames_dir)
    if count == 0:
        raise SkalaError(
            f"ffmpeg reported success but wrote no frames from {source}; the source may "
            f"be corrupt or carry no decodable video stream"
        )
    log.info("extracted %s frames", count)
    return count


def _upscale_frames(
    engine_bin: Path,
    frames_dir: Path,
    work: Path,
    passes: tuple[int, ...],
    *,
    expected: int,
    model: str,
    tile: int,
    model_dir: str | os.PathLike[str] | None,
    gpu_id: int | None,
    threads: int | None,
    log: Logger,
) -> Path:
    """Run one engine invocation per pass, chaining frame directories.

    A pass whose output directory already holds the expected frame count is reused;
    restarting a pass invalidates every later pass derived from it, so a resume never
    encodes frames produced from a stale intermediate.
    """
    current = frames_dir
    for index, factor in enumerate(passes):
        out_dir = work / f"upscaled_{index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        produced = _count_frames(out_dir)
        if produced == expected:
            log.info("pass %s/%s (%sx) already complete; resuming", index + 1, len(passes), factor)
            current = out_dir
            continue
        if produced:
            log.warning("pass %s had %s of %s frames; restarting it", index + 1, produced, expected)
            for stale in out_dir.glob("*.png"):
                stale.unlink()
        for later in range(index + 1, len(passes)):
            shutil.rmtree(work / f"upscaled_{later}", ignore_errors=True)
        argv = build_command(
            engine_bin,
            _frame_pattern(current),
            _frame_pattern(out_dir),
            model=model,
            scale=factor,
            tile=tile,
            model_dir=model_dir,
            gpu_id=gpu_id,
            threads=threads,
        )
        run_engine(argv, tile_candidates=tile_retry_sequence(tile), log=log)
        written = _count_frames(out_dir)
        if written != expected:
            raise EngineError(
                f"the engine produced {written} of {expected} frames for pass {index + 1} "
                f"({factor}x). The working directory was kept at {work} so the run can be "
                f"resumed."
            )
        current = out_dir
    return current


def _encode(
    ffmpeg: str,
    frames_dir: Path,
    destination: Path,
    info: VideoInfo,
    *,
    frames: int,
    crf: int,
    video_codec: str,
    audio: str,
    needs_resize: bool,
    final_size: tuple[int, int],
    log: Logger,
) -> None:
    """Re-encode the upscaled PNG sequence, carrying the source audio across."""
    argv = [
        ffmpeg,
        "-y",
        "-nostats",
        "-progress",
        "pipe:1",
        "-framerate",
        info.fps_text,
        "-i",
        _frame_pattern(frames_dir),
    ]
    if info.has_audio and audio != "drop":
        argv += ["-i", str(info.path), "-map", "0:v:0", "-map", "1:a:0?"]
        argv += ["-c:a", "copy"] if audio == "copy" else ["-c:a", "aac"]
    else:
        argv += ["-map", "0:v:0"]
    argv += ["-c:v", video_codec, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if needs_resize:
        argv += ["-vf", f"scale={final_size[0]}:{final_size[1]}"]
    argv.append(str(destination))

    result = _run(argv, log, progress_total=frames, progress_desc="encoding video")
    if result.returncode != 0:
        raise SkalaError(
            f"ffmpeg failed to encode {destination} (exit {result.returncode}):\n"
            f"{_tail(result.stderr)}"
        )
    if not destination.is_file():
        raise SkalaError(f"ffmpeg reported success but produced no output file: {destination}")


def _estimate_workspace(info: VideoInfo, plan: ScalePlan, frames: int) -> int:
    """Estimate peak scratch bytes: extracted, intermediate and final frames together."""
    source_bytes = info.width * info.height * 3
    intermediate_bytes = plan.intermediate_size[0] * plan.intermediate_size[1] * 2
    final_bytes = plan.final_size[0] * plan.final_size[1] * 2
    per_frame = source_bytes + intermediate_bytes + final_bytes
    return int(frames * per_frame * 1.1) + 16 * 1024 * 1024


def upscale_video(
    input_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    scale: int = 4,
    model: str = DEFAULT_VIDEO_MODEL,
    tile: int = 0,
    model_dir: str | os.PathLike[str] | None = None,
    gpu_id: int | None = None,
    threads: int | None = None,
    overwrite: bool = False,
    engine_path: str | os.PathLike[str] | None = None,
    ffmpeg_path: str | os.PathLike[str] | None = None,
    ffprobe_path: str | os.PathLike[str] | None = None,
    keep_temp: bool = False,
    crf: int = DEFAULT_CRF,
    video_codec: str = DEFAULT_VIDEO_CODEC,
    audio: str = "copy",
    work_base: str | os.PathLike[str] | None = None,
    log: Logger | None = None,
) -> VideoResult:
    """Upscale a video by ``scale`` (2, 3, 4 or 8), keeping or dropping the audio.

    Args:
        input_path: Source video file.
        output_path: Destination file; defaults to ``<name>_skala_<scale>x.<ext>``
            next to the source.
        scale: Requested overall factor, one of :data:`skala.scaling.VALID_SCALES`.
        model: Engine model name (see :data:`skala.engine.MODEL_SPECS`).
        tile: Engine tile size; ``0`` selects the engine's automatic sizing.
        model_dir: Folder holding the engine models (must contain ``models``).
        gpu_id: GPU device index; ``None`` lets the engine choose.
        threads: ``-j`` load/proc/save thread count.
        overwrite: Allow replacing an existing output file.
        engine_path: Explicit engine binary, overriding discovery.
        ffmpeg_path: Explicit ffmpeg executable.
        ffprobe_path: Explicit ffprobe executable.
        keep_temp: Keep the working directory after a successful run.
        crf: H.264 quality factor (0-51, lower is better).
        video_codec: ffmpeg video encoder (default ``libx264``).
        audio: One of :data:`skala.config.AUDIO_CHOICES` (``copy``, ``drop``, ``aac``).
        work_base: Parent directory for the working directory.
        log: Logger override; defaults to ``skala.video``.

    Returns:
        A :class:`VideoResult` describing what was written.

    Raises:
        InputError: bad source, unknown model, invalid scale/audio/crf, or output == input.
        OutputExistsError: output exists and ``overwrite`` is False.
        DependencyError: the engine, ffmpeg or ffprobe could not be found.
        DiskSpaceError: the destination volume lacks room for the workspace.
        SkalaError: ffmpeg/ffprobe failed or returned unusable output.
        EngineError: the engine failed or produced the wrong number of frames.
    """
    logger = log or _log
    source = Path(input_path)
    validate_input_path(source)
    if source.is_dir():
        raise InputError(f"expected a video file, got a directory: {source}")

    spec = model_spec(model)
    if scale not in VALID_SCALES:
        valid = ", ".join(str(item) for item in VALID_SCALES)
        raise InputError(f"scale must be one of {valid}, got {scale}")
    if audio not in AUDIO_CHOICES:
        valid = ", ".join(AUDIO_CHOICES)
        raise InputError(f"audio must be one of {valid}, got {audio!r}")
    if not 0 <= crf <= 51:
        raise InputError(f"crf must be between 0 and 51, got {crf}")

    requested_output = Path(output_path) if output_path is not None else None
    if requested_output is not None:
        destination = requested_output
    else:
        destination = default_output_path(source, scale, source.suffix or ".mp4")
    if destination.resolve() == source.resolve():
        raise InputError(f"output path must differ from the input path: {destination}")
    ensure_output_path(destination, overwrite=overwrite)

    ffmpeg = _resolve_tool("ffmpeg", ffmpeg_path, logger)
    ffprobe = _resolve_tool("ffprobe", ffprobe_path, logger)
    info = _probe(ffprobe, source, logger)
    plan = plan_scale(info.width, info.height, scale, native_scales=spec.native_scales)

    needed = _estimate_workspace(info, plan, info.expected_frames())
    require_free_space(destination, needed, what=f"upscaling {source.name} to {scale}x")
    logger.debug("estimated scratch requirement: %s", format_bytes(needed))

    warnings = list(plan.warnings)
    if not info.has_audio and audio != "drop":
        warnings.append(f"the source has no audio track; audio setting {audio!r} was ignored")

    base = Path(work_base) if work_base is not None else None
    work = work_dir(source, scale, model, tile, base=base)
    work.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    frames = 0
    try:
        frames_dir = work / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames = _count_frames(frames_dir)
        if frames == 0:
            frames = _extract(ffmpeg, source, frames_dir, logger, expected=info.expected_frames())
        else:
            logger.info("reusing %s previously extracted frames", frames)

        engine_bin = discover_engine(engine_path)
        if model_dir is None:
            model_dir = managed_model_dir(engine_bin)
        upscaled_dir = _upscale_frames(
            engine_bin,
            frames_dir,
            work,
            plan.passes,
            expected=frames,
            model=model,
            tile=tile,
            model_dir=model_dir,
            gpu_id=gpu_id,
            threads=threads,
            log=logger,
        )
        _encode(
            ffmpeg,
            upscaled_dir,
            destination,
            info,
            frames=frames,
            crf=crf,
            video_codec=video_codec,
            audio=audio,
            needs_resize=plan.needs_resize,
            final_size=plan.final_size,
            log=logger,
        )
    except BaseException:
        logger.warning("the run did not finish; keeping %s so it can be resumed", work)
        raise
    else:
        if keep_temp:
            logger.info("temporary files kept at %s", work)
        else:
            shutil.rmtree(work, ignore_errors=True)

    elapsed = time.monotonic() - started
    logger.info("%s -> %s in %s", source.name, destination.name, format_duration(elapsed))
    return VideoResult(
        source=source,
        output=destination,
        info=info,
        scale=scale,
        model=model,
        passes=plan.passes,
        output_size=plan.final_size,
        frames=frames,
        audio=audio,
        resized=plan.needs_resize,
        warnings=tuple(warnings),
        elapsed=elapsed,
    )
