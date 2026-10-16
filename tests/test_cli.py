"""Tests for the Skala command-line interface.

The CLI is exercised through Typer's :class:`CliRunner`. Every external
dependency (the engine and ffmpeg/ffprobe) is mocked so the tests need no native
tooling. Assertions run against ``output + stderr`` because the two streams are
kept separate by click and error text is written to stderr.
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import skala.cli as cli
from PIL import Image
from skala import __version__
from skala.cli import app
from skala.config import SkalaConfig
from skala.engine import DEFAULT_IMAGE_MODEL, DEFAULT_VIDEO_MODEL, ENGINE_BINARY
from skala.image import ImageResult
from skala.setup import SetupResult
from skala.utils import DependencyError, EngineError, HardwareStatus, InputError, MissingDependency
from skala.video import (
    DEFAULT_CRF,
    DEFAULT_VIDEO_CODEC,
    VideoInfo,
    VideoResult,
)
from typer.testing import CliRunner

RUNNER = CliRunner()


def _text(result: Any) -> str:
    """Return stdout and stderr joined, since errors are written to stderr."""
    return result.output + result.stderr


def _png(path: Path, size: tuple[int, int] = (12, 8)) -> Path:
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def _image_result(
    source: Path, output: Path | None = None, *, scale: int = 4, model: str = DEFAULT_IMAGE_MODEL
) -> ImageResult:
    out = output if output is not None else source
    width, height = 12, 8
    return ImageResult(
        source=source,
        output=out,
        input_size=(width, height),
        output_size=(width * scale, height * scale),
        scale=scale,
        model=model,
        passes=(scale,),
        format="png",
        resized=False,
        alpha_handled="kept",
        warnings=(),
        elapsed=0.01,
    )


def _video_result(
    source: Path, output: Path, *, scale: int = 4, model: str = DEFAULT_VIDEO_MODEL
) -> VideoResult:
    info = VideoInfo(
        path=source,
        width=16,
        height=16,
        fps=Fraction(30, 1),
        duration=1.0,
        codec="h264",
        pix_fmt="yuv420p",
        has_audio=True,
        nb_frames=30,
    )
    return VideoResult(
        source=source,
        output=output,
        info=info,
        scale=scale,
        model=model,
        passes=(scale,),
        output_size=(16 * scale, 16 * scale),
        frames=30,
        audio="copy",
        resized=False,
        warnings=(),
        elapsed=0.2,
    )


class _ImageSpy:
    """Stand-in for :func:`skala.cli.upscale_image` that records its calls."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def __call__(self, input_path: Path, **kwargs: Any) -> ImageResult:
        self.calls.append({"input": Path(input_path), **kwargs})
        if self.error is not None:
            raise self.error
        output = kwargs.get("output_path")
        return _image_result(
            Path(input_path),
            Path(output) if output is not None else None,
            scale=int(kwargs.get("scale", 4)),
            model=str(kwargs.get("model", DEFAULT_IMAGE_MODEL)),
        )


class _VideoSpy:
    """Stand-in for :func:`skala.cli.upscale_video` that records its calls."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def __call__(self, input_path: Path, **kwargs: Any) -> VideoResult:
        self.calls.append({"input": Path(input_path), **kwargs})
        if self.error is not None:
            raise self.error
        output = kwargs.get("output_path") or input_path
        return _video_result(
            Path(input_path),
            Path(output),
            scale=int(kwargs.get("scale", 4)),
            model=str(kwargs.get("model", DEFAULT_VIDEO_MODEL)),
        )


def _patch_config(monkeypatch: pytest.MonkeyPatch, config: SkalaConfig) -> dict[str, Any]:
    """Replace ``cli.load_config`` with a stub; return a dict of seen arguments."""
    seen: dict[str, Any] = {}

    def fake(path: Path | None = None, *, cwd: Path | None = None) -> SkalaConfig:
        seen["path"] = path
        return config

    monkeypatch.setattr(cli, "load_config", fake)
    return seen


# --------------------------------------------------------------------------- dispatch


def test_version_flag() -> None:
    result = RUNNER.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"skala {__version__}" in result.output


def test_no_args_prints_help() -> None:
    result = RUNNER.invoke(app, [])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "doctor" in result.output


def test_help_exits_zero() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_run_without_source_prints_help() -> None:
    result = RUNNER.invoke(app, ["run"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_default_argument_routes_to_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "photo.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 0
    assert len(spy.calls) == 1


# --------------------------------------------------------------------------- interactive


def test_bare_invocation_skips_interactive_without_tty() -> None:
    # CliRunner's stdin is not a TTY, so a bare `skala` prints help, not prompts.
    result = RUNNER.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output
    assert "File or folder to upscale" not in result.output


def test_interactive_prompts_for_source_then_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _png(tmp_path / "photo.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    monkeypatch.setattr(cli, "_wants_interactive", lambda: True)
    result = RUNNER.invoke(app, [], input=f"{source}\n2\n")
    assert result.exit_code == 0, _text(result)
    assert len(spy.calls) == 1
    assert spy.calls[0]["input"] == source
    assert spy.calls[0]["scale"] == 2


def test_interactive_retries_after_a_bad_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _png(tmp_path / "photo.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    monkeypatch.setattr(cli, "_wants_interactive", lambda: True)
    missing = tmp_path / "nope.png"
    result = RUNNER.invoke(app, [], input=f"{missing}\n{source}\n3\n")
    assert result.exit_code == 0, _text(result)
    assert len(spy.calls) == 1
    assert spy.calls[0]["scale"] == 3


def test_interactive_rejects_an_invalid_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _png(tmp_path / "photo.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    monkeypatch.setattr(cli, "_wants_interactive", lambda: True)
    result = RUNNER.invoke(app, [], input=f"{source}\n5\n8\n")
    assert result.exit_code == 0, _text(result)
    assert spy.calls[0]["scale"] == 8


def test_interactive_blank_scale_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _png(tmp_path / "photo.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    monkeypatch.setattr(cli, "_wants_interactive", lambda: True)
    result = RUNNER.invoke(app, [], input=f"{source}\n\n")
    assert result.exit_code == 0, _text(result)
    assert spy.calls[0]["scale"] == 4


def test_interactive_abort_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_wants_interactive", lambda: True)
    result = RUNNER.invoke(app, [], input="")
    assert result.exit_code == 130


# --------------------------------------------------------------------------- doctor


def test_doctor_reports_available_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_hardware", lambda: HardwareStatus(True, "1.3.0", 8))
    monkeypatch.setattr(cli, "missing_ffmpeg", lambda: [])
    monkeypatch.setattr(
        cli, "discover_engine", lambda explicit=None: Path("/opt/realesrgan-ncnn-vulkan")
    )
    result = RUNNER.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "python" in result.output
    assert "vulkan: yes (1.3.0)" in result.output
    assert "cpu cores: 8" in result.output
    assert "ffmpeg/ffprobe: found" in result.output
    assert f"engine: {Path('/opt/realesrgan-ncnn-vulkan')}" in result.output
    assert "models:" in result.output


def test_doctor_reports_missing_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_hardware", lambda: HardwareStatus(False, "unavailable", 4))
    monkeypatch.setattr(cli, "missing_ffmpeg", lambda: [])

    def _raise(explicit: Path | None = None) -> Path:
        raise DependencyError("engine not on PATH")

    monkeypatch.setattr(cli, "discover_engine", _raise)
    result = RUNNER.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "vulkan: no (unavailable)" in result.output
    assert "engine: not found" in _text(result)


def test_doctor_reports_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_hardware", lambda: HardwareStatus(None, "unknown", 2))
    monkeypatch.setattr(
        cli,
        "missing_ffmpeg",
        lambda: [MissingDependency("ffmpeg", "winget install ffmpeg", "apt install ffmpeg")],
    )
    monkeypatch.setattr(cli, "discover_engine", lambda explicit=None: Path("/e"))
    result = RUNNER.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "vulkan: unknown (unknown)" in result.output
    assert "ffmpeg tool missing: ffmpeg" in _text(result)


# --------------------------------------------------------------------------- setup


def test_setup_dry_run_prints_plan_without_writing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    result = RUNNER.invoke(app, ["setup", "--dry-run", "--home", str(home)])
    assert result.exit_code == 0
    assert "[dry-run] would install" in result.output
    assert "source: the newest release" in result.output
    assert not home.exists()


def test_setup_dry_run_notes_existing_engine(tmp_path: Path) -> None:
    home = tmp_path / "home"
    engine_dir = home / "engine"
    engine_dir.mkdir(parents=True)
    name = f"{ENGINE_BINARY}.exe" if os.name == "nt" else ENGINE_BINARY
    (engine_dir / name).write_text("stub")
    result = RUNNER.invoke(app, ["setup", "--dry-run", "--home", str(home)])
    assert result.exit_code == 0
    assert "already exists and --force would replace it" in result.output


def test_setup_reports_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**kwargs: Any) -> SetupResult:
        raise DependencyError("no engine release found")

    monkeypatch.setattr(cli, "setup_engine", _raise)
    result = RUNNER.invoke(app, ["setup"])
    assert result.exit_code == 1
    assert "no engine release found" in _text(result)


# --------------------------------------------------------------------------- image


def test_run_image_uses_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "photo.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 0
    call = spy.calls[0]
    assert call["input"] == source
    assert call["scale"] == 4
    assert call["model"] == DEFAULT_IMAGE_MODEL
    assert call["tile"] == 0
    assert call["overwrite"] is False
    assert call["engine_path"] is None
    assert call["output_path"] is None


def test_run_image_reports_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "photo.png")
    monkeypatch.setattr(cli, "upscale_image", _ImageSpy())
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 0
    assert result.output.strip()


def test_run_image_scale_and_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "photo.png")
    output = tmp_path / "big.png"
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    result = RUNNER.invoke(app, [str(source), "--scale", "2", "--output", str(output)])
    assert result.exit_code == 0
    assert spy.calls[0]["scale"] == 2
    assert spy.calls[0]["output_path"] == output


def test_run_image_output_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "photo.png")
    output_dir = tmp_path / "out"
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    result = RUNNER.invoke(app, [str(source), "--output", str(output_dir)])
    assert result.exit_code == 0
    assert output_dir.is_dir()
    assert spy.calls[0]["output_path"] == output_dir / "photo_skala_4x.png"


# --------------------------------------------------------------------------- video


def test_run_video_uses_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\x00")
    spy = _VideoSpy()
    monkeypatch.setattr(cli, "upscale_video", spy)
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 0
    call = spy.calls[0]
    assert call["input"] == source
    assert call["scale"] == 4
    assert call["model"] == DEFAULT_VIDEO_MODEL
    assert call["audio"] == "copy"
    assert call["crf"] == DEFAULT_CRF
    assert call["video_codec"] == DEFAULT_VIDEO_CODEC
    assert call["keep_temp"] is False


# --------------------------------------------------------------------------- folders


def test_run_folder_upscales_every_media_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    _png(folder / "a.png")
    Image.new("RGB", (10, 10), (1, 2, 3)).save(folder / "b.jpg")
    (folder / "notes.txt").write_text("ignore me", encoding="utf-8")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    result = RUNNER.invoke(app, [str(folder)])
    assert result.exit_code == 0
    assert len(spy.calls) == 2
    assert "upscaled 2 of 2" in result.output


def test_run_folder_output_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    _png(folder / "a.png")
    output_dir = tmp_path / "out"
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    result = RUNNER.invoke(app, [str(folder), "--output", str(output_dir), "--scale", "2"])
    assert result.exit_code == 0
    assert output_dir.is_dir()
    assert spy.calls[0]["output_path"] == output_dir / "a_skala_2x.png"


def test_run_folder_rejects_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    _png(folder / "a.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    result = RUNNER.invoke(app, [str(folder), "--output", str(tmp_path / "one.png")])
    assert result.exit_code == 1
    assert "--output must be a directory" in _text(result)
    assert spy.calls == []


def test_run_type_filter_selects_only_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = tmp_path / "mixed"
    folder.mkdir()
    _png(folder / "a.png")
    (folder / "b.mp4").write_bytes(b"\x00")
    image_spy = _ImageSpy()
    video_spy = _VideoSpy()
    monkeypatch.setattr(cli, "upscale_image", image_spy)
    monkeypatch.setattr(cli, "upscale_video", video_spy)
    result = RUNNER.invoke(app, [str(folder), "--type", "image"])
    assert result.exit_code == 0
    assert len(image_spy.calls) == 1
    assert video_spy.calls == []


def test_run_empty_folder_errors(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    result = RUNNER.invoke(app, [str(folder)])
    assert result.exit_code == 1
    assert "no image or video files found" in _text(result)


# --------------------------------------------------------------------------- validation


def test_unknown_extension_errors(tmp_path: Path) -> None:
    source = tmp_path / "data.xyz"
    source.write_bytes(b"\x00")
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 1
    assert "cannot tell whether" in _text(result)


def test_invalid_type_is_usage_error(tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    result = RUNNER.invoke(app, [str(source), "--type", "bogus"])
    assert result.exit_code == 2


def test_invalid_scale_errors(tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    result = RUNNER.invoke(app, [str(source), "--scale", "5"])
    assert result.exit_code == 1
    assert "scale must be one of 2, 3, 4, 8" in _text(result)


def test_invalid_audio_errors(tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    result = RUNNER.invoke(app, [str(source), "--audio", "flac"])
    assert result.exit_code == 1
    assert "audio must be one of" in _text(result)


def test_missing_input_errors(tmp_path: Path) -> None:
    result = RUNNER.invoke(app, [str(tmp_path / "nope.png")])
    assert result.exit_code == 1
    assert "Input not found" in _text(result)


# --------------------------------------------------------------------------- dry run


def test_dry_run_image_skips_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")

    def _boom(*args: Any, **kwargs: Any) -> ImageResult:
        raise AssertionError("upscale_image must not run during --dry-run")

    monkeypatch.setattr(cli, "upscale_image", _boom)
    result = RUNNER.invoke(app, [str(source), "--dry-run", "--scale", "2"])
    assert result.exit_code == 0
    assert "[dry-run] image a.png" in result.output
    assert "2x" in result.output


def test_dry_run_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\x00")

    def _probe(path: Path, **kwargs: Any) -> VideoInfo:
        return VideoInfo(
            path=Path(path),
            width=64,
            height=32,
            fps=Fraction(30, 1),
            duration=1.0,
            codec="h264",
            pix_fmt="yuv420p",
            has_audio=False,
            nb_frames=30,
        )

    monkeypatch.setattr(cli, "probe_video", _probe)
    result = RUNNER.invoke(app, [str(source), "--dry-run"])
    assert result.exit_code == 0
    assert "[dry-run] video clip.mp4" in result.output


def test_dry_run_folder_lists_each_file(tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    _png(folder / "a.png")
    _png(folder / "b.png")
    result = RUNNER.invoke(app, [str(folder), "--dry-run"])
    assert result.exit_code == 0
    assert result.output.count("[dry-run]") == 2


# --------------------------------------------------------------------------- failures


def test_single_failure_exits_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    monkeypatch.setattr(cli, "upscale_image", _ImageSpy(error=EngineError("engine exploded")))
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 1
    assert "engine exploded" in _text(result)


def test_folder_failure_continues_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    _png(folder / "a.png")
    _png(folder / "b.png")
    monkeypatch.setattr(cli, "upscale_image", _ImageSpy(error=EngineError("nope")))
    result = RUNNER.invoke(app, [str(folder)])
    assert result.exit_code == 1
    assert "upscaled 0 of 2" in result.output
    # stderr holds exactly one line per failed target; output mirrors it.
    assert result.stderr.count("nope") == 2


def test_keyboard_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    monkeypatch.setattr(cli, "upscale_image", _ImageSpy(error=KeyboardInterrupt()))
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 130


# --------------------------------------------------------------------------- config


def test_config_model_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    _patch_config(monkeypatch, SkalaConfig(model="realesrgan-x4plus-anime"))
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 0
    assert spy.calls[0]["model"] == "realesrgan-x4plus-anime"


def test_cli_model_overrides_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    _patch_config(monkeypatch, SkalaConfig(model="from-config"))
    result = RUNNER.invoke(app, [str(source), "--model", "from-cli"])
    assert result.exit_code == 0
    assert spy.calls[0]["model"] == "from-cli"


def test_explicit_config_path_is_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    config_file = tmp_path / "skala.toml"
    config_file.write_text("[skala]\n", encoding="utf-8")
    seen = _patch_config(monkeypatch, SkalaConfig())
    result = RUNNER.invoke(app, [str(source), "--config", str(config_file)])
    assert result.exit_code == 0
    assert seen["path"] == config_file


def test_config_image_options_are_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _png(tmp_path / "a.png")
    spy = _ImageSpy()
    monkeypatch.setattr(cli, "upscale_image", spy)
    _patch_config(
        monkeypatch,
        SkalaConfig(
            tile=200,
            gpu_id=1,
            threads=4,
            overwrite=True,
            engine_path="E:/eng.exe",
            workdir="W:/work",
        ),
    )
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 0
    call = spy.calls[0]
    assert call["tile"] == 200
    assert call["gpu_id"] == 1
    assert call["threads"] == 4
    assert call["overwrite"] is True
    assert call["engine_path"] == "E:/eng.exe"
    assert call["work_base"] == "W:/work"


def test_config_video_options_are_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\x00")
    spy = _VideoSpy()
    monkeypatch.setattr(cli, "upscale_video", spy)
    _patch_config(
        monkeypatch,
        SkalaConfig(
            crf=23,
            video_codec="libx265",
            audio="drop",
            ffmpeg_path="F:/ffmpeg.exe",
            ffprobe_path="G:/ffprobe.exe",
            keep_temp=True,
        ),
    )
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 0
    call = spy.calls[0]
    assert call["crf"] == 23
    assert call["video_codec"] == "libx265"
    assert call["audio"] == "drop"
    assert call["ffmpeg_path"] == "F:/ffmpeg.exe"
    assert call["ffprobe_path"] == "G:/ffprobe.exe"
    assert call["keep_temp"] is True


def test_config_load_error_exits_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")

    def _raise(path: Path | None = None, *, cwd: Path | None = None) -> SkalaConfig:
        raise InputError("bad skala.toml")

    monkeypatch.setattr(cli, "load_config", _raise)
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 1
    assert "bad skala.toml" in _text(result)


def test_verbose_flag_enables_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    monkeypatch.setattr(cli, "upscale_image", _ImageSpy())
    seen: list[bool] = []
    monkeypatch.setattr(cli, "setup_logging", lambda verbose: seen.append(verbose))
    result = RUNNER.invoke(app, [str(source), "--verbose"])
    assert result.exit_code == 0
    assert seen == [True]


def test_verbose_defaults_to_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    monkeypatch.setattr(cli, "upscale_image", _ImageSpy())
    seen: list[bool] = []
    monkeypatch.setattr(cli, "setup_logging", lambda verbose: seen.append(verbose))
    result = RUNNER.invoke(app, [str(source)])
    assert result.exit_code == 0
    assert seen == [False]
