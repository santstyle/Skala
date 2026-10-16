"""Tests for the video pipeline.

The engine, ffmpeg and ffprobe are never invoked for real. ``video._run`` is replaced
by a fake that answers ffprobe with a JSON payload and synthesises PNG frames for the
extraction stage (and writes the encode target), while ``video.run_engine`` nearest-
neighbour resizes a frame directory. ``video._resolve_tool`` is short-circuited so the
tests never touch ``PATH``. This exercises Skala's own probing, planning, resume,
cleanup and error paths without ffmpeg, a GPU, or the real binary.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from skala import engine, scaling, video
from skala.utils import (
    DependencyError,
    DiskSpaceError,
    EngineError,
    InputError,
    OutputExistsError,
    default_output_path,
    work_dir,
)

VIDEO = "realesr-animevideov3"
X4PLUS = "realesrgan-x4plus"
FRAMES = 25


# --------------------------------------------------------------------------------------
# Source and frame helpers
# --------------------------------------------------------------------------------------
def _source(tmp_path: Path, name: str = "clip.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"not-a-real-video")
    return path


def _write_frames(directory: Path, count: int, size: tuple[int, int] = (8, 6)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        Image.new("RGB", size, (10, 20, 30)).save(directory / f"{index:08d}.png")


def _probe_payload(
    *,
    width: int = 8,
    height: int = 6,
    fps: str = "25/1",
    duration: str = "1.0",
    nb_frames: str | None = str(FRAMES),
    has_video: bool = True,
    has_audio: bool = True,
    codec: str = "h264",
    pix_fmt: str = "yuv420p",
) -> dict[str, Any]:
    streams: list[dict[str, Any]] = []
    if has_video:
        streams.append(
            {
                "codec_type": "video",
                "width": width,
                "height": height,
                "avg_frame_rate": fps,
                "r_frame_rate": fps,
                "duration": duration,
                "nb_frames": nb_frames,
                "codec_name": codec,
                "pix_fmt": pix_fmt,
            }
        )
    if has_audio:
        streams.append({"codec_type": "audio", "codec_name": "aac"})
    return {"streams": streams, "format": {"duration": duration}}


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------
class _FakeTools:
    """Stand-in for :func:`skala.video._run` handling ffprobe, extract and encode."""

    def __init__(self, *, frame_count: int = FRAMES, payload: dict[str, Any] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.frame_count = frame_count
        self.payload = (
            payload if payload is not None else _probe_payload(nb_frames=str(frame_count))
        )
        self.raw_probe_stdout: str | None = None
        self.probe_fail = False
        self.extract_fail = False
        self.encode_fail = False
        self.encode_writes = True

    def __call__(self, argv: Sequence[str], log: Any = None) -> subprocess.CompletedProcess[str]:
        recorded = list(argv)
        self.calls.append(recorded)
        if "-print_format" in recorded:
            if self.probe_fail:
                return subprocess.CompletedProcess(recorded, 1, "", "probe boom")
            stdout = self.raw_probe_stdout
            if stdout is None:
                stdout = json.dumps(self.payload)
            return subprocess.CompletedProcess(recorded, 0, stdout, "")
        if "-vsync" in recorded:
            if self.extract_fail:
                return subprocess.CompletedProcess(recorded, 1, "", "extract boom")
            _write_frames(Path(recorded[-1]).parent, self.frame_count)
            return subprocess.CompletedProcess(recorded, 0, "", "")
        if self.encode_fail:
            return subprocess.CompletedProcess(recorded, 1, "", "encode boom")
        if self.encode_writes:
            Path(recorded[-1]).write_bytes(b"video-bytes")
        return subprocess.CompletedProcess(recorded, 0, "", "")

    def probe_calls(self) -> list[list[str]]:
        return [call for call in self.calls if "-print_format" in call]

    def extract_calls(self) -> list[list[str]]:
        return [call for call in self.calls if "-vsync" in call]

    def encode_calls(self) -> list[list[str]]:
        return [call for call in self.calls if "-framerate" in call]


class _FakeEngine:
    """Stand-in for :func:`skala.video.run_engine`."""

    def __init__(self, *, drop_frames: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.drop_frames = drop_frames

    def __call__(
        self,
        argv: Sequence[str],
        *,
        tile_candidates: Sequence[int] = (0,),
        log: Any = None,
    ) -> subprocess.CompletedProcess[str]:
        recorded = list(argv)
        self.calls.append(recorded)
        source_dir = Path(recorded[recorded.index("-i") + 1]).parent
        out_dir = Path(recorded[recorded.index("-o") + 1]).parent
        factor = int(recorded[recorded.index("-s") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(source_dir.glob("*.png"))
        if self.drop_frames:
            frames = frames[: max(0, len(frames) - self.drop_frames)]
        for frame in frames:
            with Image.open(frame) as handle:
                handle.load()
                resized = handle.resize(
                    (handle.width * factor, handle.height * factor), Image.Resampling.NEAREST
                )
            resized.save(out_dir / frame.name, format="PNG")
        return subprocess.CompletedProcess(recorded, 0, "", "")


@pytest.fixture()
def fake_tools(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install fake ffmpeg/ffprobe/engine; call to configure the scenario."""

    def install(
        *,
        frame_count: int = FRAMES,
        payload: dict[str, Any] | None = None,
        drop_frames: int = 0,
    ) -> tuple[_FakeTools, _FakeEngine]:
        tools = _FakeTools(frame_count=frame_count, payload=payload)
        fake_engine = _FakeEngine(drop_frames=drop_frames)
        monkeypatch.setattr(video, "_run", tools)
        monkeypatch.setattr(video, "run_engine", fake_engine)
        monkeypatch.setattr(video, "discover_engine", lambda explicit=None: Path("fake-engine"))
        monkeypatch.setattr(video, "_resolve_tool", lambda name, explicit, log: name)
        return tools, fake_engine

    return install


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------
def test_frame_pattern_uses_printf_glob(tmp_path: Path) -> None:
    assert video._frame_pattern(tmp_path) == str(tmp_path / "%08d.png")


def test_count_frames_ignores_missing_and_non_png(tmp_path: Path) -> None:
    assert video._count_frames(tmp_path / "missing") == 0
    _write_frames(tmp_path / "frames", 3)
    (tmp_path / "frames" / "note.txt").write_text("x")
    assert video._count_frames(tmp_path / "frames") == 3


def test_parse_rate_falls_back_to_25_when_unusable() -> None:
    assert video._parse_rate("30000/1001") == Fraction(30000, 1001)
    assert video._parse_rate("25") == Fraction(25, 1)
    assert video._parse_rate("0/0") == Fraction(25, 1)
    assert video._parse_rate(None) == Fraction(25, 1)
    assert video._parse_rate("garbage") == Fraction(25, 1)


def test_parse_float_rejects_non_positive_and_garbage() -> None:
    assert video._parse_float("1.5") == 1.5
    assert video._parse_float("0") is None
    assert video._parse_float("N/A") is None
    assert video._parse_float(None) is None
    assert video._parse_float("abc") is None


def test_estimate_workspace_counts_source_intermediate_and_final() -> None:
    info = video.VideoInfo(
        path=Path("a.mp4"),
        width=8,
        height=6,
        fps=Fraction(25, 1),
        duration=1.0,
        codec="h264",
        pix_fmt="yuv420p",
        has_audio=True,
        nb_frames=25,
    )
    plan = scaling.plan_scale(8, 6, 4)
    needed = video._estimate_workspace(info, plan, 25)
    assert needed > 25 * 8 * 6
    assert needed >= 16 * 1024 * 1024


def test_video_info_fps_text_and_expected_frames() -> None:
    info = video.VideoInfo(
        path=Path("a.mp4"),
        width=8,
        height=6,
        fps=Fraction(30000, 1001),
        duration=2.0,
        codec="h264",
        pix_fmt="yuv420p",
        has_audio=False,
        nb_frames=100,
    )
    assert info.fps_text == "30000/1001"
    assert info.expected_frames() == 100
    derived = video.VideoInfo(
        path=Path("a.mp4"),
        width=8,
        height=6,
        fps=Fraction(25, 1),
        duration=2.0,
        codec="h264",
        pix_fmt="yuv420p",
        has_audio=False,
        nb_frames=None,
    )
    assert derived.expected_frames() == 50
    empty = video.VideoInfo(
        path=Path("a.mp4"),
        width=8,
        height=6,
        fps=Fraction(25, 1),
        duration=0.0,
        codec="h264",
        pix_fmt="yuv420p",
        has_audio=False,
        nb_frames=None,
    )
    assert empty.expected_frames() == 1


# --------------------------------------------------------------------------------------
# Tool resolution
# --------------------------------------------------------------------------------------
def test_resolve_tool_prefers_existing_explicit_file(tmp_path: Path) -> None:
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"")
    assert video._resolve_tool("ffmpeg", exe, video._log) == str(exe)


def test_resolve_tool_rejects_missing_explicit_path(tmp_path: Path) -> None:
    with pytest.raises(DependencyError):
        video._resolve_tool("ffmpeg", tmp_path / "nope.exe", video._log)


def test_resolve_tool_reports_missing_on_path() -> None:
    with pytest.raises(DependencyError) as excinfo:
        video._resolve_tool("skala-not-a-real-tool", None, video._log)
    assert "skala-not-a-real-tool" in str(excinfo.value)


def test_resolve_tool_includes_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    with pytest.raises(DependencyError) as excinfo:
        video._resolve_tool("ffmpeg", None, video._log)
    assert "ffmpeg" in str(excinfo.value)
    assert "install" in str(excinfo.value).lower()


# --------------------------------------------------------------------------------------
# probe_video
# --------------------------------------------------------------------------------------
def test_probe_video_reports_metadata(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    info = video.probe_video(_source(tmp_path))
    assert info.width == 8
    assert info.height == 6
    assert info.fps == Fraction(25, 1)
    assert info.fps_text == "25/1"
    assert info.has_audio is True
    assert info.codec == "h264"
    assert info.nb_frames == FRAMES


def test_probe_video_defaults_fps_when_stream_reports_zero(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools(payload=_probe_payload(fps="0/0"))
    info = video.probe_video(_source(tmp_path))
    assert info.fps == Fraction(25, 1)


def test_probe_video_rejects_missing_file(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    with pytest.raises(InputError):
        video.probe_video(tmp_path / "absent.mp4")


def test_probe_video_rejects_directory(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    with pytest.raises(InputError):
        video.probe_video(tmp_path)


def test_probe_video_rejects_streamless_file(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools(payload=_probe_payload(has_video=False, has_audio=True))
    with pytest.raises(InputError):
        video.probe_video(_source(tmp_path))


def test_probe_video_raises_on_ffprobe_failure(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    tools.probe_fail = True
    with pytest.raises(video.SkalaError):
        video.probe_video(_source(tmp_path))


def test_probe_video_raises_on_malformed_json(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    tools.raw_probe_stdout = "not json"
    with pytest.raises(video.SkalaError):
        video.probe_video(_source(tmp_path))


# --------------------------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------------------------
def test_upscale_video_4x_single_pass(tmp_path: Path, fake_tools: Any) -> None:
    tools, fake_engine = fake_tools()
    result = video.upscale_video(_source(tmp_path), scale=4, work_base=tmp_path)
    assert len(fake_engine.calls) == 1
    assert len(tools.extract_calls()) == 1
    assert len(tools.encode_calls()) == 1
    assert result.passes == (4,)
    assert result.output_size == (32, 24)
    assert result.frames == FRAMES
    assert result.resized is False
    assert result.output.is_file()


def test_upscale_video_extract_uses_vsync_and_pattern(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    video.upscale_video(_source(tmp_path), scale=4, work_base=tmp_path)
    extract = tools.extract_calls()[0]
    assert extract[extract.index("-vsync") + 1] == "0"
    assert extract[-1].endswith("%08d.png")


def test_upscale_video_copies_audio_by_default(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    video.upscale_video(_source(tmp_path), scale=4, work_base=tmp_path)
    encode = tools.encode_calls()[0]
    assert "-c:a" in encode
    assert encode[encode.index("-c:a") + 1] == "copy"
    assert "1:a:0?" in encode


def test_upscale_video_aac_reencodes_audio(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    video.upscale_video(_source(tmp_path), scale=4, audio="aac", work_base=tmp_path)
    encode = tools.encode_calls()[0]
    assert encode[encode.index("-c:a") + 1] == "aac"


def test_upscale_video_drop_audio_uses_single_input(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    video.upscale_video(_source(tmp_path), scale=4, audio="drop", work_base=tmp_path)
    encode = tools.encode_calls()[0]
    assert "-c:a" not in encode
    assert encode.count("-i") == 1


def test_upscale_video_warns_when_source_has_no_audio(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools(payload=_probe_payload(has_audio=False))
    result = video.upscale_video(_source(tmp_path), scale=4, audio="copy", work_base=tmp_path)
    assert any("no audio" in warning for warning in result.warnings)
    encode = tools.encode_calls()[0]
    assert "-c:a" not in encode


def test_upscale_video_8x_video_model_chains_passes(tmp_path: Path, fake_tools: Any) -> None:
    _, fake_engine = fake_tools()
    result = video.upscale_video(_source(tmp_path), scale=8, model=VIDEO, work_base=tmp_path)
    assert len(fake_engine.calls) == 2
    assert result.passes == (4, 2)
    assert result.resized is False
    assert result.output_size == (64, 48)


def test_upscale_video_8x_x4plus_resizes_via_filter(tmp_path: Path, fake_tools: Any) -> None:
    tools, fake_engine = fake_tools()
    result = video.upscale_video(_source(tmp_path), scale=8, model=X4PLUS, work_base=tmp_path)
    assert len(fake_engine.calls) == 1
    assert result.passes == (4,)
    assert result.resized is True
    assert result.output_size == (64, 48)
    encode = tools.encode_calls()[0]
    assert encode[encode.index("-vf") + 1] == "scale=64:48"


def test_upscale_video_default_output_name(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    source = _source(tmp_path, "clip.mp4")
    result = video.upscale_video(source, scale=4, work_base=tmp_path)
    assert result.output == default_output_path(source, 4, ".mp4")
    assert result.output.name == "clip_skala_4x.mp4"


def test_upscale_video_uses_custom_output_and_cleans_work(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    destination = tmp_path / "out" / "result.mp4"
    destination.parent.mkdir()
    result = video.upscale_video(
        _source(tmp_path), output_path=destination, scale=4, work_base=tmp_path
    )
    assert result.output == destination
    assert destination.is_file()
    assert list((tmp_path).glob("skala_work_*")) == []


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def test_upscale_video_rejects_unknown_model(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    with pytest.raises(InputError):
        video.upscale_video(_source(tmp_path), model="nope", work_base=tmp_path)


def test_upscale_video_rejects_invalid_scale(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    with pytest.raises(InputError):
        video.upscale_video(_source(tmp_path), scale=5, work_base=tmp_path)


def test_upscale_video_rejects_invalid_audio(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    with pytest.raises(InputError):
        video.upscale_video(_source(tmp_path), audio="flac", work_base=tmp_path)


def test_upscale_video_rejects_out_of_range_crf(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    with pytest.raises(InputError):
        video.upscale_video(_source(tmp_path), crf=99, work_base=tmp_path)


def test_upscale_video_rejects_output_equal_to_input(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    source = _source(tmp_path)
    with pytest.raises(InputError):
        video.upscale_video(source, output_path=source, scale=4, work_base=tmp_path)


def test_upscale_video_refuses_existing_output(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    source = _source(tmp_path)
    default_output_path(source, 4, ".mp4").write_bytes(b"")
    with pytest.raises(OutputExistsError):
        video.upscale_video(source, scale=4, work_base=tmp_path)


def test_upscale_video_replaces_output_when_overwrite(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    source = _source(tmp_path)
    destination = default_output_path(source, 4, ".mp4")
    destination.write_bytes(b"")
    result = video.upscale_video(source, scale=4, overwrite=True, work_base=tmp_path)
    assert result.output == destination
    assert destination.stat().st_size > 0


# --------------------------------------------------------------------------------------
# Failure handling, resume and cleanup
# --------------------------------------------------------------------------------------
def test_upscale_video_ffmpeg_extract_failure_keeps_work(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    tools.extract_fail = True
    base = tmp_path / "work"
    with pytest.raises(video.SkalaError):
        video.upscale_video(_source(tmp_path), scale=4, work_base=base)
    assert list(base.glob("skala_work_*"))


def test_upscale_video_encode_failure_keeps_work(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    tools.encode_fail = True
    base = tmp_path / "work"
    with pytest.raises(video.SkalaError):
        video.upscale_video(_source(tmp_path), scale=4, work_base=base)
    keeps = list(base.glob("skala_work_*"))
    assert keeps and (keeps[0] / "frames").is_dir()


def test_upscale_video_encode_writing_nothing_is_error(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    tools.encode_writes = False
    with pytest.raises(video.SkalaError):
        video.upscale_video(_source(tmp_path), scale=4, work_base=tmp_path)


def test_upscale_video_engine_dropping_frames_is_error(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools(drop_frames=1)
    base = tmp_path / "work"
    with pytest.raises(EngineError):
        video.upscale_video(_source(tmp_path), scale=4, work_base=base)
    keeps = list(base.glob("skala_work_*"))
    assert keeps and (keeps[0] / "frames").is_dir()


def test_upscale_video_resumes_from_existing_frames(tmp_path: Path, fake_tools: Any) -> None:
    tools, _ = fake_tools()
    source = _source(tmp_path)
    work = work_dir(source, 4, VIDEO, 0, base=tmp_path)
    _write_frames(work / "frames", FRAMES)
    result = video.upscale_video(source, scale=4, work_base=tmp_path)
    assert tools.extract_calls() == []
    assert result.frames == FRAMES


def test_upscale_video_missing_engine_propagates(
    tmp_path: Path, fake_tools: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tools()

    def _boom(explicit: Any = None) -> Path:
        raise DependencyError("no engine")

    monkeypatch.setattr(video, "discover_engine", _boom)
    with pytest.raises(DependencyError):
        video.upscale_video(_source(tmp_path), scale=4, work_base=tmp_path)


def test_upscale_video_disk_space_error_propagates(
    tmp_path: Path, fake_tools: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tools()

    def _boom(path: Path, needed_bytes: int, *, what: str) -> None:
        raise DiskSpaceError("no room")

    monkeypatch.setattr(video, "require_free_space", _boom)
    with pytest.raises(DiskSpaceError):
        video.upscale_video(_source(tmp_path), scale=4, work_base=tmp_path)


def test_upscale_video_keeps_temp_when_requested(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    base = tmp_path / "work"
    video.upscale_video(_source(tmp_path), scale=4, keep_temp=True, work_base=base)
    keeps = list(base.glob("skala_work_*"))
    assert keeps and (keeps[0] / "frames").is_dir()


# --------------------------------------------------------------------------------------
# Result surface
# --------------------------------------------------------------------------------------
def test_result_describe_mentions_scale_and_size(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    result = video.upscale_video(_source(tmp_path), scale=4, work_base=tmp_path)
    text = result.describe()
    assert "8x6" in text
    assert "32x24" in text
    assert "4x" in text
    assert "audio: copy" in text


def test_video_result_and_info_are_frozen(tmp_path: Path, fake_tools: Any) -> None:
    fake_tools()
    result = video.upscale_video(_source(tmp_path), scale=4, work_base=tmp_path)
    with pytest.raises(FrozenInstanceError):
        result.scale = 8  # type: ignore[misc]
    assert engine.model_spec(VIDEO).native_scales == (2, 3, 4)
