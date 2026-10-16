"""Tests for the single-image pipeline.

The engine is never invoked for real: ``image.discover_engine`` and ``image.run_engine``
are monkeypatched so the tests exercise Skala's own planning, alpha handling, resizing,
formatting and cleanup logic without a GPU or the real binary.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from skala import engine, image
from skala.utils import (
    DependencyError,
    DiskSpaceError,
    EngineError,
    InputError,
    OutputExistsError,
    default_output_path,
)

X4PLUS = "realesrgan-x4plus"
VIDEO = "realesr-animevideov3"


# --------------------------------------------------------------------------------------
# Source helpers
# --------------------------------------------------------------------------------------
def _rgb(path: Path, size: tuple[int, int] = (8, 6)) -> Path:
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def _split_alpha(path: Path, size: tuple[int, int] = (8, 6)) -> Path:
    """RGBA image: left half fully transparent, right half fully opaque."""
    rgb = Image.new("RGB", size, (10, 20, 30))
    alpha = Image.new("L", size, 0)
    for x in range(size[0] // 2, size[0]):
        for y in range(size[1]):
            alpha.putpixel((x, y), 255)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    rgba.save(path)
    return path


def _gray(path: Path, size: tuple[int, int] = (5, 4)) -> Path:
    Image.new("L", size, 120).save(path)
    return path


# --------------------------------------------------------------------------------------
# Fake engine
# --------------------------------------------------------------------------------------
class _FakeEngine:
    """Stand-in for :func:`skala.engine.run_engine`.

    Reads the engine argv, nearest-neighbour resizes the input by the requested factor
    and writes a PNG. ``keep_alpha=False`` mimics a build that silently drops alpha.
    """

    def __init__(self, *, keep_alpha: bool = True, write: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.keep_alpha = keep_alpha
        self.write = write

    def __call__(
        self,
        argv: Sequence[str],
        *,
        tile_candidates: Sequence[int] = (0,),
        log: Any = None,
    ) -> subprocess.CompletedProcess[str]:
        recorded = list(argv)
        self.calls.append(recorded)
        if not self.write:
            return subprocess.CompletedProcess(recorded, 0, "", "")
        source = Path(recorded[recorded.index("-i") + 1])
        destination = Path(recorded[recorded.index("-o") + 1])
        factor = int(recorded[recorded.index("-s") + 1])
        with Image.open(source) as handle:
            handle.load()
            result = handle.copy() if self.keep_alpha else handle.convert("RGB")
        result = result.resize(
            (result.width * factor, result.height * factor), Image.Resampling.NEAREST
        )
        result.save(destination, format="PNG")
        return subprocess.CompletedProcess(recorded, 0, "", "")


@pytest.fixture()
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fake engine; call it to configure alpha/behaviour."""

    def install(*, keep_alpha: bool = True, write: bool = True) -> _FakeEngine:
        fake = _FakeEngine(keep_alpha=keep_alpha, write=write)
        monkeypatch.setattr(image, "run_engine", fake)
        monkeypatch.setattr(image, "discover_engine", lambda explicit=None: Path("fake-engine"))
        return fake

    return install


# --------------------------------------------------------------------------------------
# inspect_image / pure helpers
# --------------------------------------------------------------------------------------
def test_inspect_image_reports_mode_and_size(tmp_path: Path) -> None:
    info = image.inspect_image(_rgb(tmp_path / "a.png", (12, 7)))
    assert info.size == (12, 7)
    assert info.mode == "RGB"
    assert info.has_alpha is False
    assert info.pixels == 84


def test_inspect_image_detects_alpha(tmp_path: Path) -> None:
    info = image.inspect_image(_split_alpha(tmp_path / "a.png"))
    assert info.mode == "RGBA"
    assert info.has_alpha is True


def test_inspect_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        image.inspect_image(tmp_path / "nope.png")


def test_inspect_image_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        image.inspect_image(tmp_path)


def test_has_alpha_detects_palette_transparency() -> None:
    palette = Image.new("P", (2, 2))
    palette.info["transparency"] = 0
    assert image._has_alpha(palette) is True


def test_target_mode_maps_grayscale_and_alpha() -> None:
    assert image._target_mode("L", False) == "L"
    assert image._target_mode("I;16", False) == "L"
    assert image._target_mode("RGB", False) == "RGB"
    assert image._target_mode("P", True) == "RGBA"


def test_depth_warning_only_for_wide_modes() -> None:
    assert image._depth_warning("RGB") is None
    assert image._depth_warning("L") is None
    assert "8-bit" in (image._depth_warning("I;16") or "")


def test_resolve_format_prefers_explicit_and_detects_extension(tmp_path: Path) -> None:
    source = tmp_path / "in.png"
    assert image._resolve_format("webp", Path("a.jpg"), source) == "webp"
    assert image._resolve_format(None, Path("a.jpeg"), source) == "jpg"
    assert image._resolve_format(None, Path("a.PNG"), source) == "png"
    assert image._resolve_format(None, None, source) == "png"
    assert image._resolve_format(None, None, tmp_path / "in.tiff") == "png"


def test_resolve_format_rejects_unsupported_output_extension(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        image._resolve_format(None, Path("a.bmp"), tmp_path / "in.png")


def test_resolve_format_rejects_unknown_explicit_format(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        image._resolve_format("gif", None, tmp_path / "in.png")


# --------------------------------------------------------------------------------------
# Planning: single pass, native, resize and chained passes
# --------------------------------------------------------------------------------------
def test_4x_rgb_uses_one_native_pass(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine()
    result = image.upscale_image(_rgb(tmp_path / "a.png"), scale=4, work_base=tmp_path)
    assert len(fake.calls) == 1
    assert result.passes == (4,)
    assert result.resized is False
    assert result.output_size == (32, 24)
    assert result.alpha_handled == "none"
    assert result.format == "png"


def test_2x_with_x4plus_resizes_down(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine()
    result = image.upscale_image(
        _rgb(tmp_path / "a.png"), scale=2, model=X4PLUS, work_base=tmp_path
    )
    assert len(fake.calls) == 1
    assert result.resized is True
    assert result.output_size == (16, 12)


def test_2x_with_video_model_is_native(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine()
    result = image.upscale_image(_rgb(tmp_path / "a.png"), scale=2, model=VIDEO, work_base=tmp_path)
    assert len(fake.calls) == 1
    assert result.passes == (2,)
    assert result.resized is False


def test_8x_with_x4plus_resizes_and_warns(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine()
    result = image.upscale_image(
        _rgb(tmp_path / "a.png"), scale=8, model=X4PLUS, work_base=tmp_path
    )
    assert len(fake.calls) == 1
    assert result.passes == (4,)
    assert result.resized is True
    assert result.output_size == (64, 48)
    assert result.warnings


def test_8x_with_video_model_chains_two_passes(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine()
    result = image.upscale_image(_rgb(tmp_path / "a.png"), scale=8, model=VIDEO, work_base=tmp_path)
    assert len(fake.calls) == 2
    assert result.passes == (4, 2)
    assert result.resized is False
    assert result.output_size == (64, 48)


# --------------------------------------------------------------------------------------
# Alpha handling
# --------------------------------------------------------------------------------------
def test_alpha_is_kept_by_a_single_pass(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine(keep_alpha=True)
    source = _split_alpha(tmp_path / "a.png")
    result = image.upscale_image(source, scale=4, work_base=tmp_path)
    assert len(fake.calls) == 1
    assert result.alpha_handled == "engine"
    with Image.open(result.output) as handle:
        assert handle.mode == "RGBA"
        assert handle.getpixel((0, 0)) == (10, 20, 30, 0)
        assert handle.getpixel((31, 0)) == (10, 20, 30, 255)


def test_alpha_falls_back_to_a_second_pass(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine(keep_alpha=False)
    source = _split_alpha(tmp_path / "a.png")
    result = image.upscale_image(source, scale=4, work_base=tmp_path)
    assert len(fake.calls) == 2
    assert result.alpha_handled == "fallback"
    with Image.open(result.output) as handle:
        assert handle.mode == "RGBA"
        alpha = handle.getchannel("A")
        assert alpha.getextrema() == (0, 255)
        assert alpha.getpixel((0, 0)) == 0
        assert alpha.getpixel((31, 0)) == 255


# --------------------------------------------------------------------------------------
# Modes and formats
# --------------------------------------------------------------------------------------
def test_grayscale_source_stays_grayscale(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine()
    result = image.upscale_image(_gray(tmp_path / "a.png"), scale=4, work_base=tmp_path)
    assert len(fake.calls) == 1
    with Image.open(result.output) as handle:
        assert handle.mode == "L"
        assert handle.size == (20, 16)


def test_jpeg_flattens_transparency_onto_white(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine(keep_alpha=True)
    source = _split_alpha(tmp_path / "a.png")
    result = image.upscale_image(source, scale=4, fmt="jpg", work_base=tmp_path)
    assert result.format == "jpg"
    assert result.output.suffix == ".jpg"
    assert any("JPEG" in warning for warning in result.warnings)
    with Image.open(result.output) as handle:
        assert handle.mode == "RGB"
        assert handle.getpixel((0, 0)) == (255, 255, 255)
        assert handle.getpixel((31, 0)) == (10, 20, 30)


def test_default_output_name(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine()
    source = _rgb(tmp_path / "photo.png")
    result = image.upscale_image(source, scale=4, work_base=tmp_path)
    assert result.output == default_output_path(source, 4, ".png")
    assert result.output.name == "photo_skala_4x.png"


# --------------------------------------------------------------------------------------
# Output policy and validation
# --------------------------------------------------------------------------------------
def test_existing_output_is_refused_without_overwrite(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine()
    source = _rgb(tmp_path / "photo.png")
    default_output_path(source, 4, ".png").write_bytes(b"")
    with pytest.raises(OutputExistsError):
        image.upscale_image(source, scale=4, work_base=tmp_path)


def test_existing_output_is_replaced_when_allowed(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine()
    source = _rgb(tmp_path / "photo.png")
    destination = default_output_path(source, 4, ".png")
    destination.write_bytes(b"")
    result = image.upscale_image(source, scale=4, overwrite=True, work_base=tmp_path)
    assert result.output == destination
    assert destination.stat().st_size > 0


def test_output_equal_to_input_is_rejected(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine()
    source = _rgb(tmp_path / "photo.png")
    with pytest.raises(InputError):
        image.upscale_image(source, output_path=source, scale=4, work_base=tmp_path)


def test_unknown_model_is_rejected(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine()
    source = _rgb(tmp_path / "photo.png")
    with pytest.raises(InputError):
        image.upscale_image(source, model="nope", work_base=tmp_path)


def test_invalid_scale_is_rejected(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine()
    source = _rgb(tmp_path / "photo.png")
    with pytest.raises(InputError):
        image.upscale_image(source, scale=5, work_base=tmp_path)


# --------------------------------------------------------------------------------------
# Failure propagation, engine contract, cleanup
# --------------------------------------------------------------------------------------
def test_missing_engine_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _rgb(tmp_path / "photo.png")

    def _boom(explicit: Any = None) -> Path:
        raise DependencyError("no engine")

    monkeypatch.setattr(image, "discover_engine", _boom)
    with pytest.raises(DependencyError):
        image.upscale_image(source, work_base=tmp_path)


def test_disk_space_error_propagates(
    tmp_path: Path, fake_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_engine()
    source = _rgb(tmp_path / "photo.png")

    def _boom(path: Path, needed_bytes: int, *, what: str) -> None:
        raise DiskSpaceError("no room")

    monkeypatch.setattr(image, "require_free_space", _boom)
    with pytest.raises(DiskSpaceError):
        image.upscale_image(source, work_base=tmp_path)


def test_engine_writing_nothing_is_an_error(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine(write=False)
    source = _rgb(tmp_path / "photo.png")
    with pytest.raises(EngineError):
        image.upscale_image(source, work_base=tmp_path)


def test_no_format_flag_is_passed_to_the_engine(tmp_path: Path, fake_engine: Any) -> None:
    fake = fake_engine()
    source = _split_alpha(tmp_path / "a.png")
    image.upscale_image(source, scale=4, fmt="webp", work_base=tmp_path)
    assert fake.calls
    assert all("-f" not in argv for argv in fake.calls)
    assert all(argv[argv.index("-o") + 1].endswith(".png") for argv in fake.calls)


def test_work_directory_is_cleaned_up(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine()
    base = tmp_path / "work"
    image.upscale_image(_rgb(tmp_path / "a.png"), scale=4, work_base=base)
    assert base.is_dir()
    assert list(base.iterdir()) == []


def test_work_directory_is_cleaned_up_on_failure(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine(write=False)
    base = tmp_path / "work"
    with pytest.raises(EngineError):
        image.upscale_image(_rgb(tmp_path / "a.png"), scale=4, work_base=base)
    assert base.is_dir()
    assert list(base.iterdir()) == []


def test_result_describe_mentions_scale_and_size(tmp_path: Path, fake_engine: Any) -> None:
    fake_engine()
    result = image.upscale_image(_rgb(tmp_path / "a.png"), scale=4, work_base=tmp_path)
    text = result.describe()
    assert "8x6" in text
    assert "32x24" in text
    assert "4x" in text


def test_model_spec_matches_engine_registry() -> None:
    assert engine.model_spec(X4PLUS).native_scales == (4,)
