"""Smoke tests that drive the real engine and ffmpeg binaries.

These are the only tests that prove the end-to-end pipeline actually runs; the
unit suite mocks every subprocess. They are opt-in (see ``tests/conftest.py``)
and self-skip when the external tools are missing, so leaving them in the
default suite is safe. Run them with::

    SKALA_RUN_INTEGRATION=1 python -m pytest -m integration
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from skala.engine import discover_engine
from skala.image import upscale_image
from skala.utils import DependencyError, missing_ffmpeg
from skala.video import probe_video, upscale_video

pytestmark = pytest.mark.integration


def _require_engine() -> Path:
    try:
        return discover_engine()
    except DependencyError as exc:
        pytest.skip(f"engine unavailable: {exc}")


def _require_ffmpeg() -> None:
    missing = missing_ffmpeg()
    if missing:
        names = ", ".join(dep.name for dep in missing)
        pytest.skip(f"ffmpeg/ffprobe unavailable: {names}")


def _make_png(path: Path, size: tuple[int, int] = (16, 16)) -> Path:
    image = Image.new("RGB", size, color=(32, 64, 128))
    # A little structure so the engine has something to work on.
    for x in range(0, size[0], 4):
        for y in range(size[1]):
            image.putpixel((x, y), (240, 240, 240))
    image.save(path, format="PNG")
    return path


def test_upscale_image_end_to_end(tmp_path: Path) -> None:
    engine = _require_engine()
    source = _make_png(tmp_path / "tile.png")
    output = tmp_path / "tile_out.png"

    result = upscale_image(
        source, output_path=output, scale=2, engine_path=engine, work_base=tmp_path
    )

    assert result.output == output
    assert result.output.is_file()
    assert result.output_size == (32, 32)
    with Image.open(output) as produced:
        assert produced.size == (32, 32)


def test_upscale_video_end_to_end(tmp_path: Path) -> None:
    engine = _require_engine()
    _require_ffmpeg()
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None  # guaranteed by _require_ffmpeg

    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    assert source.is_file()

    info = probe_video(source)
    assert (info.width, info.height) == (64, 48)

    output = tmp_path / "clip_out.mp4"
    result = upscale_video(
        source,
        output_path=output,
        scale=2,
        model="realesr-animevideov3",
        audio="drop",
        engine_path=engine,
        work_base=tmp_path,
    )

    assert result.output == output
    assert result.output.is_file()
    assert result.output_size == (128, 96)
    upscaled = probe_video(output)
    assert (upscaled.width, upscaled.height) == (128, 96)
