"""Unit tests for skala.utils: formatting, disk checks, work dirs, validation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
import skala.utils as utils
from skala.utils import (
    DependencyError,
    DiskSpaceError,
    InputError,
    MissingDependency,
    OutputExistsError,
    default_output_path,
    detect_hardware,
    ensure_output_path,
    format_bytes,
    format_duration,
    free_space,
    get_logger,
    missing_ffmpeg,
    require_free_space,
    setup_logging,
    validate_input_path,
    work_dir,
)


class TestFormatting:
    def test_format_bytes_small(self) -> None:
        assert format_bytes(0) == "0 B"
        assert format_bytes(512) == "512 B"

    def test_format_bytes_units(self) -> None:
        assert format_bytes(1024) == "1.0 KiB"
        assert format_bytes(1536) == "1.5 KiB"
        assert format_bytes(1024**2) == "1.0 MiB"
        assert format_bytes(3 * 1024**3) == "3.0 GiB"

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0:00:00"),
            (59, "0:00:59"),
            (61, "0:01:01"),
            (3661, "1:01:01"),
            (86399, "23:59:59"),
        ],
    )
    def test_format_duration(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    def test_format_duration_negative_or_nan(self) -> None:
        assert format_duration(-5) == "--:--:--"
        assert format_duration(float("nan")) == "--:--:--"


class TestDiskSpace:
    def test_free_space_returns_positive_int(self, tmp_path: Path) -> None:
        assert free_space(tmp_path) > 0

    def test_free_space_works_for_missing_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "does" / "not" / "exist"
        assert free_space(missing) > 0

    def test_require_free_space_passes_when_plenty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        usage = shutil._ntuple_diskusage(total=100, used=10, free=90)
        monkeypatch.setattr(utils.shutil, "disk_usage", lambda _p: usage)
        require_free_space(tmp_path, 50, what="test operation")

    def test_require_free_space_raises_when_short(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        usage = shutil._ntuple_diskusage(total=100, used=95, free=5)
        monkeypatch.setattr(utils.shutil, "disk_usage", lambda _p: usage)
        with pytest.raises(DiskSpaceError) as excinfo:
            require_free_space(tmp_path, 1024**3, what="video frames")
        message = str(excinfo.value)
        assert "video frames" in message
        assert "1.0 GiB" in message  # required amount shown
        assert "Insufficient disk space" in message


class TestWorkDir:
    def test_same_input_and_options_same_dir(self, tmp_path: Path) -> None:
        src = tmp_path / "input.png"
        src.write_bytes(b"x")
        a = work_dir(src, 4, "realesrgan-x4plus", base=tmp_path / "w")
        b = work_dir(src, 4, "realesrgan-x4plus", base=tmp_path / "w")
        assert a == b
        assert a.name.startswith("skala_work_")
        assert a.parent == tmp_path / "w"

    def test_different_options_different_dir(self, tmp_path: Path) -> None:
        src = tmp_path / "input.png"
        src.write_bytes(b"x")
        a = work_dir(src, 4, base=tmp_path)
        b = work_dir(src, 2, base=tmp_path)
        assert a != b

    def test_different_files_different_dir(self, tmp_path: Path) -> None:
        one = tmp_path / "one.png"
        two = tmp_path / "two.png"
        one.write_bytes(b"1")
        two.write_bytes(b"2")
        assert work_dir(one, 4, base=tmp_path) != work_dir(two, 4, base=tmp_path)

    def test_content_change_changes_dir(self, tmp_path: Path) -> None:
        src = tmp_path / "input.png"
        src.write_bytes(b"before")
        before = work_dir(src, 4, base=tmp_path)
        src.write_bytes(b"after!")  # size/mtime change
        after = work_dir(src, 4, base=tmp_path)
        assert before != after

    def test_default_work_base_honours_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SKALA_WORKDIR", str(tmp_path))
        assert utils.default_work_base() == tmp_path / "skala"
        monkeypatch.delenv("SKALA_WORKDIR")
        assert utils.default_work_base().name == "skala"


class TestOutputHandling:
    def test_default_output_path(self, tmp_path: Path) -> None:
        src = tmp_path / "photo.jpeg"
        assert default_output_path(src, 4, ".png") == tmp_path / "photo_skala_4x.png"
        assert default_output_path(src, 2, "webp") == tmp_path / "photo_skala_2x.webp"

    def test_existing_output_rejected_without_overwrite(self, tmp_path: Path) -> None:
        out = tmp_path / "out.png"
        out.write_bytes(b"old")
        with pytest.raises(OutputExistsError, match="--overwrite"):
            ensure_output_path(out, overwrite=False)

    def test_existing_output_allowed_with_overwrite(self, tmp_path: Path) -> None:
        out = tmp_path / "out.png"
        out.write_bytes(b"old")
        ensure_output_path(out, overwrite=True)

    def test_creates_missing_parents(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "out.png"
        ensure_output_path(out, overwrite=False)
        assert out.parent.is_dir()


class TestInputValidation:
    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InputError, match="not found"):
            validate_input_path(tmp_path / "ghost.png")

    def test_file_ok(self, tmp_path: Path) -> None:
        f = tmp_path / "a.png"
        f.write_bytes(b"data")
        assert validate_input_path(f) == f

    def test_dir_ok(self, tmp_path: Path) -> None:
        assert validate_input_path(tmp_path) == tmp_path

    def test_special_file_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = tmp_path / "weird"
        f.write_bytes(b"x")
        monkeypatch.setattr(Path, "is_file", lambda _self: False)
        monkeypatch.setattr(Path, "is_dir", lambda _self: False)
        with pytest.raises(InputError, match="neither a file nor a directory"):
            validate_input_path(f)


class TestDependencies:
    def test_missing_ffmpeg_lists_both_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils.shutil, "which", lambda _cmd: None)
        missing = missing_ffmpeg()
        assert [m.name for m in missing] == ["ffmpeg", "ffprobe"]
        assert all(m.hint() for m in missing)

    def test_missing_ffmpeg_empty_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils.shutil, "which", lambda cmd: Path(f"/usr/bin/{cmd}"))
        assert missing_ffmpeg() == []

    def test_hint_differs_by_platform(self) -> None:
        dep = MissingDependency(
            name="ffmpeg", hint_windows="winget install X", hint_linux="apt install x"
        )
        original = utils.sys.platform
        try:
            utils.sys.platform = "win32"
            assert dep.hint() == "winget install X"
            utils.sys.platform = "linux"
            assert dep.hint() == "apt install x"
        finally:
            utils.sys.platform = original

    def test_engine_hint_mentions_discovery_options(self) -> None:
        hints = utils._missing_engine()
        for needle in ("SKALA_ENGINE", "--engine-path", "bin"):
            assert needle in hints.hint_windows
            assert needle in hints.hint_linux


class TestHardwareDetection:
    def test_vulkaninfo_result_is_preferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils, "_vulkan_from_vulkaninfo", lambda: (True, "GPU: TestGPU"))
        monkeypatch.setattr(utils, "_vulkan_windows", lambda: (False, "nope"))
        status = detect_hardware()
        assert status.vulkan_available is True
        assert status.vulkan_detail == "GPU: TestGPU"
        assert status.cpu_count >= 1

    def test_detection_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> tuple[bool | None, str]:
            raise OSError("probe exploded")

        monkeypatch.setattr(utils, "_vulkan_from_vulkaninfo", boom)
        monkeypatch.setattr(utils, "_vulkan_linux", boom)
        monkeypatch.setattr(utils, "_vulkan_windows", boom)
        monkeypatch.setattr(utils.sys, "platform", "linux")
        status = detect_hardware()  # must not raise
        assert status.vulkan_available is None
        assert "probe exploded" in status.vulkan_detail

    def test_cpu_count_wrapper(self) -> None:
        assert utils.cpu_count() >= 1


class TestLogging:
    def test_verbose_sets_debug_level(self) -> None:
        setup_logging(verbose=True)
        assert logging.getLogger("skala").level == logging.DEBUG
        setup_logging(verbose=False)
        assert logging.getLogger("skala").level == logging.INFO

    def test_child_logger_names(self) -> None:
        setup_logging(verbose=False)
        logger = get_logger("video")
        assert logger.name == "skala.video"
        assert logger.parent is not None
        assert logger.parent.name == "skala"


def test_dependency_error_is_skala_error() -> None:
    with pytest.raises(DependencyError):
        raise DependencyError("missing")
    with pytest.raises(utils.SkalaError):
        raise DependencyError("missing")
