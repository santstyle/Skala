"""Tests for skala.config: discovery, parsing, validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from skala.config import AUDIO_CHOICES, CONFIG_FILENAME, SkalaConfig, find_config_file, load_config
from skala.utils import InputError


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestDiscovery:
    def test_returns_defaults_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert load_config() == SkalaConfig()

    def test_find_config_file_none_when_absent(self, tmp_path: Path) -> None:
        assert find_config_file(tmp_path) is None

    def test_find_config_file_returns_path(self, tmp_path: Path) -> None:
        expected = _write(tmp_path / CONFIG_FILENAME, "tile = 256\n")
        assert find_config_file(tmp_path) == expected

    def test_auto_discovers_in_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write(tmp_path / CONFIG_FILENAME, "tile = 256\n")
        monkeypatch.chdir(tmp_path)
        assert load_config().tile == 256

    def test_explicit_path_wins_over_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / CONFIG_FILENAME, "tile = 256\n")
        other = _write(tmp_path / "custom.toml", "tile = 128\n")
        monkeypatch.chdir(tmp_path)
        assert load_config(other).tile == 128

    def test_explicit_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InputError, match="not found"):
            load_config(tmp_path / "ghost.toml")

    def test_directory_as_explicit_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InputError, match="not found"):
            load_config(tmp_path)


class TestParsing:
    def test_empty_file_is_all_defaults(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "")
        assert load_config(path) == SkalaConfig()

    def test_partial_config_leaves_others_none(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "crf = 20\n")
        cfg = load_config(path)
        assert cfg.crf == 20
        assert cfg.tile is None
        assert cfg.model is None

    def test_full_config_round_trips(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "c.toml",
            "\n".join(
                [
                    "model = 'realesrgan-x4plus-anime'",
                    "engine_path = 'C:/tools/realesrgan-ncnn-vulkan.exe'",
                    "ffmpeg_path = 'ffmpeg'",
                    "ffprobe_path = 'ffprobe'",
                    "tile = 256",
                    "gpu_id = 0",
                    "threads = 2",
                    "workdir = 'D:/tmp/skala'",
                    "keep_temp = true",
                    "overwrite = false",
                    "verbose = true",
                    "crf = 18",
                    "video_codec = 'h264'",
                    "audio = 'copy'",
                ]
            ),
        )
        assert load_config(path) == SkalaConfig(
            model="realesrgan-x4plus-anime",
            engine_path="C:/tools/realesrgan-ncnn-vulkan.exe",
            ffmpeg_path="ffmpeg",
            ffprobe_path="ffprobe",
            tile=256,
            gpu_id=0,
            threads=2,
            workdir="D:/tmp/skala",
            keep_temp=True,
            overwrite=False,
            verbose=True,
            crf=18,
            video_codec="h264",
            audio="copy",
        )

    def test_invalid_toml_raises_with_path(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "tile = = 4\n")
        with pytest.raises(InputError, match="Invalid TOML") as excinfo:
            load_config(path)
        assert "c.toml" in str(excinfo.value)

    def test_section_table_is_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "[video]\ncrf = 18\n")
        with pytest.raises(InputError, match="Unknown key"):
            load_config(path)


class TestValidation:
    def test_unknown_key_lists_valid_keys(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "modle = 'x'\n")
        with pytest.raises(InputError, match="Unknown key") as excinfo:
            load_config(path)
        message = str(excinfo.value)
        assert "modle" in message
        assert "model" in message

    def test_bool_is_not_accepted_as_int(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "tile = true\n")
        with pytest.raises(InputError, match="must be an integer"):
            load_config(path)

    def test_float_is_not_accepted_as_int(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "tile = 4.5\n")
        with pytest.raises(InputError, match="must be an integer"):
            load_config(path)

    def test_int_is_not_accepted_as_bool(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "keep_temp = 1\n")
        with pytest.raises(InputError, match="must be a boolean"):
            load_config(path)

    def test_string_is_not_accepted_as_int(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "tile = 'big'\n")
        with pytest.raises(InputError, match="must be an integer"):
            load_config(path)

    def test_int_is_not_accepted_as_string(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "model = 4\n")
        with pytest.raises(InputError, match="must be a non-empty string"):
            load_config(path)

    def test_empty_string_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "model = ''\n")
        with pytest.raises(InputError, match="non-empty string"):
            load_config(path)

    @pytest.mark.parametrize(
        ("body", "needle"),
        [
            ("tile = -1\n", "'tile' must be"),
            ("gpu_id = -1\n", "'gpu_id' must be"),
            ("threads = 0\n", "'threads' must be"),
            ("crf = 52\n", "'crf' must be"),
            ("crf = -1\n", "'crf' must be"),
            ("audio = 'loudness'\n", "audio"),
        ],
    )
    def test_range_violations(self, tmp_path: Path, body: str, needle: str) -> None:
        path = _write(tmp_path / "c.toml", body)
        with pytest.raises(InputError, match=needle):
            load_config(path)

    def test_audio_choice_message_lists_options(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "audio = 'loudness'\n")
        with pytest.raises(InputError) as excinfo:
            load_config(path)
        message = str(excinfo.value)
        for choice in AUDIO_CHOICES:
            assert choice in message

    def test_boundary_values_accepted(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "tile = 0\ncrf = 0\nthreads = 1\ngpu_id = 0\n")
        cfg = load_config(path)
        assert (cfg.tile, cfg.crf, cfg.threads, cfg.gpu_id) == (0, 0, 1, 0)
