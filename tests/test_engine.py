"""Tests for the engine wrapper: discovery, pure command building, and OOM retry."""

from __future__ import annotations

import itertools
import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from skala import engine
from skala.utils import DependencyError, EngineError, InputError

ENGINE = "C:/tools/realesrgan-ncnn-vulkan.exe"
X4PLUS = "realesrgan-x4plus"
VIDEO = "realesr-animevideov3"


def _completed(
    argv: Sequence[str],
    code: int,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(argv), code, stdout, stderr)


class _Recorder:
    """Fake ``engine._run`` that replays prepared results and records every argv."""

    def __init__(self, results: list[subprocess.CompletedProcess[str]]) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        return self.results[len(self.calls) - 1]


def _write_engine(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = engine.ENGINE_BINARY + (".exe" if os.name == "nt" else "")
    path = directory / name
    path.write_text("")
    return path


# ---------------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------------
def test_build_command_includes_every_option_in_documented_order() -> None:
    argv = engine.build_command(
        ENGINE,
        "in.png",
        "out.png",
        model=X4PLUS,
        scale=4,
        tile=200,
        model_dir="C:/models",
        gpu_id=0,
        threads=2,
        fmt="png",
        verbose=True,
    )
    assert argv == [
        ENGINE,
        "-i",
        "in.png",
        "-o",
        "out.png",
        "-n",
        X4PLUS,
        "-s",
        "4",
        "-t",
        "200",
        "-m",
        "C:/models",
        "-g",
        "0",
        "-j",
        "2:2:2",
        "-f",
        "png",
        "-v",
    ]


def test_build_command_omits_optional_flags() -> None:
    argv = engine.build_command(ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=0)
    assert argv == [ENGINE, "-i", "in.png", "-o", "out.png", "-n", X4PLUS, "-s", "4", "-t", "0"]
    for flag in ("-m", "-g", "-j", "-f", "-v"):
        assert flag not in argv


def test_build_command_never_emits_fp16_or_y_flags() -> None:
    argv = engine.build_command(ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=0)
    assert "-x" not in argv
    assert "-y" not in argv


def test_build_command_accepts_scale_three_for_video_model() -> None:
    argv = engine.build_command(ENGINE, "f.png", "g.png", model=VIDEO, scale=3, tile=0)
    assert argv[argv.index("-s") + 1] == "3"
    assert argv[argv.index("-n") + 1] == VIDEO


def test_build_command_accepts_pathlib_paths() -> None:
    argv = engine.build_command(
        Path("C:/tools/engine.exe"),
        Path("C:/in.png"),
        Path("C:/out.png"),
        model=X4PLUS,
        scale=4,
        tile=0,
    )
    assert argv[0] == str(Path("C:/tools/engine.exe"))
    assert argv[2] == str(Path("C:/in.png"))


def test_build_command_rejects_unknown_model() -> None:
    with pytest.raises(InputError, match="unknown model"):
        engine.build_command(ENGINE, "in.png", "out.png", model="nope", scale=4, tile=0)


def test_build_command_rejects_non_native_scale() -> None:
    with pytest.raises(InputError, match="does not support scale 2x"):
        engine.build_command(ENGINE, "in.png", "out.png", model=X4PLUS, scale=2, tile=0)


@pytest.mark.parametrize("tile", [16, 31, -1])
def test_build_command_rejects_tile_below_minimum(tile: int) -> None:
    with pytest.raises(InputError, match="tile size"):
        engine.build_command(ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=tile)


def test_build_command_rejects_negative_gpu_id() -> None:
    with pytest.raises(InputError, match="gpu id"):
        engine.build_command(ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=0, gpu_id=-1)


def test_build_command_rejects_zero_threads() -> None:
    with pytest.raises(InputError, match="thread count"):
        engine.build_command(ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=0, threads=0)


@pytest.mark.parametrize("fmt", ["bmp", "tiff", "PNG", ""])
def test_build_command_rejects_unsupported_format(fmt: str) -> None:
    with pytest.raises(InputError, match="format must be one of"):
        engine.build_command(ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=0, fmt=fmt)


def test_build_command_rejects_model_dir_without_models_in_path() -> None:
    with pytest.raises(InputError, match="must contain 'models'"):
        engine.build_command(
            ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=0, model_dir="C:/weights"
        )


@pytest.mark.parametrize("model_dir", ["C:/models", "C:/opt/my-models", "models2"])
def test_build_command_accepts_paths_containing_models(model_dir: str) -> None:
    argv = engine.build_command(
        ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=0, model_dir=model_dir
    )
    assert argv[argv.index("-m") + 1] == model_dir


# ---------------------------------------------------------------------------------
# MODEL_SPECS
# ---------------------------------------------------------------------------------
def test_model_specs_cover_the_documented_names() -> None:
    assert set(engine.MODEL_SPECS) == {
        "realesr-animevideov3",
        "realesrgan-x4plus",
        "realesrgan-x4plus-anime",
        "realesrnet-x4plus",
    }


def test_animevideov3_supports_two_three_and_four() -> None:
    assert engine.MODEL_SPECS[VIDEO].native_scales == (2, 3, 4)


@pytest.mark.parametrize(
    "name", ["realesrgan-x4plus", "realesrgan-x4plus-anime", "realesrnet-x4plus"]
)
def test_other_models_are_fixed_four_times(name: str) -> None:
    assert engine.MODEL_SPECS[name].native_scales == (4,)


def test_default_models_exist_in_specs() -> None:
    assert engine.DEFAULT_IMAGE_MODEL in engine.MODEL_SPECS
    assert engine.DEFAULT_VIDEO_MODEL in engine.MODEL_SPECS


def test_spec_name_matches_its_key() -> None:
    for key, spec in engine.MODEL_SPECS.items():
        assert spec.name == key


# ---------------------------------------------------------------------------------
# tile_retry_sequence
# ---------------------------------------------------------------------------------
def test_auto_tile_tries_auto_then_ladder() -> None:
    assert engine.tile_retry_sequence(0) == (0, 400, 200, 100, 64, 32)


def test_ladder_starts_at_requested_tile_when_below_top() -> None:
    assert engine.tile_retry_sequence(150) == (150, 100, 64, 32)


def test_requested_tile_above_ladder_is_tried_first() -> None:
    assert engine.tile_retry_sequence(800) == (800, 400, 200, 100, 64, 32)


def test_tile_at_ladder_top_descends_the_ladder() -> None:
    assert engine.tile_retry_sequence(400) == (400, 200, 100, 64, 32)


def test_minimum_tile_has_no_fallback() -> None:
    assert engine.tile_retry_sequence(engine.MIN_TILE) == (engine.MIN_TILE,)


@pytest.mark.parametrize("tile", [1, 16, 31, -10])
def test_illegal_tile_is_rejected(tile: int) -> None:
    with pytest.raises(InputError, match="tile size"):
        engine.tile_retry_sequence(tile)


@pytest.mark.parametrize("start", [32, 64, 100, 150, 400, 800])
def test_every_sequence_is_strictly_decreasing(start: int) -> None:
    sequence = engine.tile_retry_sequence(start)
    assert all(left > right for left, right in itertools.pairwise(sequence))


def test_auto_sequence_is_the_sentinel_then_a_decreasing_ladder() -> None:
    sequence = engine.tile_retry_sequence(0)
    assert sequence[0] == 0
    assert all(left > right for left, right in itertools.pairwise(sequence[1:]))


# ---------------------------------------------------------------------------------
# discover_engine
# ---------------------------------------------------------------------------------
def test_explicit_path_is_used(tmp_path: Path) -> None:
    binary = _write_engine(tmp_path)
    assert engine.discover_engine(binary) == binary.resolve()


def test_explicit_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(DependencyError, match="--engine-path"):
        engine.discover_engine(tmp_path / "missing-engine")


def test_environment_variable_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = _write_engine(tmp_path)
    monkeypatch.setenv(engine.ENV_ENGINE, str(binary))
    assert engine.discover_engine() == binary.resolve()


def test_environment_variable_pointing_nowhere_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(engine.ENV_ENGINE, str(tmp_path / "ghost"))
    with pytest.raises(DependencyError, match=engine.ENV_ENGINE):
        engine.discover_engine()


def test_explicit_path_beats_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = _write_engine(tmp_path / "real")
    monkeypatch.setenv(engine.ENV_ENGINE, str(tmp_path / "ghost"))
    assert engine.discover_engine(binary) == binary.resolve()


def test_local_bin_directory_is_searched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(engine.ENV_ENGINE, raising=False)
    monkeypatch.chdir(tmp_path)
    binary = _write_engine(tmp_path / "bin")
    assert engine.discover_engine() == binary.resolve()


def test_environment_beats_local_bin_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_binary = _write_engine(tmp_path / "envdir")
    _write_engine(tmp_path / "bin")
    monkeypatch.setenv(engine.ENV_ENGINE, str(env_binary))
    assert engine.discover_engine() == env_binary.resolve()


def test_path_lookup_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(engine.ENV_ENGINE, raising=False)
    monkeypatch.chdir(tmp_path)
    tools = tmp_path / "tools"
    binary = _write_engine(tools)
    monkeypatch.setenv("PATH", str(tools))
    assert engine.discover_engine() == binary.resolve()


def test_bare_command_name_is_resolved_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = tmp_path / "tools"
    binary = _write_engine(tools)
    monkeypatch.setenv("PATH", str(tools))
    assert engine.discover_engine(engine.ENGINE_BINARY) == binary.resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows executable suffix handling")
def test_windows_name_without_suffix_is_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    binary = _write_engine(tmp_path)
    assert engine.discover_engine(tmp_path / engine.ENGINE_BINARY) == binary.resolve()


def test_missing_engine_raises_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(engine.ENV_ENGINE, raising=False)
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(DependencyError) as excinfo:
        engine.discover_engine()
    message = str(excinfo.value)
    assert engine.ENGINE_BINARY in message
    assert "--engine-path" in message


# ---------------------------------------------------------------------------------
# managed_model_dir
# ---------------------------------------------------------------------------------
def test_managed_model_dir_returns_sibling_models(tmp_path: Path) -> None:
    binary = _write_engine(tmp_path)
    (tmp_path / "models").mkdir()
    found = engine.managed_model_dir(binary)
    assert found is not None
    assert found == tmp_path.resolve() / "models"


def test_managed_model_dir_returns_none_without_models(tmp_path: Path) -> None:
    binary = _write_engine(tmp_path)
    assert engine.managed_model_dir(binary) is None


def test_managed_model_dir_rejects_models_that_are_a_file(tmp_path: Path) -> None:
    binary = _write_engine(tmp_path)
    (tmp_path / "models").write_text("not a directory")
    assert engine.managed_model_dir(binary) is None


def test_managed_model_dir_does_not_require_the_binary_to_exist(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    found = engine.managed_model_dir(str(tmp_path / "missing-binary"))
    assert found is not None
    assert found.is_dir()


# ---------------------------------------------------------------------------------
# run_engine
# ---------------------------------------------------------------------------------
def _argv(tile: int = 0) -> list[str]:
    return engine.build_command(ENGINE, "in.png", "out.png", model=X4PLUS, scale=4, tile=tile)


def test_successful_run_returns_first_result(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder([_completed([], 0)])
    monkeypatch.setattr(engine, "_run", recorder)
    result = engine.run_engine(_argv(), tile_candidates=(0, 400))
    assert result.returncode == 0
    assert len(recorder.calls) == 1
    assert recorder.calls[0][recorder.calls[0].index("-t") + 1] == "0"


def test_oom_failure_retries_with_smaller_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder(
        [
            _completed([], 1, stderr="vkAllocateMemory failed"),
            _completed([], 0),
        ]
    )
    monkeypatch.setattr(engine, "_run", recorder)
    result = engine.run_engine(_argv(), tile_candidates=(0, 400))
    assert result.returncode == 0
    assert len(recorder.calls) == 2
    assert recorder.calls[1][recorder.calls[1].index("-t") + 1] == "400"


def test_oom_detected_in_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder(
        [
            _completed([], 1, stdout="out of device memory"),
            _completed([], 0),
        ]
    )
    monkeypatch.setattr(engine, "_run", recorder)
    assert engine.run_engine(_argv(), tile_candidates=(0, 400)).returncode == 0
    assert len(recorder.calls) == 2


def test_retry_preserves_the_remaining_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _argv(tile=400)
    recorder = _Recorder([_completed([], 1, stderr="bad_alloc"), _completed([], 0)])
    monkeypatch.setattr(engine, "_run", recorder)
    engine.run_engine(argv, tile_candidates=(400, 200))
    expected = list(argv)
    expected[expected.index("-t") + 1] = "200"
    assert recorder.calls[1] == expected


def test_non_memory_failure_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder([_completed([], 7, stderr="decode image in.png failed")])
    monkeypatch.setattr(engine, "_run", recorder)
    with pytest.raises(EngineError, match="exited with code 7") as excinfo:
        engine.run_engine(_argv(), tile_candidates=(0, 400))
    assert "decode image in.png failed" in str(excinfo.value)
    assert len(recorder.calls) == 1


def test_exhausting_every_tile_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder(
        [
            _completed([], 1, stderr="vkAllocateMemory failed"),
            _completed([], 1, stderr="vkAllocateMemory failed"),
        ]
    )
    monkeypatch.setattr(engine, "_run", recorder)
    with pytest.raises(EngineError, match="every tile size") as excinfo:
        engine.run_engine(_argv(), tile_candidates=(0, 400))
    message = str(excinfo.value)
    assert "0, 400" in message
    assert len(recorder.calls) == 2


def test_empty_tile_candidates_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_run", _Recorder([_completed([], 0)]))
    with pytest.raises(ValueError, match="must not be empty"):
        engine.run_engine(_argv(), tile_candidates=())


def test_command_without_tile_flag_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder([_completed([], 0)])
    monkeypatch.setattr(engine, "_run", recorder)
    assert engine.run_engine([ENGINE, "-i", "in.png", "-o", "out.png"]).returncode == 0
    assert recorder.calls[0] == [ENGINE, "-i", "in.png", "-o", "out.png"]


def test_retry_emits_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = _Recorder(
        [
            _completed([], 1, stderr="vkAllocateMemory failed"),
            _completed([], 0),
        ]
    )
    monkeypatch.setattr(engine, "_run", recorder)
    with caplog.at_level(logging.WARNING, logger="skala.engine"):
        engine.run_engine(_argv(), tile_candidates=(0, 400))
    assert "out of GPU memory" in caplog.text


def test_custom_logger_receives_the_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = _Recorder([_completed([], 1, stderr="bad_alloc"), _completed([], 0)])
    monkeypatch.setattr(engine, "_run", recorder)
    custom = logging.getLogger("skala.tests.custom")
    with caplog.at_level(logging.WARNING, logger="skala.tests.custom"):
        engine.run_engine(_argv(), tile_candidates=(0, 400), log=custom)
    assert "retries left: 1" in caplog.text


def test_failure_message_truncates_huge_output(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder([_completed([], 3, stderr="x" * 5000)])
    monkeypatch.setattr(engine, "_run", recorder)
    with pytest.raises(EngineError) as excinfo:
        engine.run_engine(_argv(), tile_candidates=(0,))
    assert str(excinfo.value).endswith("x" * 100)
    assert len(str(excinfo.value)) < 1200


def test_model_spec_returns_the_registered_spec() -> None:
    spec = engine.model_spec(VIDEO)
    assert spec is engine.MODEL_SPECS[VIDEO]
    assert spec.native_scales == (2, 3, 4)


def test_model_spec_rejects_unknown_model() -> None:
    with pytest.raises(InputError) as excinfo:
        engine.model_spec("does-not-exist")
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert VIDEO in message
