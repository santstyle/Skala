"""Offline tests for the ``skala setup`` engine bootstrap.

Every network seam is faked: :func:`skala.setup.setup_engine` receives injected
``releases_loader`` / ``downloader`` callables, and the lower-level fetch helpers
run against a patched ``urllib.request.urlopen``. No test touches the GitHub API.
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
import skala.setup as setup
from skala import engine
from skala.utils import DependencyError


def _binary_name() -> str:
    return engine.ENGINE_BINARY + (".exe" if os.name == "nt" else "")


def _engine_asset_name(token: str | None = None) -> str:
    os_token = setup.asset_os_token() if token is None else token
    return f"{engine.ENGINE_BINARY}-20220424-{os_token}.zip"


def _asset(name: str, *, size: int | None = None, with_url: bool = True) -> dict[str, object]:
    asset: dict[str, object] = {"name": name}
    if with_url:
        asset["browser_download_url"] = f"https://example.invalid/{name}"
    if size is not None:
        asset["size"] = size
    return asset


def _release(tag: str, assets: object, *, draft: bool = False) -> dict[str, object]:
    return {"tag_name": tag, "draft": draft, "assets": assets}


def _engine_zip_bytes(binary: str, *, with_models: bool = True, wrapper: str = "wrapper") -> bytes:
    stream = io.BytesIO()
    prefix = f"{wrapper}/" if wrapper else ""
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr(f"{prefix}{binary}", b"binary-bytes")
        bundle.writestr(f"{prefix}vcomp140.dll", b"dll-bytes")
        bundle.writestr(f"{prefix}README.md", b"demo-doc")
        if with_models:
            bundle.writestr(f"{prefix}models/realesrgan-x4plus.param", b"param-bytes")
            bundle.writestr(f"{prefix}models/realesrgan-x4plus.bin", b"weight-bytes")
    return stream.getvalue()


def _forbid_loader() -> Sequence[Mapping[str, object]]:
    raise AssertionError("setup_engine must not reach the network")


def _forbid_downloader(url: str, destination: Path) -> Path:
    raise AssertionError("setup_engine must not download anything")


def _downloader(payload: bytes) -> Callable[[str, Path], Path]:
    def _fetch(url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return destination

    return _fetch


class _FakeResponse:
    """Stand-in for an ``urlopen`` response: supports ``with`` and chunked ``read``."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload)
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


# ---------------------------------------------------------------------------------
# asset_os_token
# ---------------------------------------------------------------------------------
def test_asset_os_token_maps_supported_platforms() -> None:
    assert setup.asset_os_token("win32") == "windows"
    assert setup.asset_os_token("linux") == "ubuntu"
    assert setup.asset_os_token("darwin") == "macos"


@pytest.mark.skipif(
    sys.platform not in setup.OS_ASSET_TOKENS, reason="platform has no published engine"
)
def test_asset_os_token_defaults_to_current_platform() -> None:
    assert setup.asset_os_token() == setup.OS_ASSET_TOKENS[sys.platform]


def test_asset_os_token_rejects_unknown_platform() -> None:
    with pytest.raises(DependencyError, match="no prebuilt engine is published"):
        setup.asset_os_token("plan9")


# ---------------------------------------------------------------------------------
# find_release_asset
# ---------------------------------------------------------------------------------
def test_find_release_asset_returns_newest_matching_release() -> None:
    wanted = _engine_asset_name("windows")
    newest = _release("v0.3.1", [_asset(wanted)])
    older = _release("v0.2.5.0", [_asset(wanted)])
    release, asset = setup.find_release_asset([newest, older], "windows")
    assert release["tag_name"] == "v0.3.1"
    assert asset["name"] == wanted


def test_find_release_asset_skips_drafts_other_platforms_and_malformed() -> None:
    wanted = _engine_asset_name("windows")
    releases = [
        _release("v0.3.2", [_asset(wanted)], draft=True),
        _release("v0.3.1", []),
        _release("v0.3.0", [_asset(_engine_asset_name("macos"))]),
        _release("v0.2.6", ["not-a-dict"]),
        _release("v0.2.6", "not-a-list"),
        _release("v0.2.5.0", [_asset(wanted)]),
    ]
    release, asset = setup.find_release_asset(releases, "windows")
    assert release["tag_name"] == "v0.2.5.0"
    assert asset["name"] == wanted


def test_find_release_asset_version_pin_selects_that_release() -> None:
    wanted = _engine_asset_name("windows")
    releases = [
        _release("v0.3.1", [_asset(wanted)]),
        _release("v0.2.5.0", [_asset(wanted)]),
    ]
    release, picked = setup.find_release_asset(releases, "windows", version="v0.2.5.0")
    assert release["tag_name"] == "v0.2.5.0"
    assert picked["name"] == wanted


def test_find_release_asset_version_pin_without_build_raises() -> None:
    releases = [_release("v0.3.1", [_asset(_engine_asset_name("windows"))])]
    with pytest.raises(DependencyError, match=r"release v9\.9\.9 has no"):
        setup.find_release_asset(releases, "windows", version="v9.9.9")


def test_find_release_asset_without_any_match_raises() -> None:
    with pytest.raises(DependencyError, match=r"no published .* build for windows"):
        setup.find_release_asset([_release("v0.3.0", [])], "windows")


# ---------------------------------------------------------------------------------
# extract_release
# ---------------------------------------------------------------------------------
def test_extract_release_unpacks_binary_models_and_dll(tmp_path: Path) -> None:
    archive = tmp_path / "engine.zip"
    archive.write_bytes(_engine_zip_bytes(_binary_name()))
    destination = tmp_path / "engine"
    binary, models = setup.extract_release(archive, destination)
    assert binary == destination / _binary_name()
    assert binary.is_file()
    assert models is not None
    assert models == destination / "models"
    assert (models / "realesrgan-x4plus.bin").is_file()
    assert (destination / "vcomp140.dll").is_file()
    assert not (destination / "README.md").exists()


def test_extract_release_accepts_flat_archive_without_models(tmp_path: Path) -> None:
    archive = tmp_path / "engine.zip"
    archive.write_bytes(_engine_zip_bytes(_binary_name(), with_models=False, wrapper=""))
    binary, models = setup.extract_release(archive, tmp_path / "engine")
    assert binary == tmp_path / "engine" / _binary_name()
    assert models is None


def test_extract_release_rejects_archive_without_binary(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("wrapper/README.md", b"demo-doc")
    archive = tmp_path / "engine.zip"
    archive.write_bytes(stream.getvalue())
    with pytest.raises(DependencyError, match="contains no"):
        setup.extract_release(archive, tmp_path / "engine")


def test_extract_release_skips_zip_slip_members(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr(f"../{_binary_name()}", b"evil-bytes")
        bundle.writestr(_binary_name(), b"good-bytes")
    archive = tmp_path / "engine.zip"
    archive.write_bytes(stream.getvalue())
    binary, _ = setup.extract_release(archive, tmp_path / "engine")
    assert binary.read_bytes() == b"good-bytes"
    assert not (tmp_path / _binary_name()).exists()


# ---------------------------------------------------------------------------------
# setup_engine
# ---------------------------------------------------------------------------------
def test_setup_engine_defaults_home_from_skala_home() -> None:
    result = setup.setup_engine(dry_run=True, releases_loader=_forbid_loader)
    assert result.home == setup.skala_home()
    assert not result.home.exists()


def test_setup_engine_dry_run_plans_without_writing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    result = setup.setup_engine(
        home=home,
        dry_run=True,
        releases_loader=_forbid_loader,
        downloader=_forbid_downloader,
    )
    assert result.action == "plan"
    assert result.home == home
    assert result.engine_dir == home / "engine"
    assert result.engine_binary == home / "engine" / _binary_name()
    assert result.models_dir is None
    assert result.version is None
    assert not home.exists()


def test_setup_engine_dry_run_reports_existing_install(tmp_path: Path) -> None:
    home = tmp_path / "home"
    engine_dir = home / "engine"
    engine_dir.mkdir(parents=True)
    binary = engine_dir / _binary_name()
    binary.write_bytes(b"")
    (engine_dir / "models").mkdir()
    result = setup.setup_engine(
        home=home, dry_run=True, releases_loader=_forbid_loader, downloader=_forbid_downloader
    )
    assert result.action == "plan"
    assert result.engine_binary == binary
    assert result.models_dir == engine_dir / "models"


def test_setup_engine_reuses_existing_install_without_network(tmp_path: Path) -> None:
    home = tmp_path / "home"
    engine_dir = home / "engine"
    engine_dir.mkdir(parents=True)
    binary = engine_dir / _binary_name()
    binary.write_bytes(b"")
    result = setup.setup_engine(
        home=home, releases_loader=_forbid_loader, downloader=_forbid_downloader
    )
    assert result.action == "reuse"
    assert result.engine_binary == binary


def test_setup_engine_fresh_install_extracts_engine(tmp_path: Path) -> None:
    home = tmp_path / "home"
    payload = _engine_zip_bytes(_binary_name())
    release = _release("v0.2.5.0", [_asset(_engine_asset_name(), size=len(payload))])
    result = setup.setup_engine(
        home=home,
        releases_loader=lambda: [release],
        downloader=_downloader(payload),
    )
    assert result.action == "install"
    assert result.version == "v0.2.5.0"
    assert result.asset_name == _engine_asset_name()
    assert result.engine_binary.is_file()
    assert result.models_dir is not None
    assert result.models_dir.is_dir()


def test_setup_engine_force_replaces_existing_install(tmp_path: Path) -> None:
    home = tmp_path / "home"
    engine_dir = home / "engine"
    engine_dir.mkdir(parents=True)
    (engine_dir / _binary_name()).write_bytes(b"stale")
    payload = _engine_zip_bytes(_binary_name())
    release = _release("v0.2.5.0", [_asset(_engine_asset_name(), size=len(payload))])
    result = setup.setup_engine(
        home=home,
        force=True,
        releases_loader=lambda: [release],
        downloader=_downloader(payload),
    )
    assert result.action == "upgrade"
    assert result.engine_binary.read_bytes() == b"binary-bytes"
    assert result.models_dir is not None


def test_setup_engine_rejects_truncated_download(tmp_path: Path) -> None:
    payload = _engine_zip_bytes(_binary_name())
    release = _release("v0.2.5.0", [_asset(_engine_asset_name(), size=len(payload) + 512)])
    with pytest.raises(DependencyError, match="wrong size"):
        setup.setup_engine(
            home=tmp_path / "home",
            releases_loader=lambda: [release],
            downloader=_downloader(payload),
        )


def test_setup_engine_requires_asset_download_url(tmp_path: Path) -> None:
    release = _release("v0.2.5.0", [_asset(_engine_asset_name(), with_url=False)])
    with pytest.raises(DependencyError, match="missing an asset name or download URL"):
        setup.setup_engine(
            home=tmp_path / "home",
            releases_loader=lambda: [release],
            downloader=_forbid_downloader,
        )


# ---------------------------------------------------------------------------------
# load_releases / download_asset (patched urlopen)
# ---------------------------------------------------------------------------------
def test_load_releases_parses_the_release_list(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        [{"tag_name": "v0.3.0", "assets": []}, "not-a-dict", {"tag_name": "v0.2.5.0"}]
    ).encode()
    monkeypatch.setattr(
        setup.urllib.request, "urlopen", lambda *args, **kwargs: _FakeResponse(payload)
    )
    releases = setup.load_releases()
    assert [release["tag_name"] for release in releases] == ["v0.3.0", "v0.2.5.0"]


def test_load_releases_rejects_non_list_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"message": "rate limited"}).encode()
    monkeypatch.setattr(
        setup.urllib.request, "urlopen", lambda *args, **kwargs: _FakeResponse(payload)
    )
    with pytest.raises(DependencyError, match="expected a list"):
        setup.load_releases()


def test_load_releases_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> _FakeResponse:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(setup.urllib.request, "urlopen", _boom)
    with pytest.raises(DependencyError, match="could not reach"):
        setup.load_releases()


def test_download_asset_streams_into_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"engine-payload-bytes"
    monkeypatch.setattr(
        setup.urllib.request, "urlopen", lambda *args, **kwargs: _FakeResponse(payload)
    )
    destination = tmp_path / "nested" / "engine.zip"
    written = setup.download_asset("https://example.invalid/engine.zip", destination)
    assert written == destination
    assert destination.read_bytes() == payload


def test_download_asset_wraps_network_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> _FakeResponse:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(setup.urllib.request, "urlopen", _boom)
    with pytest.raises(DependencyError, match="could not download"):
        setup.download_asset("https://example.invalid/engine.zip", tmp_path / "engine.zip")
