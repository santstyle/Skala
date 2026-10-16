"""Bootstrap the ``realesrgan-ncnn-vulkan`` engine into a per-user location.

``skala setup`` downloads the official Real-ESRGAN release zip for the current
platform and unpacks the engine binary and its bundled ``models/`` folder into
:func:`skala.utils.skala_home` (``<home>/engine``). It uses only the standard
library (:mod:`urllib.request`, :mod:`zipfile`) so it works before any optional
extra is installed.

The upstream project has no single canonical "engine" asset URL: the ``latest``
release is v0.3.0 and ships no binaries at all, while older releases carry the
``realesrgan-ncnn-vulkan-<date>-<os>.zip`` bundles. :func:`find_release_asset`
therefore scans the release list newest-first and picks the first one that
actually has a matching asset for this platform.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from logging import Logger
from pathlib import Path, PurePosixPath

from . import __version__
from .engine import ENGINE_BINARY, managed_model_dir
from .utils import DependencyError, get_logger, skala_home

__all__ = [
    "OS_ASSET_TOKENS",
    "RELEASES_API",
    "SetupResult",
    "asset_os_token",
    "download_asset",
    "extract_release",
    "find_release_asset",
    "load_releases",
    "setup_engine",
]

#: GitHub API endpoint that lists every Real-ESRGAN release.
RELEASES_API = "https://api.github.com/repos/xinntao/Real-ESRGAN/releases"

#: Maps :data:`sys.platform` to the OS token used in the release asset names.
OS_ASSET_TOKENS: dict[str, str] = {
    "win32": "windows",
    "linux": "ubuntu",
    "darwin": "macos",
}

_USER_AGENT = f"skala/{__version__}"


@dataclass(frozen=True)
class SetupResult:
    """Outcome of a :func:`setup_engine` call.

    ``action`` is ``"install"`` on a fresh install, ``"upgrade"`` when an
    existing engine was replaced, ``"reuse"`` when a usable install was found,
    and ``"plan"`` when ``dry_run`` was requested.
    """

    home: Path
    engine_dir: Path
    engine_binary: Path
    models_dir: Path | None
    version: str | None
    asset_name: str | None
    action: str


def asset_os_token(platform_name: str | None = None) -> str:
    """Return the release OS token (``windows``, ``ubuntu`` or ``macos``).

    Raises:
        DependencyError: when the platform has no published engine build.
    """
    key = sys.platform if platform_name is None else platform_name
    try:
        return OS_ASSET_TOKENS[key]
    except KeyError:
        supported = ", ".join(sorted(OS_ASSET_TOKENS))
        raise DependencyError(
            f"no prebuilt engine is published for {key!r}; supported platforms: "
            f"{supported}. Install {ENGINE_BINARY} manually and pass --engine-path."
        ) from None


def load_releases(timeout: float = 30.0) -> list[dict[str, object]]:
    """Fetch and parse the Real-ESRGAN release list.

    Raises:
        DependencyError: on any network or decoding failure.
    """
    request = urllib.request.Request(
        RELEASES_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise DependencyError(f"could not reach {RELEASES_API}: {exc}") from exc
    if not isinstance(payload, list):
        raise DependencyError(
            f"unexpected response from {RELEASES_API}: expected a list of releases"
        )
    return [item for item in payload if isinstance(item, dict)]


def download_asset(
    url: str,
    destination: Path,
    *,
    timeout: float = 300.0,
    log: Logger | None = None,
) -> Path:
    """Stream ``url`` into ``destination`` and return the written path.

    Raises:
        DependencyError: on any network or filesystem failure.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            destination.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError) as exc:
        raise DependencyError(f"could not download {url}: {exc}") from exc
    if log is not None:
        log.debug("downloaded %s to %s", url, destination)
    return destination


def find_release_asset(
    releases: Sequence[Mapping[str, object]],
    os_token: str,
    *,
    version: str | None = None,
    engine_binary: str = ENGINE_BINARY,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Return the ``(release, asset)`` pair for the newest matching engine zip.

    Releases are scanned in the order given (GitHub returns newest first) and
    drafts are skipped. ``version`` pins a specific release tag.

    Raises:
        DependencyError: when no matching release or asset exists.
    """
    prefix = f"{engine_binary}-"
    suffix = f"-{os_token}.zip"
    for release in releases:
        if release.get("draft"):
            continue
        if version is not None and release.get("tag_name") != version:
            continue
        assets = release.get("assets")
        if not isinstance(assets, list):
            continue
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = asset.get("name")
            if isinstance(name, str) and name.startswith(prefix) and name.endswith(suffix):
                return release, asset
    if version is not None:
        raise DependencyError(
            f"release {version} has no {engine_binary} build for {os_token}; "
            f"omit --version to use the newest available release"
        )
    raise DependencyError(
        f"no published {engine_binary} build for {os_token} was found in the Real-ESRGAN releases"
    )


def extract_release(
    archive: Path,
    destination: Path,
    *,
    engine_binary: str = ENGINE_BINARY,
    log: Logger | None = None,
) -> tuple[Path, Path | None]:
    """Unpack the engine binary and its models from ``archive``.

    Only the binary, its DLLs and the ``models/`` tree are extracted; demo
    media and docs are skipped. Member paths are sanitised against zip-slip and
    a single shared wrapper directory is stripped when present.

    Returns ``(engine_path, models_dir_or_None)``. Raises:
        DependencyError: when the archive carries no engine binary.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for info in bundle.infolist():
            if info.is_dir():
                continue
            safe = _safe_member(info.filename)
            if safe is None or not _is_wanted(safe, engine_binary):
                continue
            members.append((info, safe))
        if not members:
            raise DependencyError(f"{archive.name} contains no {engine_binary} build")
        roots = _strip_common_root([safe for _, safe in members])
        for (info, _), relative in zip(members, roots, strict=True):
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)

    binary = _find_binary(destination, engine_binary)
    if binary is None:
        raise DependencyError(f"{archive.name} did not contain a {engine_binary} binary")
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | 0o111)
    if log is not None:
        log.debug("extracted %s into %s", archive.name, destination)
    return binary, managed_model_dir(binary)


def setup_engine(
    *,
    home: str | os.PathLike[str] | None = None,
    version: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    releases_loader: Callable[[], Sequence[Mapping[str, object]]] | None = None,
    downloader: Callable[[str, Path], Path] | None = None,
    log: Logger | None = None,
) -> SetupResult:
    """Install the engine into ``home`` (default: :func:`skala_home`).

    With ``dry_run`` this performs no network access and writes nothing; it only
    describes what would happen. ``releases_loader`` and ``downloader`` are
    injection seams so the flow can be tested without a network.
    """
    logger = log if log is not None else get_logger("setup")
    base = Path(home) if home is not None else skala_home()
    engine_dir = base / "engine"
    existing = _find_binary(engine_dir, ENGINE_BINARY) if engine_dir.is_dir() else None

    if dry_run:
        binary = existing if existing is not None else engine_dir / _binary_name(ENGINE_BINARY)
        return SetupResult(
            home=base,
            engine_dir=engine_dir,
            engine_binary=binary,
            models_dir=managed_model_dir(binary),
            version=None,
            asset_name=None,
            action="plan",
        )

    if existing is not None and not force:
        _warn_if_no_models(existing, logger)
        return SetupResult(
            home=base,
            engine_dir=engine_dir,
            engine_binary=existing,
            models_dir=managed_model_dir(existing),
            version=None,
            asset_name=None,
            action="reuse",
        )

    loader = releases_loader if releases_loader is not None else load_releases
    fetch = downloader if downloader is not None else download_asset

    release, asset = find_release_asset(
        loader(), asset_os_token(), version=version, engine_binary=ENGINE_BINARY
    )
    tag = release.get("tag_name")
    asset_name = asset.get("name")
    url = asset.get("browser_download_url")
    if not isinstance(asset_name, str) or not isinstance(url, str):
        raise DependencyError(f"release {tag!r} is missing an asset name or download URL")

    with tempfile.TemporaryDirectory(prefix="skala_setup_") as scratch:
        archive = fetch(url, Path(scratch) / asset_name)
        expected = asset.get("size")
        actual = archive.stat().st_size
        if isinstance(expected, int) and actual != expected:
            raise DependencyError(
                f"downloaded {asset_name} has the wrong size ({actual} != {expected} "
                f"bytes); try again"
            )
        if engine_dir.exists():
            shutil.rmtree(engine_dir, ignore_errors=True)
        binary, models = extract_release(archive, engine_dir, log=logger)

    action = "upgrade" if existing is not None else "install"
    logger.debug("%s %s into %s", action, asset_name, engine_dir)
    return SetupResult(
        home=base,
        engine_dir=engine_dir,
        engine_binary=binary,
        models_dir=models,
        version=tag if isinstance(tag, str) else None,
        asset_name=asset_name,
        action=action,
    )


def _binary_name(engine_binary: str) -> str:
    """Platform filename for the engine, adding ``.exe`` on Windows."""
    return f"{engine_binary}.exe" if os.name == "nt" else engine_binary


def _find_binary(directory: Path, engine_binary: str) -> Path | None:
    """Return the engine binary under ``directory``, or ``None``."""
    wanted = _binary_name(engine_binary).lower()
    for candidate in directory.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == wanted:
            return candidate
    return None


def _safe_member(name: str) -> PurePosixPath | None:
    """Return a zip-slip-safe relative path for ``name``, or ``None``.

    Rejects absolute paths, parent traversals and drive-qualified names.
    """
    parts = [part for part in name.replace("\\", "/").split("/") if part not in ("", ".")]
    if not parts or any(part == ".." or ":" in part for part in parts):
        return None
    return PurePosixPath(*parts)


def _is_wanted(member: PurePosixPath, engine_binary: str) -> bool:
    """Whether a zip member is part of the engine we install."""
    name = member.name.lower()
    if "models" in {part.lower() for part in member.parts}:
        return True
    if name.startswith(engine_binary.lower()):
        return True
    return name.endswith(".dll")


def _strip_common_root(members: list[PurePosixPath]) -> list[PurePosixPath]:
    """Drop a single wrapping directory shared by every member, if any."""
    if not members or any(len(path.parts) < 2 for path in members):
        return members
    root = members[0].parts[0]
    if all(path.parts[0] == root for path in members):
        return [PurePosixPath(*path.parts[1:]) for path in members]
    return members


def _warn_if_no_models(binary: Path, log: Logger) -> None:
    """Warn when an existing install has no ``models/`` folder beside it."""
    if managed_model_dir(binary) is None:
        log.warning(
            "no models/ folder found next to %s; pass --model-dir or rerun 'skala setup --force'",
            binary,
        )
