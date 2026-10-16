"""Shared utilities: errors, logging, disk space, work dirs, dependency and hardware checks.

This module must stay engine-agnostic: it knows about ffmpeg/ffprobe and generic
binaries only, never about realesrgan-specific flags.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ENV_HOME",
    "DependencyError",
    "DiskSpaceError",
    "EngineError",
    "HardwareStatus",
    "InputError",
    "MissingDependency",
    "OutputExistsError",
    "SkalaError",
    "cpu_count",
    "default_output_path",
    "detect_hardware",
    "ensure_output_path",
    "format_bytes",
    "format_duration",
    "free_space",
    "get_logger",
    "require_free_space",
    "setup_logging",
    "skala_home",
    "validate_input_path",
    "work_dir",
]

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

#: Environment variable overriding the root of Skala-managed downloads.
ENV_HOME = "SKALA_HOME"


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------
class SkalaError(Exception):
    """Base class for expected, user-facing Skala errors.

    The exception message is written for end users (actionable, no traceback noise).
    """


class DependencyError(SkalaError):
    """A required external dependency (ffmpeg, ffprobe, engine binary) is missing."""


class EngineError(SkalaError):
    """The upscaling engine process failed or could not be started."""


class InputError(SkalaError):
    """The input path is missing, unreadable, or of an unsupported type."""


class DiskSpaceError(SkalaError):
    """Not enough free disk space to complete the operation safely."""


class OutputExistsError(SkalaError):
    """The output path already exists and --overwrite was not given."""


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    """Configure the ``skala`` logger hierarchy.

    Args:
        verbose: When True the effective level is DEBUG, otherwise INFO.
    """
    logger = logging.getLogger("skala")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``skala`` hierarchy (e.g. ``skala.video``)."""
    return logging.getLogger(f"skala.{name}")


# --------------------------------------------------------------------------------------
# Human-readable formatting
# --------------------------------------------------------------------------------------
def format_bytes(num: float) -> str:
    """Format a byte count as a short human-readable string (e.g. ``1.5 GiB``)."""
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024.0 or unit == "PiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"  # pragma: no cover - unreachable, loop always returns


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``H:MM:SS`` (used for estimates and ETAs)."""
    if seconds < 0 or seconds != seconds:  # negative or NaN
        return "--:--:--"
    total = round(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


# --------------------------------------------------------------------------------------
# Disk space
# --------------------------------------------------------------------------------------
def _existing_ancestor(path: Path) -> Path:
    """Return ``path`` itself if it exists, else its nearest existing ancestor."""
    candidate = path if path.is_absolute() else path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def free_space(path: Path) -> int:
    """Return the free bytes on the volume that will hold ``path``.

    Works for not-yet-created paths by walking up to the nearest existing ancestor.

    Raises:
        OSError: if the usage query fails.
    """
    return shutil.disk_usage(_existing_ancestor(path)).free


def require_free_space(path: Path, needed_bytes: int, *, what: str) -> None:
    """Raise :class:`DiskSpaceError` unless ``path``'s volume has ``needed_bytes`` free.

    Args:
        path: Any path on the target volume (may not exist yet).
        needed_bytes: Estimated bytes required by the operation.
        what: Human description of the operation, used in the error message.

    Raises:
        DiskSpaceError: with a message showing required vs. available space.
    """
    available = free_space(path)
    if available < needed_bytes:
        raise DiskSpaceError(
            f"Insufficient disk space for {what}: "
            f"need ~{format_bytes(needed_bytes)}, only {format_bytes(available)} free "
            f"on {_existing_ancestor(path)}. Free up space or choose another "
            f"location (video frames use the temp directory; see SKALA_WORKDIR)."
        )


# --------------------------------------------------------------------------------------
# Work directories (resume support)
# --------------------------------------------------------------------------------------
def work_dir(input_path: Path, *options: object, base: Path | None = None) -> Path:
    """Return a deterministic per-(input, options) working directory.

    The directory name is derived from a SHA-256 hash of the resolved input path,
    its size and mtime, plus the option values that affect frame content. The same
    input processed with the same options therefore maps to the same directory,
    which is what makes ``--resume`` work across runs.

    Args:
        input_path: Input file being processed.
        *options: Option values affecting produced content (e.g. scale, model, tile).
        base: Parent directory for work dirs; defaults to ``$SKALA_WORKDIR`` or
            ``<system temp>/skala``.

    Returns:
        A path like ``<base>/skala_work_<16 hex chars>`` (not created).
    """
    resolved = input_path.resolve()
    try:
        stat = resolved.stat()
        identity = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        identity = str(resolved)
    key = "|".join([identity, *(str(opt) for opt in options)])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    parent = base if base is not None else default_work_base()
    return parent / f"skala_work_{digest}"


def default_work_base() -> Path:
    """Return the base directory for working directories.

    Honours the ``SKALA_WORKDIR`` environment variable; otherwise uses
    ``<system temp>/skala``.
    """
    env = os.environ.get("SKALA_WORKDIR")
    if env:
        return Path(env) / "skala"
    return Path(tempfile.gettempdir()) / "skala"


# --------------------------------------------------------------------------------------
# Managed install location
# --------------------------------------------------------------------------------------
def skala_home() -> Path:
    """Return the root directory for Skala-managed downloads.

    Honours ``$SKALA_HOME``; otherwise a per-user data directory is used:
    ``%LOCALAPPDATA%/skala`` on Windows and ``$XDG_DATA_HOME/skala`` (falling
    back to ``~/.local/share/skala``) elsewhere. The directory is not created.
    """
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local) / "skala" if local else Path.home() / "AppData" / "Local" / "skala"
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) / "skala" if xdg else Path.home() / ".local" / "share" / "skala"


# --------------------------------------------------------------------------------------
# Output handling
# --------------------------------------------------------------------------------------
def default_output_path(input_path: Path, scale: int, ext: str) -> Path:
    """Build the default output path ``<name>_skala_<scale>x.<ext>`` next to the input.

    Args:
        input_path: Input file.
        scale: Upscale factor (2, 3, 4, 8).
        ext: Output extension including the dot (e.g. ``.png``) or without.
    """
    extension = ext if ext.startswith(".") else f".{ext}"
    return input_path.with_name(f"{input_path.stem}_skala_{scale}x{extension}")


def ensure_output_path(path: Path, *, overwrite: bool) -> None:
    """Validate that ``path`` may be written: parents created, overwrite policy applied.

    Args:
        path: Proposed output file or directory.
        overwrite: When False, an existing output raises :class:`OutputExistsError`.

    Raises:
        OutputExistsError: output exists and overwrite is False.
        SkalaError: parent directory cannot be created or is not writable.
    """
    if path.exists() and not overwrite:
        raise OutputExistsError(
            f"Output already exists: {path}\n"
            f"Pass --overwrite to replace it, or choose another path with -o."
        )
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SkalaError(f"Cannot create output directory {parent}: {exc}") from exc
    if not os.access(parent, os.W_OK):
        raise SkalaError(f"Output directory is not writable: {parent}")


def validate_input_path(path: Path) -> Path:
    """Validate that ``path`` exists, is a file or directory, and is readable.

    Returns:
        The path unchanged (callers normalize/resolve separately as needed).

    Raises:
        InputError: with an actionable message when validation fails.
    """
    if not path.exists():
        raise InputError(f"Input not found: {path}")
    if path.is_dir():
        if not os.access(path, os.R_OK):
            raise InputError(f"Input directory is not readable: {path}")
        return path
    if not path.is_file():
        raise InputError(f"Input is neither a file nor a directory: {path}")
    if not os.access(path, os.R_OK):
        raise InputError(f"Input file is not readable: {path}")
    return path


# --------------------------------------------------------------------------------------
# Dependencies and hardware
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MissingDependency:
    """A missing external tool together with per-OS installation hints."""

    name: str
    hint_windows: str
    hint_linux: str

    def hint(self) -> str:
        """Return the installation hint for the current platform."""
        return self.hint_windows if sys.platform == "win32" else self.hint_linux


def _missing_ffmpeg_tool(name: str) -> MissingDependency:
    return MissingDependency(
        name=name,
        hint_windows="Install FFmpeg: `winget install Gyan.FFmpeg` "
        "(or `scoop install ffmpeg`), then reopen your terminal.",
        hint_linux="Install FFmpeg: `sudo apt install ffmpeg` (Debian/Ubuntu) "
        "or `sudo dnf install ffmpeg` (Fedora), or use your package manager.",
    )


def _missing_engine() -> MissingDependency:
    return MissingDependency(
        name="realesrgan-ncnn-vulkan",
        hint_windows="Download the release zip from "
        "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases, extract it, "
        "then either put realesrgan-ncnn-vulkan.exe in .\\bin\\, add its folder to "
        "PATH, set SKALA_ENGINE=<full path to exe>, or pass --engine-path.",
        hint_linux="Download the linux release from "
        "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases, chmod +x the "
        "binary, then put it in ./bin/, add it to PATH, set SKALA_ENGINE=<path>, "
        "or pass --engine-path.",
    )


def missing_ffmpeg() -> list[MissingDependency]:
    """Return the subset of {ffmpeg, ffprobe} that is not resolvable on PATH."""
    missing: list[MissingDependency] = []
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            missing.append(_missing_ffmpeg_tool(tool))
    return missing


@dataclass(frozen=True)
class HardwareStatus:
    """Result of best-effort hardware detection.

    Attributes:
        vulkan_available: True/False when confident, None when undeterminable.
        vulkan_detail: Human-readable evidence (GPU names, tool output, or reason).
        cpu_count: Logical CPU count.
    """

    vulkan_available: bool | None
    vulkan_detail: str
    cpu_count: int


def _vulkan_from_vulkaninfo() -> tuple[bool | None, str]:
    exe = shutil.which("vulkaninfo")
    if exe is None:
        return None, "vulkaninfo not found"
    try:
        proc = subprocess.run(
            [exe, "--summary"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"vulkaninfo failed to run: {exc}"
    if proc.returncode != 0:
        return False, f"vulkaninfo exited with {proc.returncode}"
    devices = [line.strip() for line in (proc.stdout or "").splitlines() if "deviceName" in line]
    if devices:
        return True, "; ".join(devices)
    return None, "vulkaninfo ran but reported no devices"


def _vulkan_windows() -> tuple[bool | None, str]:
    loader = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "vulkan-1.dll"
    if not loader.exists():
        # No loader at all: Vulkan is very unlikely to be usable.
        if shutil.which("vulkaninfo") is None:
            return False, "Vulkan loader (vulkan-1.dll) not found"
        return None, "vulkan-1.dll not found"
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "Vulkan loader present; GPU query failed"
    gpus = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if not gpus:
        return False, "Vulkan loader present but no display adapter reported"
    return True, "; ".join(gpus)


def _vulkan_linux() -> tuple[bool | None, str]:
    icd_dir = Path("/usr/share/vulkan/icd.d")
    has_icd = icd_dir.is_dir() and any(icd_dir.iterdir())
    try:
        proc = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpus = [
            line.strip()
            for line in (proc.stdout or "").splitlines()
            if "VGA" in line or "3D controller" in line or "Display controller" in line
        ]
    except (OSError, subprocess.TimeoutExpired):
        gpus = []
    if gpus:
        return True, "; ".join(gpus)
    if has_icd:
        return True, "Vulkan ICD drivers installed"
    return False, "No GPU detected and no Vulkan ICD drivers installed"


def detect_hardware() -> HardwareStatus:
    """Best-effort detection of Vulkan GPU availability and CPU count.

    Never raises: any probe failure degrades to ``vulkan_available=None`` with an
    explanatory detail, so callers can warn without aborting.

    Returns:
        HardwareStatus with detection results.
    """
    cpu = os.cpu_count() or 1
    status: tuple[bool | None, str]
    try:
        if sys.platform == "win32":
            status = _vulkan_windows()
        elif sys.platform.startswith("linux"):
            status = _vulkan_linux()
        else:
            status = (None, f"unsupported platform for detection: {sys.platform}")
    except Exception as exc:
        status = (None, f"Vulkan probe raised: {exc}")
    # vulkaninfo, when present, is the most authoritative signal; prefer it.
    try:
        info_status, info_detail = _vulkan_from_vulkaninfo()
    except Exception as exc:
        info_status, info_detail = None, f"vulkaninfo probe raised: {exc}"
    if info_status is not None:
        status = (info_status, info_detail)
    available, detail = status
    return HardwareStatus(vulkan_available=available, vulkan_detail=detail, cpu_count=cpu)


def cpu_count() -> int:
    """Return the number of logical CPUs (at least 1)."""
    return os.cpu_count() or 1
