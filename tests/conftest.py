"""Shared pytest configuration.

Integration tests (marked ``integration``) drive the real
``realesrgan-ncnn-vulkan`` and ffmpeg binaries, so they are opt-in: they only
run when ``SKALA_RUN_INTEGRATION=1`` is set. The rest of the suite mocks every
subprocess and always runs, so a plain ``pytest`` stays green on a machine with
no external tools installed.
"""

from __future__ import annotations

import os

import pytest

_ENV_FLAG = "SKALA_RUN_INTEGRATION"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``integration`` tests unless they were explicitly opted into."""
    if os.environ.get(_ENV_FLAG) == "1":
        return
    skip = pytest.mark.skip(
        reason=f"set {_ENV_FLAG}=1 to run integration tests (needs the real engine/ffmpeg)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
