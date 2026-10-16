"""Unit tests for the pure scale planner (skala.scaling)."""

from __future__ import annotations

import pytest
from skala.scaling import VALID_SCALES, ScalePlan, plan_scale

NATIVE_4 = (4,)
NATIVE_ALL = (2, 3, 4)


class TestScaleMatrix:
    """plan_scale must choose the right passes for every scale/model combination."""

    @pytest.mark.parametrize("scale", VALID_SCALES)
    def test_scale4_model_produces_valid_plan(self, scale: int) -> None:
        plan = plan_scale(1920, 1080, scale, native_scales=NATIVE_4)
        assert plan.scale == scale
        assert plan.final_size == (1920 * scale, 1080 * scale)
        assert plan.passes  # at least one pass
        assert all(p in (2, 3, 4) for p in plan.passes)
        assert plan.pass_count == len(plan.passes)

    @pytest.mark.parametrize("scale", VALID_SCALES)
    def test_full_scale_model_produces_valid_plan(self, scale: int) -> None:
        plan = plan_scale(100, 50, scale, native_scales=NATIVE_ALL)
        assert plan.final_size == (100 * scale, 50 * scale)
        assert all(p in (2, 3, 4) for p in plan.passes)

    def test_2x_with_4x_only_model_downscales(self) -> None:
        plan = plan_scale(100, 100, 2, native_scales=NATIVE_4)
        assert plan.passes == (4,)
        assert plan.intermediate_size == (400, 400)
        assert plan.final_size == (200, 200)
        assert plan.needs_resize is True

    def test_2x_with_native_2x_model_is_exact(self) -> None:
        plan = plan_scale(100, 100, 2, native_scales=NATIVE_ALL)
        assert plan.passes == (2,)
        assert plan.intermediate_size == plan.final_size == (200, 200)
        assert plan.needs_resize is False

    def test_3x_with_4x_only_model_downscales(self) -> None:
        plan = plan_scale(1920, 1080, 3, native_scales=NATIVE_4)
        assert plan.passes == (4,)
        assert plan.intermediate_size == (7680, 4320)
        assert plan.final_size == (5760, 3240)
        assert plan.needs_resize is True

    def test_3x_with_native_3x_model_is_exact(self) -> None:
        plan = plan_scale(1920, 1080, 3, native_scales=NATIVE_ALL)
        assert plan.passes == (3,)
        assert plan.final_size == (5760, 3240)
        assert plan.needs_resize is False

    def test_4x_is_always_single_pass(self) -> None:
        for native in (NATIVE_4, NATIVE_ALL):
            plan = plan_scale(64, 48, 4, native_scales=native)
            assert plan.passes == (4,)
            assert plan.intermediate_size == plan.final_size == (256, 192)
            assert plan.needs_resize is False

    def test_8x_with_2x_capable_model_is_two_pass_exact(self) -> None:
        plan = plan_scale(100, 100, 8, native_scales=NATIVE_ALL)
        assert plan.passes == (4, 2)
        assert plan.intermediate_size == (800, 800)
        assert plan.final_size == (800, 800)
        assert plan.needs_resize is False

    def test_8x_with_4x_only_model_is_single_pass_plus_upscale(self) -> None:
        plan = plan_scale(100, 100, 8, native_scales=NATIVE_4)
        assert plan.passes == (4,)
        assert plan.intermediate_size == (400, 400)
        assert plan.final_size == (800, 800)
        assert plan.needs_resize is True


class TestWarnings:
    def test_8x_always_warns(self) -> None:
        for native in (NATIVE_4, NATIVE_ALL):
            plan = plan_scale(10, 10, 8, native_scales=native)
            assert plan.warnings, "8x must warn about artifacts/VRAM/disk"
            assert any("artifacts" in w for w in plan.warnings)

    def test_8x_resize_model_has_extra_warning(self) -> None:
        plan = plan_scale(10, 10, 8, native_scales=NATIVE_4)
        assert len(plan.warnings) == 2
        assert any("LANCZOS" in w for w in plan.warnings)

    def test_8x_two_pass_model_has_no_resize_warning(self) -> None:
        plan = plan_scale(10, 10, 8, native_scales=NATIVE_ALL)
        assert len(plan.warnings) == 1

    @pytest.mark.parametrize("scale", [2, 3, 4])
    def test_non_8x_has_no_warnings(self, scale: int) -> None:
        plan = plan_scale(10, 10, scale, native_scales=NATIVE_4)
        assert plan.warnings == ()


class TestValidation:
    def test_invalid_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="scale must be one of"):
            plan_scale(10, 10, 5)

    @pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (-5, 10)])
    def test_non_positive_dimensions_raise(self, width: int, height: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            plan_scale(width, height, 4)

    def test_empty_native_scales_raise(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            plan_scale(10, 10, 4, native_scales=())

    def test_native_scales_without_4_raise(self) -> None:
        with pytest.raises(ValueError, match="must include 4"):
            plan_scale(10, 10, 4, native_scales=(2,))

    def test_native_scales_with_unknown_value_raise(self) -> None:
        with pytest.raises(ValueError, match="subset of"):
            plan_scale(10, 10, 4, native_scales=(4, 8))


class TestDescribe:
    def test_describe_exact_pass(self) -> None:
        plan = plan_scale(64, 64, 4)
        assert plan.describe() == "engine 4x -> 256x256"

    def test_describe_two_pass(self) -> None:
        plan = plan_scale(64, 64, 8, native_scales=NATIVE_ALL)
        assert plan.describe() == "engine 4x -> engine 2x -> 512x512"

    def test_describe_downscale(self) -> None:
        plan = plan_scale(64, 64, 2, native_scales=NATIVE_4)
        text = plan.describe()
        assert "engine 4x" in text
        assert "LANCZOS resize down" in text
        assert "256x256" in text and "128x128" in text

    def test_describe_upscale(self) -> None:
        plan = plan_scale(64, 64, 8, native_scales=NATIVE_4)
        assert "LANCZOS resize up" in plan.describe()


def test_scale_plan_is_frozen() -> None:
    plan = plan_scale(10, 10, 4)
    assert isinstance(plan, ScalePlan)
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        plan.scale = 2  # type: ignore[misc]
