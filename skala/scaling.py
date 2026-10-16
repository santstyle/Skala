"""Pure scale-planning logic: map a requested scale factor to engine passes.

Everything in this module is pure (no I/O, no subprocess) so it can be unit-tested
trivially. The planner knows nothing about specific model names: callers pass the
set of scales the chosen model supports natively (``native_scales``).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

__all__ = ["VALID_SCALES", "ScalePlan", "plan_scale"]

#: Scale factors Skala accepts from the user.
VALID_SCALES: tuple[int, ...] = (2, 3, 4, 8)

#: Scales any shipped realesrgan model can produce in a single engine pass.
_ALL_POSSIBLE_NATIVE: frozenset[int] = frozenset({2, 3, 4})


@dataclass(frozen=True)
class ScalePlan:
    """Result of planning how to reach a requested scale.

    Attributes:
        width: Input width in pixels.
        height: Input height in pixels.
        scale: Requested overall scale factor (2, 3, 4 or 8).
        passes: Engine scale factors to apply, in order (each is 2, 3 or 4).
        intermediate_size: ``(width, height)`` produced after all engine passes.
        final_size: ``(width, height)`` the caller must end up with.
        needs_resize: True when a final LANCZOS step (up or down) is required
            because the engine passes do not land exactly on ``final_size``.
        warnings: Human-readable warnings the UI should surface (e.g. for 8x).
    """

    width: int
    height: int
    scale: int
    passes: tuple[int, ...]
    intermediate_size: tuple[int, int]
    final_size: tuple[int, int]
    needs_resize: bool
    warnings: tuple[str, ...]

    @property
    def pass_count(self) -> int:
        """Number of engine invocations required."""
        return len(self.passes)

    def describe(self) -> str:
        """Return a one-line human description of the plan (for --dry-run)."""
        engine = " -> ".join(f"engine {p}x" for p in self.passes)
        w, h = self.final_size
        if self.needs_resize:
            intermediate_w, intermediate_h = self.intermediate_size
            direction = "up" if intermediate_w < w and intermediate_h < h else "down"
            return (
                f"{engine} (intermediate {intermediate_w}x{intermediate_h}) "
                f"-> LANCZOS resize {direction} to {w}x{h}"
            )
        return f"{engine} -> {w}x{h}"


def _passes_for(scale: int, native: Collection[int]) -> tuple[int, ...]:
    """Choose engine passes for ``scale`` given natively supported scales.

    Rules (see README for rationale):

    * 2x: native 2x pass if available, else a 4x pass downscaled afterwards.
    * 3x: native 3x pass if available, else a 4x pass downscaled afterwards.
    * 4x: always a single 4x pass (every shipped model supports 4x).
    * 8x: 4x then 2x when the model has a native 2x pass; otherwise a single
      4x pass followed by a LANCZOS 2x upscale (a 4x+4x chain would require an
      impractical 16x intermediate resolution).
    """
    if scale == 2:
        return (2,) if 2 in native else (4,)
    if scale == 3:
        return (3,) if 3 in native else (4,)
    if scale == 4:
        return (4,)
    # scale == 8
    return (4, 2) if 2 in native else (4,)


def _warnings_for(scale: int, needs_resize: bool) -> tuple[str, ...]:
    """Return user-facing warnings for the planned scale."""
    warnings: list[str] = []
    if scale == 8:
        warnings.append(
            "8x amplifies upscaling artifacts and needs substantially more "
            "VRAM, RAM and disk than 4x; expect slower runs and softer results."
        )
        if needs_resize:
            warnings.append(
                "The chosen model has no native 2x pass, so 8x = one 4x engine "
                "pass + LANCZOS 2x resize (softer than a true two-pass 8x). "
                "Use realesr-animevideov3 for a true 4x->2x chain."
            )
    return tuple(warnings)


def plan_scale(
    width: int,
    height: int,
    scale: int,
    *,
    native_scales: Collection[int] = (4,),
) -> ScalePlan:
    """Plan how to upscale ``width`` x ``height`` by ``scale``.

    This is the single source of truth for pass selection; the image and video
    pipelines both consume the returned :class:`ScalePlan`.

    Args:
        width: Input width in pixels (> 0).
        height: Input height in pixels (> 0).
        scale: Requested factor, one of :data:`VALID_SCALES`.
        native_scales: Engine scales the chosen model supports in a single pass.
            Must be a non-empty subset of ``{2, 3, 4}`` containing 4 (all shipped
            realesrgan models support 4x).

    Returns:
        ScalePlan describing engine passes, intermediate/final sizes and warnings.

    Raises:
        ValueError: on non-positive dimensions, an unsupported ``scale``, or an
            invalid ``native_scales`` collection.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions must be positive, got {width}x{height}")
    if scale not in VALID_SCALES:
        raise ValueError(f"scale must be one of {VALID_SCALES}, got {scale}")

    native = frozenset(native_scales)
    if not native:
        raise ValueError("native_scales must not be empty")
    if not native <= _ALL_POSSIBLE_NATIVE:
        raise ValueError(
            f"native_scales must be a subset of {sorted(_ALL_POSSIBLE_NATIVE)}, "
            f"got {sorted(native)}"
        )
    if 4 not in native:
        raise ValueError("native_scales must include 4 (all shipped models support 4x)")

    passes = _passes_for(scale, native)
    intermediate = (width * passes[0], height * passes[0])
    if len(passes) == 2:
        intermediate = (intermediate[0] * passes[1], intermediate[1] * passes[1])
    final = (width * scale, height * scale)
    needs_resize = intermediate != final

    return ScalePlan(
        width=width,
        height=height,
        scale=scale,
        passes=passes,
        intermediate_size=intermediate,
        final_size=final,
        needs_resize=needs_resize,
        warnings=_warnings_for(scale, needs_resize),
    )
