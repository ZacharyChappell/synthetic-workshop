"""Ribbon-like cross-section model."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from synthworkshop.coordinates import FloatArray
from synthworkshop.cross_sections.base import (
    CrossSectionEvaluation,
    validate_positive_float,
)
from synthworkshop.cross_sections.superellipse import SuperellipseCrossSection


@dataclass(frozen=True)
class RibbonCrossSection:
    """Flattened ribbon-like local cross-section.

    This is a semantic wrapper around a superellipse. It is useful when scene
    specifications should express a flattened support using width/thickness
    terminology rather than semi-axis terminology.

    The support is equivalent to:

        |u / (width_mm / 2)|^n + |v / (thickness_mm / 2)|^n <= 1

    where ``n`` is ``exponent``.
    """

    width_mm: float
    thickness_mm: float
    exponent: float = 6.0

    def __post_init__(self) -> None:
        width = validate_positive_float(self.width_mm, name="width_mm")
        thickness = validate_positive_float(self.thickness_mm, name="thickness_mm")
        exponent = validate_positive_float(self.exponent, name="exponent")
        if exponent < 1.0:
            raise ValueError("exponent must be at least 1 for convex support.")
        object.__setattr__(self, "width_mm", width)
        object.__setattr__(self, "thickness_mm", thickness)
        object.__setattr__(self, "exponent", exponent)

    @property
    def kind(self) -> str:
        """Cross-section kind name."""

        return "ribbon"

    @property
    def semi_axis_u_mm(self) -> float:
        """Semi-axis along local u."""

        return 0.5 * self.width_mm

    @property
    def semi_axis_v_mm(self) -> float:
        """Semi-axis along local v."""

        return 0.5 * self.thickness_mm

    @property
    def semi_axes_mm(self) -> tuple[float, float]:
        """Semi-axes in the local u and v directions."""

        return (self.semi_axis_u_mm, self.semi_axis_v_mm)

    @property
    def area_mm2(self) -> float:
        """Analytic ribbon/superellipse area."""

        return self._as_superellipse().area_mm2

    def _as_superellipse(self) -> SuperellipseCrossSection:
        """Return the equivalent superellipse model."""

        return SuperellipseCrossSection(
            semi_axis_u_mm=self.semi_axis_u_mm,
            semi_axis_v_mm=self.semi_axis_v_mm,
            exponent=self.exponent,
        )

    def evaluate(
        self,
        u_mm: ArrayLike,
        v_mm: ArrayLike,
        *,
        longitudinal_mm: ArrayLike | None = None,
    ) -> CrossSectionEvaluation:
        """Evaluate ribbon support at local coordinates."""

        result = self._as_superellipse().evaluate(
            u_mm,
            v_mm,
            longitudinal_mm=longitudinal_mm,
        )
        return CrossSectionEvaluation(
            u_mm=result.u_mm,
            v_mm=result.v_mm,
            rho=result.rho,
            radial_distance_mm=result.radial_distance_mm,
            inside=result.inside,
            radius_mm=result.radius_mm,
            metadata=self.summary(),
        )

    def boundary_radius_mm(self, angle_radians: ArrayLike) -> FloatArray:
        """Return boundary radius for directions in the local u-v plane."""

        return self._as_superellipse().boundary_radius_mm(angle_radians)

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "width_mm": self.width_mm,
            "thickness_mm": self.thickness_mm,
            "semi_axis_u_mm": self.semi_axis_u_mm,
            "semi_axis_v_mm": self.semi_axis_v_mm,
            "exponent": self.exponent,
            "area_mm2": self.area_mm2,
            "equivalent_model": "superellipse",
        }
