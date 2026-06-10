"""Elliptic cross-section model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.coordinates import FloatArray
from synthworkshop.cross_sections.base import (
    CrossSectionEvaluation,
    validate_local_coordinates,
    validate_positive_float,
)


@dataclass(frozen=True)
class EllipticCrossSection:
    """Axis-aligned ellipse in a local u-v plane."""

    semi_axis_u_mm: float
    semi_axis_v_mm: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semi_axis_u_mm",
            validate_positive_float(self.semi_axis_u_mm, name="semi_axis_u_mm"),
        )
        object.__setattr__(
            self,
            "semi_axis_v_mm",
            validate_positive_float(self.semi_axis_v_mm, name="semi_axis_v_mm"),
        )

    @property
    def kind(self) -> str:
        """Cross-section kind name."""

        return "ellipse"

    @property
    def area_mm2(self) -> float:
        """Analytic elliptic area."""

        return float(np.pi * self.semi_axis_u_mm * self.semi_axis_v_mm)

    @property
    def semi_axes_mm(self) -> tuple[float, float]:
        """Semi-axes in the local u and v directions."""

        return (self.semi_axis_u_mm, self.semi_axis_v_mm)

    def evaluate(
        self,
        u_mm: ArrayLike,
        v_mm: ArrayLike,
        *,
        longitudinal_mm: ArrayLike | None = None,
    ) -> CrossSectionEvaluation:
        """Evaluate normalised elliptic radius and inside support."""

        _ = longitudinal_mm
        u, v = validate_local_coordinates(u_mm, v_mm)
        rho = np.sqrt((u / self.semi_axis_u_mm) ** 2 + (v / self.semi_axis_v_mm) ** 2)
        radial = np.sqrt(u**2 + v**2)
        inside = rho <= 1.0
        return CrossSectionEvaluation(
            u_mm=u,
            v_mm=v,
            rho=rho,
            radial_distance_mm=radial,
            inside=inside,
            metadata=self.summary(),
        )

    def boundary_radius_mm(self, angle_radians: ArrayLike) -> FloatArray:
        """Return boundary radius for directions in the local u-v plane."""

        angle = np.asarray(angle_radians, dtype=float)
        if not np.all(np.isfinite(angle)):
            raise ValueError("angle_radians contains non-finite values.")

        cos_term = np.cos(angle) / self.semi_axis_u_mm
        sin_term = np.sin(angle) / self.semi_axis_v_mm
        denominator = np.sqrt(cos_term**2 + sin_term**2)
        return 1.0 / denominator

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "semi_axis_u_mm": self.semi_axis_u_mm,
            "semi_axis_v_mm": self.semi_axis_v_mm,
            "area_mm2": self.area_mm2,
        }
