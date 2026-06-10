"""Circular cross-section model."""

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
class CircularCrossSection:
    """Circular cross-section in a local u-v plane."""

    radius_mm: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radius_mm",
            validate_positive_float(self.radius_mm, name="radius_mm"),
        )

    @property
    def kind(self) -> str:
        """Cross-section kind name."""

        return "circle"

    @property
    def area_mm2(self) -> float:
        """Analytic circular area."""

        return float(np.pi * self.radius_mm**2)

    @property
    def semi_axes_mm(self) -> tuple[float, float]:
        """Semi-axes in the local u and v directions."""

        return (self.radius_mm, self.radius_mm)

    def evaluate(
        self,
        u_mm: ArrayLike,
        v_mm: ArrayLike,
        *,
        longitudinal_mm: ArrayLike | None = None,
    ) -> CrossSectionEvaluation:
        """Evaluate normalised circular radius and inside support."""

        _ = longitudinal_mm
        u, v = validate_local_coordinates(u_mm, v_mm)
        radial = np.sqrt(u**2 + v**2)
        rho = radial / self.radius_mm
        inside = rho <= 1.0
        return CrossSectionEvaluation(
            u_mm=u,
            v_mm=v,
            rho=rho,
            radial_distance_mm=radial,
            inside=inside,
            radius_mm=np.full(radial.shape, self.radius_mm, dtype=float),
            metadata=self.summary(),
        )

    def boundary_radius_mm(self, angle_radians: ArrayLike) -> FloatArray:
        """Return the circular boundary radius for any local direction."""

        angles = np.asarray(angle_radians, dtype=float)
        if not np.all(np.isfinite(angles)):
            raise ValueError("angle_radians contains non-finite values.")
        return np.full(angles.shape, self.radius_mm, dtype=float)

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "radius_mm": self.radius_mm,
            "semi_axis_u_mm": self.radius_mm,
            "semi_axis_v_mm": self.radius_mm,
            "area_mm2": self.area_mm2,
        }
