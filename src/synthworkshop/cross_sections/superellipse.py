"""Superellipse cross-section model."""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.coordinates import FloatArray
from synthworkshop.cross_sections.base import (
    CrossSectionEvaluation,
    validate_local_coordinates,
    validate_positive_float,
)


@dataclass(frozen=True)
class SuperellipseCrossSection:
    """Axis-aligned superellipse in a local u-v plane.

    The inside support is defined by:

        |u / a|^n + |v / b|^n <= 1

    where ``a`` and ``b`` are the semi-axes and ``n`` is the exponent. An
    exponent of 2 gives a standard ellipse. Larger values give progressively
    boxier or flatter support, useful for ribbon-like validation cases without
    yet introducing a separate sheet/ribbon primitive.
    """

    semi_axis_u_mm: float
    semi_axis_v_mm: float
    exponent: float = 4.0

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
        exponent = validate_positive_float(self.exponent, name="exponent")
        if exponent < 1.0:
            raise ValueError("exponent must be at least 1 for convex support.")
        object.__setattr__(self, "exponent", exponent)

    @property
    def kind(self) -> str:
        """Cross-section kind name."""

        return "superellipse"

    @property
    def area_mm2(self) -> float:
        """Analytic superellipse area."""

        n = self.exponent
        return float(
            4.0
            * self.semi_axis_u_mm
            * self.semi_axis_v_mm
            * gamma(1.0 + 1.0 / n) ** 2
            / gamma(1.0 + 2.0 / n)
        )

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
        """Evaluate normalised superellipse radius and inside support."""

        _ = longitudinal_mm
        u, v = validate_local_coordinates(u_mm, v_mm)

        term = (
            np.abs(u / self.semi_axis_u_mm) ** self.exponent
            + np.abs(v / self.semi_axis_v_mm) ** self.exponent
        )
        rho = term ** (1.0 / self.exponent)
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

        denominator = (
            np.abs(np.cos(angle) / self.semi_axis_u_mm) ** self.exponent
            + np.abs(np.sin(angle) / self.semi_axis_v_mm) ** self.exponent
        ) ** (1.0 / self.exponent)
        return 1.0 / denominator

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "semi_axis_u_mm": self.semi_axis_u_mm,
            "semi_axis_v_mm": self.semi_axis_v_mm,
            "exponent": self.exponent,
            "area_mm2": self.area_mm2,
        }
