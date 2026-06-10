"""Variable-radius and longitudinally varying cross-section models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.coordinates import FloatArray
from synthworkshop.cross_sections.base import (
    CrossSectionEvaluation,
    validate_local_coordinates,
    validate_longitudinal_coordinates,
    validate_positive_float,
)


def _validate_finite_float(value: float, *, name: str) -> float:
    """Validate a finite floating-point value."""

    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


@dataclass(frozen=True)
class VariableCircularCrossSection:
    """Circular cross-section with linearly varying longitudinal radius."""

    radius_start_mm: float
    radius_end_mm: float
    length_mm: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radius_start_mm",
            validate_positive_float(self.radius_start_mm, name="radius_start_mm"),
        )
        object.__setattr__(
            self,
            "radius_end_mm",
            validate_positive_float(self.radius_end_mm, name="radius_end_mm"),
        )
        object.__setattr__(
            self,
            "length_mm",
            validate_positive_float(self.length_mm, name="length_mm"),
        )

    @property
    def kind(self) -> str:
        """Cross-section kind name."""

        return "variable_circle_linear"

    @property
    def area_mm2(self) -> float:
        """Representative area using the mean endpoint radius."""

        mean_radius = 0.5 * (self.radius_start_mm + self.radius_end_mm)
        return float(np.pi * mean_radius**2)

    @property
    def radius_range_mm(self) -> tuple[float, float]:
        """Minimum and maximum possible radius."""

        return (
            min(self.radius_start_mm, self.radius_end_mm),
            max(self.radius_start_mm, self.radius_end_mm),
        )

    def radius_at(self, longitudinal_mm: ArrayLike) -> FloatArray:
        """Evaluate radius at longitudinal positions."""

        longitudinal = np.asarray(longitudinal_mm, dtype=float)
        if not np.all(np.isfinite(longitudinal)):
            raise ValueError("longitudinal_mm contains non-finite values.")
        fraction = np.clip(longitudinal / self.length_mm, 0.0, 1.0)
        radius = (
            self.radius_start_mm
            + (self.radius_end_mm - self.radius_start_mm) * fraction
        )
        return np.asarray(radius, dtype=float)

    def evaluate(
        self,
        u_mm: ArrayLike,
        v_mm: ArrayLike,
        *,
        longitudinal_mm: ArrayLike | None = None,
    ) -> CrossSectionEvaluation:
        """Evaluate a linearly variable circular cross-section."""

        if longitudinal_mm is None:
            raise ValueError(
                "longitudinal_mm is required for VariableCircularCrossSection."
            )

        u, v = validate_local_coordinates(u_mm, v_mm)
        longitudinal = validate_longitudinal_coordinates(
            longitudinal_mm,
            shape=u.shape,
        )
        radial = np.sqrt(u**2 + v**2)
        radius = self.radius_at(longitudinal)
        rho = radial / radius
        inside = rho <= 1.0

        return CrossSectionEvaluation(
            u_mm=u,
            v_mm=v,
            rho=rho,
            radial_distance_mm=radial,
            inside=inside,
            radius_mm=radius,
            metadata=self.summary(),
        )

    def boundary_radius_mm(self, angle_radians: ArrayLike) -> FloatArray:
        """Return the maximum boundary radius envelope."""

        angles = np.asarray(angle_radians, dtype=float)
        if not np.all(np.isfinite(angles)):
            raise ValueError("angle_radians contains non-finite values.")
        return np.full(angles.shape, self.radius_range_mm[1], dtype=float)

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "radius_start_mm": self.radius_start_mm,
            "radius_end_mm": self.radius_end_mm,
            "length_mm": self.length_mm,
            "radius_min_mm": self.radius_range_mm[0],
            "radius_max_mm": self.radius_range_mm[1],
            "representative_area_mm2": self.area_mm2,
        }


@dataclass(frozen=True)
class VariableEllipticCrossSection:
    """Axis-aligned ellipse with linearly varying longitudinal semi-axes."""

    semi_axis_u_start_mm: float
    semi_axis_u_end_mm: float
    semi_axis_v_start_mm: float
    semi_axis_v_end_mm: float
    length_mm: float

    def __post_init__(self) -> None:
        for name in (
            "semi_axis_u_start_mm",
            "semi_axis_u_end_mm",
            "semi_axis_v_start_mm",
            "semi_axis_v_end_mm",
            "length_mm",
        ):
            object.__setattr__(
                self,
                name,
                validate_positive_float(getattr(self, name), name=name),
            )

    @property
    def kind(self) -> str:
        """Cross-section kind name."""

        return "variable_ellipse_linear"

    @property
    def area_mm2(self) -> float:
        """Representative area using mean endpoint semi-axes."""

        mean_u = 0.5 * (self.semi_axis_u_start_mm + self.semi_axis_u_end_mm)
        mean_v = 0.5 * (self.semi_axis_v_start_mm + self.semi_axis_v_end_mm)
        return float(np.pi * mean_u * mean_v)

    @property
    def semi_axis_u_range_mm(self) -> tuple[float, float]:
        """Minimum and maximum u semi-axis."""

        return (
            min(self.semi_axis_u_start_mm, self.semi_axis_u_end_mm),
            max(self.semi_axis_u_start_mm, self.semi_axis_u_end_mm),
        )

    @property
    def semi_axis_v_range_mm(self) -> tuple[float, float]:
        """Minimum and maximum v semi-axis."""

        return (
            min(self.semi_axis_v_start_mm, self.semi_axis_v_end_mm),
            max(self.semi_axis_v_start_mm, self.semi_axis_v_end_mm),
        )

    def axes_at(self, longitudinal_mm: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Evaluate semi-axes at longitudinal positions."""

        longitudinal = np.asarray(longitudinal_mm, dtype=float)
        if not np.all(np.isfinite(longitudinal)):
            raise ValueError("longitudinal_mm contains non-finite values.")
        fraction = np.clip(longitudinal / self.length_mm, 0.0, 1.0)
        semi_u = (
            self.semi_axis_u_start_mm
            + (self.semi_axis_u_end_mm - self.semi_axis_u_start_mm) * fraction
        )
        semi_v = (
            self.semi_axis_v_start_mm
            + (self.semi_axis_v_end_mm - self.semi_axis_v_start_mm) * fraction
        )
        return np.asarray(semi_u, dtype=float), np.asarray(semi_v, dtype=float)

    def evaluate(
        self,
        u_mm: ArrayLike,
        v_mm: ArrayLike,
        *,
        longitudinal_mm: ArrayLike | None = None,
    ) -> CrossSectionEvaluation:
        """Evaluate a linearly variable elliptic cross-section."""

        if longitudinal_mm is None:
            raise ValueError(
                "longitudinal_mm is required for VariableEllipticCrossSection."
            )

        u, v = validate_local_coordinates(u_mm, v_mm)
        longitudinal = validate_longitudinal_coordinates(
            longitudinal_mm,
            shape=u.shape,
        )
        semi_u, semi_v = self.axes_at(longitudinal)
        rho = np.sqrt((u / semi_u) ** 2 + (v / semi_v) ** 2)
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
        """Return the maximum endpoint-axis envelope boundary radius."""

        angle = np.asarray(angle_radians, dtype=float)
        if not np.all(np.isfinite(angle)):
            raise ValueError("angle_radians contains non-finite values.")

        semi_u = self.semi_axis_u_range_mm[1]
        semi_v = self.semi_axis_v_range_mm[1]
        denominator = np.sqrt(
            (np.cos(angle) / semi_u) ** 2 + (np.sin(angle) / semi_v) ** 2
        )
        return 1.0 / denominator

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "semi_axis_u_start_mm": self.semi_axis_u_start_mm,
            "semi_axis_u_end_mm": self.semi_axis_u_end_mm,
            "semi_axis_v_start_mm": self.semi_axis_v_start_mm,
            "semi_axis_v_end_mm": self.semi_axis_v_end_mm,
            "length_mm": self.length_mm,
            "semi_axis_u_min_mm": self.semi_axis_u_range_mm[0],
            "semi_axis_u_max_mm": self.semi_axis_u_range_mm[1],
            "semi_axis_v_min_mm": self.semi_axis_v_range_mm[0],
            "semi_axis_v_max_mm": self.semi_axis_v_range_mm[1],
            "representative_area_mm2": self.area_mm2,
        }


@dataclass(frozen=True)
class RotatingEllipticCrossSection:
    """Ellipse whose local axes rotate linearly along longitudinal position."""

    semi_axis_u_mm: float
    semi_axis_v_mm: float
    length_mm: float
    angle_start_radians: float = 0.0
    angle_end_radians: float = np.pi / 2.0

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
        object.__setattr__(
            self,
            "length_mm",
            validate_positive_float(self.length_mm, name="length_mm"),
        )
        object.__setattr__(
            self,
            "angle_start_radians",
            _validate_finite_float(
                self.angle_start_radians,
                name="angle_start_radians",
            ),
        )
        object.__setattr__(
            self,
            "angle_end_radians",
            _validate_finite_float(
                self.angle_end_radians,
                name="angle_end_radians",
            ),
        )

    @property
    def kind(self) -> str:
        """Cross-section kind name."""

        return "rotating_ellipse_linear"

    @property
    def area_mm2(self) -> float:
        """Analytic elliptic area."""

        return float(np.pi * self.semi_axis_u_mm * self.semi_axis_v_mm)

    def angle_at(self, longitudinal_mm: ArrayLike) -> FloatArray:
        """Evaluate rotation angle at longitudinal positions."""

        longitudinal = np.asarray(longitudinal_mm, dtype=float)
        if not np.all(np.isfinite(longitudinal)):
            raise ValueError("longitudinal_mm contains non-finite values.")
        fraction = np.clip(longitudinal / self.length_mm, 0.0, 1.0)
        angle = (
            self.angle_start_radians
            + (self.angle_end_radians - self.angle_start_radians) * fraction
        )
        return np.asarray(angle, dtype=float)

    def evaluate(
        self,
        u_mm: ArrayLike,
        v_mm: ArrayLike,
        *,
        longitudinal_mm: ArrayLike | None = None,
    ) -> CrossSectionEvaluation:
        """Evaluate a longitudinally rotating elliptic cross-section."""

        if longitudinal_mm is None:
            raise ValueError(
                "longitudinal_mm is required for RotatingEllipticCrossSection."
            )

        u, v = validate_local_coordinates(u_mm, v_mm)
        longitudinal = validate_longitudinal_coordinates(
            longitudinal_mm,
            shape=u.shape,
        )
        angle = self.angle_at(longitudinal)

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotated_u = cos_angle * u + sin_angle * v
        rotated_v = -sin_angle * u + cos_angle * v

        rho = np.sqrt(
            (rotated_u / self.semi_axis_u_mm) ** 2
            + (rotated_v / self.semi_axis_v_mm) ** 2
        )
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
        """Return a conservative maximum-radius envelope.

        The true boundary radius depends on both direction and longitudinal
        rotation angle. For direction-only support queries this returns the
        maximum possible semi-axis as a conservative envelope.
        """

        angles = np.asarray(angle_radians, dtype=float)
        if not np.all(np.isfinite(angles)):
            raise ValueError("angle_radians contains non-finite values.")
        return np.full(
            angles.shape,
            max(self.semi_axis_u_mm, self.semi_axis_v_mm),
            dtype=float,
        )

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "semi_axis_u_mm": self.semi_axis_u_mm,
            "semi_axis_v_mm": self.semi_axis_v_mm,
            "length_mm": self.length_mm,
            "angle_start_radians": self.angle_start_radians,
            "angle_end_radians": self.angle_end_radians,
            "area_mm2": self.area_mm2,
        }
