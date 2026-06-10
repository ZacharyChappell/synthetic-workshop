"""Base containers and helpers for local cross-section models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.coordinates import FloatArray


def validate_positive_float(value: float, *, name: str) -> float:
    """Validate a finite positive floating-point value."""

    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return out


def validate_local_coordinates(
    u_mm: ArrayLike,
    v_mm: ArrayLike,
    *,
    name_u: str = "u_mm",
    name_v: str = "v_mm",
) -> tuple[FloatArray, FloatArray]:
    """Validate matching local cross-sectional coordinates."""

    u = np.asarray(u_mm, dtype=float)
    v = np.asarray(v_mm, dtype=float)
    if u.shape != v.shape:
        raise ValueError(f"{name_u} and {name_v} must have matching shapes.")
    if not np.all(np.isfinite(u)):
        raise ValueError(f"{name_u} contains non-finite values.")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name_v} contains non-finite values.")
    return u, v


def validate_longitudinal_coordinates(
    longitudinal_mm: ArrayLike,
    *,
    shape: tuple[int, ...],
    name: str = "longitudinal_mm",
) -> FloatArray:
    """Validate longitudinal coordinates matching a cross-sectional evaluation."""

    longitudinal = np.asarray(longitudinal_mm, dtype=float)
    if longitudinal.shape != shape:
        raise ValueError(f"{name} must match local-coordinate shape.")
    if not np.all(np.isfinite(longitudinal)):
        raise ValueError(f"{name} contains non-finite values.")
    return longitudinal


@dataclass(frozen=True)
class CrossSectionEvaluation:
    """Evaluation of a cross-section at local coordinates."""

    u_mm: ArrayLike
    v_mm: ArrayLike
    rho: ArrayLike
    radial_distance_mm: ArrayLike
    inside: ArrayLike
    radius_mm: ArrayLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        u, v = validate_local_coordinates(self.u_mm, self.v_mm)
        rho = np.asarray(self.rho, dtype=float)
        radial = np.asarray(self.radial_distance_mm, dtype=float)
        inside = np.asarray(self.inside, dtype=bool)

        if rho.shape != u.shape:
            raise ValueError("rho must match local-coordinate shape.")
        if radial.shape != u.shape:
            raise ValueError("radial_distance_mm must match local-coordinate shape.")
        if inside.shape != u.shape:
            raise ValueError("inside must match local-coordinate shape.")
        if not np.all(np.isfinite(rho)):
            raise ValueError("rho contains non-finite values.")
        if not np.all(np.isfinite(radial)):
            raise ValueError("radial_distance_mm contains non-finite values.")
        if np.any(rho < 0):
            raise ValueError("rho must be non-negative.")
        if np.any(radial < 0):
            raise ValueError("radial_distance_mm must be non-negative.")

        radius = None
        if self.radius_mm is not None:
            radius = np.asarray(self.radius_mm, dtype=float)
            if radius.shape != u.shape:
                raise ValueError("radius_mm must match local-coordinate shape.")
            if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
                raise ValueError("radius_mm must contain finite positive values.")

        object.__setattr__(self, "u_mm", u)
        object.__setattr__(self, "v_mm", v)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "radial_distance_mm", radial)
        object.__setattr__(self, "inside", inside)
        object.__setattr__(self, "radius_mm", radius)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the evaluated coordinate arrays."""

        return tuple(np.asarray(self.rho).shape)

    @property
    def n_points(self) -> int:
        """Number of evaluated points."""

        return int(np.asarray(self.rho).size)

    @property
    def inside_fraction(self) -> float:
        """Fraction of evaluated points inside the cross-section."""

        if self.n_points == 0:
            return float("nan")
        return float(np.mean(self.inside))

    def to_dataframe(self):
        """Return flattened evaluation values as a table."""

        import pandas as pd

        data = {
            "u_mm": np.ravel(self.u_mm),
            "v_mm": np.ravel(self.v_mm),
            "rho": np.ravel(self.rho),
            "radial_distance_mm": np.ravel(self.radial_distance_mm),
            "inside": np.ravel(self.inside),
        }
        if self.radius_mm is not None:
            data["radius_mm"] = np.ravel(self.radius_mm)
        return pd.DataFrame(data)


@runtime_checkable
class CrossSection(Protocol):
    """Protocol for local cross-section models."""

    @property
    def kind(self) -> str:
        """Cross-section kind name."""

    @property
    def area_mm2(self) -> float:
        """Analytic or representative cross-sectional area."""

    def evaluate(
        self,
        u_mm: ArrayLike,
        v_mm: ArrayLike,
        *,
        longitudinal_mm: ArrayLike | None = None,
    ) -> CrossSectionEvaluation:
        """Evaluate the cross-section at local coordinates."""

    def boundary_radius_mm(self, angle_radians: ArrayLike) -> FloatArray:
        """Return boundary radius for directions in the local u-v plane."""

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""
