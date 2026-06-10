"""Base containers and helpers for analytic scalar profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.coordinates import BoolArray, FloatArray


def validate_finite_float(value: float, *, name: str) -> float:
    """Validate a finite floating-point value."""

    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def validate_positive_float(value: float, *, name: str) -> float:
    """Validate a finite positive floating-point value."""

    out = validate_finite_float(value, name=name)
    if out <= 0:
        raise ValueError(f"{name} must be positive.")
    return out


def validate_fraction(
    value: float,
    *,
    name: str,
    lower_open: bool = True,
    upper_closed: bool = True,
) -> float:
    """Validate a scalar fraction in a configurable interval near [0, 1]."""

    out = validate_finite_float(value, name=name)
    lower_ok = out > 0.0 if lower_open else out >= 0.0
    upper_ok = out <= 1.0 if upper_closed else out < 1.0
    if not (lower_ok and upper_ok):
        left = "(" if lower_open else "["
        right = "]" if upper_closed else ")"
        raise ValueError(f"{name} must lie in {left}0, 1{right}.")
    return out


def validate_profile_inputs(
    *,
    rho: ArrayLike,
    radial_distance_mm: ArrayLike,
    inside: ArrayLike | None = None,
    signed_u_mm: ArrayLike | None = None,
    longitudinal_mm: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray | None, FloatArray | None]:
    """Validate common profile-evaluation inputs."""

    rho_arr = np.asarray(rho, dtype=float)
    radial_arr = np.asarray(radial_distance_mm, dtype=float)
    if rho_arr.shape != radial_arr.shape:
        raise ValueError("rho and radial_distance_mm must have matching shapes.")
    if not np.all(np.isfinite(rho_arr)):
        raise ValueError("rho contains non-finite values.")
    if not np.all(np.isfinite(radial_arr)):
        raise ValueError("radial_distance_mm contains non-finite values.")
    if np.any(rho_arr < 0):
        raise ValueError("rho must be non-negative.")
    if np.any(radial_arr < 0):
        raise ValueError("radial_distance_mm must be non-negative.")

    if inside is None:
        inside_arr = rho_arr <= 1.0
    else:
        inside_arr = np.asarray(inside, dtype=bool)
        if inside_arr.shape != rho_arr.shape:
            raise ValueError("inside must match rho shape.")

    signed_arr = None
    if signed_u_mm is not None:
        signed_arr = np.asarray(signed_u_mm, dtype=float)
        if signed_arr.shape != rho_arr.shape:
            raise ValueError("signed_u_mm must match rho shape.")
        if not np.all(np.isfinite(signed_arr)):
            raise ValueError("signed_u_mm contains non-finite values.")

    longitudinal_arr = None
    if longitudinal_mm is not None:
        longitudinal_arr = np.asarray(longitudinal_mm, dtype=float)
        if longitudinal_arr.shape != rho_arr.shape:
            raise ValueError("longitudinal_mm must match rho shape.")
        if not np.all(np.isfinite(longitudinal_arr)):
            raise ValueError("longitudinal_mm contains non-finite values.")

    return rho_arr, radial_arr, inside_arr, signed_arr, longitudinal_arr


@dataclass(frozen=True)
class ScalarProfileEvaluation:
    """Evaluated scalar-profile values."""

    values: ArrayLike
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Profile values contain non-finite values.")
        object.__setattr__(self, "values", values)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the evaluated profile."""

        return tuple(np.asarray(self.values).shape)


@runtime_checkable
class ScalarProfile(Protocol):
    """Protocol for analytic scalar profiles."""

    @property
    def kind(self) -> str:
        """Profile kind name."""

    @property
    def background_value(self) -> float:
        """Value outside the object support."""

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate scalar-profile values."""

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""
