"""Simple analytic scalar profiles for local tube coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.profiles.base import (
    ScalarProfileEvaluation,
    validate_finite_float,
    validate_fraction,
    validate_positive_float,
    validate_profile_inputs,
)


def _apply_background(
    values: np.ndarray,
    *,
    inside: np.ndarray,
    background_value: float,
) -> np.ndarray:
    """Apply outside-object background values."""

    return np.where(inside, values, background_value)


@dataclass(frozen=True)
class ConstantProfile:
    """Constant scalar value inside the object."""

    value: float = 1.0
    background_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value", validate_finite_float(self.value, name="value")
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "constant"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate a constant profile."""

        rho_arr, _, inside_arr, _, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        values = np.full(rho_arr.shape, self.value, dtype=float)
        values = _apply_background(
            values,
            inside=inside_arr,
            background_value=self.background_value,
        )
        return ScalarProfileEvaluation(values=values, metadata=self.summary())

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "value": self.value,
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class LinearRadialProfile:
    """Linear radial profile from centre to boundary."""

    centre_value: float = 1.0
    edge_value: float = 0.2
    background_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "centre_value",
            validate_finite_float(self.centre_value, name="centre_value"),
        )
        object.__setattr__(
            self,
            "edge_value",
            validate_finite_float(self.edge_value, name="edge_value"),
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "linear_radial"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate a linear radial profile."""

        rho_arr, _, inside_arr, _, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        scaled = np.clip(rho_arr, 0.0, 1.0)
        values = self.centre_value + (self.edge_value - self.centre_value) * scaled
        values = _apply_background(
            values,
            inside=inside_arr,
            background_value=self.background_value,
        )
        return ScalarProfileEvaluation(values=values, metadata=self.summary())

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "centre_value": self.centre_value,
            "edge_value": self.edge_value,
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class GaussianRadialProfile:
    """Gaussian centre-bright radial profile."""

    centre_value: float = 1.0
    edge_value: float = 0.2
    sigma_fraction: float = 0.45
    background_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "centre_value",
            validate_finite_float(self.centre_value, name="centre_value"),
        )
        object.__setattr__(
            self,
            "edge_value",
            validate_finite_float(self.edge_value, name="edge_value"),
        )
        object.__setattr__(
            self,
            "sigma_fraction",
            validate_positive_float(self.sigma_fraction, name="sigma_fraction"),
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "gaussian_radial"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate a Gaussian radial profile."""

        rho_arr, _, inside_arr, _, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        values = self.edge_value + (self.centre_value - self.edge_value) * np.exp(
            -0.5 * (rho_arr / self.sigma_fraction) ** 2
        )
        values = _apply_background(
            values,
            inside=inside_arr,
            background_value=self.background_value,
        )
        return ScalarProfileEvaluation(values=values, metadata=self.summary())

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "centre_value": self.centre_value,
            "edge_value": self.edge_value,
            "sigma_fraction": self.sigma_fraction,
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class EdgeEnhancedProfile:
    """Profile with an enhanced rim near the object boundary."""

    centre_value: float = 0.2
    edge_value: float = 1.0
    edge_width_fraction: float = 0.15
    background_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "centre_value",
            validate_finite_float(self.centre_value, name="centre_value"),
        )
        object.__setattr__(
            self,
            "edge_value",
            validate_finite_float(self.edge_value, name="edge_value"),
        )
        object.__setattr__(
            self,
            "edge_width_fraction",
            validate_fraction(self.edge_width_fraction, name="edge_width_fraction"),
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "edge_enhanced"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate an edge-enhanced radial profile."""

        rho_arr, _, inside_arr, _, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        rim = np.exp(-0.5 * ((rho_arr - 1.0) / self.edge_width_fraction) ** 2)
        values = self.centre_value + (self.edge_value - self.centre_value) * rim
        values = _apply_background(
            values,
            inside=inside_arr,
            background_value=self.background_value,
        )
        return ScalarProfileEvaluation(values=values, metadata=self.summary())

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "centre_value": self.centre_value,
            "edge_value": self.edge_value,
            "edge_width_fraction": self.edge_width_fraction,
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class AsymmetricLinearProfile:
    """Linear radial profile with a signed left/right modulation."""

    centre_value: float = 1.0
    edge_value: float = 0.2
    asymmetry: float = 0.1
    background_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "centre_value",
            validate_finite_float(self.centre_value, name="centre_value"),
        )
        object.__setattr__(
            self,
            "edge_value",
            validate_finite_float(self.edge_value, name="edge_value"),
        )
        object.__setattr__(
            self,
            "asymmetry",
            validate_finite_float(self.asymmetry, name="asymmetry"),
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "asymmetric_linear"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate an asymmetric linear profile.

        The asymmetry is expressed along the local signed u-axis. It is zero at
        the centre and largest near the cross-sectional boundary.
        """

        rho_arr, _, inside_arr, signed_u_arr, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        if signed_u_arr is None:
            raise ValueError("signed_u_mm is required for asymmetric_linear profiles.")

        scaled = np.clip(rho_arr, 0.0, 1.0)
        base = self.centre_value + (self.edge_value - self.centre_value) * scaled
        side = np.sign(signed_u_arr)
        values = base + self.asymmetry * side * scaled
        values = _apply_background(
            values,
            inside=inside_arr,
            background_value=self.background_value,
        )
        return ScalarProfileEvaluation(values=values, metadata=self.summary())

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary."""

        return {
            "kind": self.kind,
            "centre_value": self.centre_value,
            "edge_value": self.edge_value,
            "asymmetry": self.asymmetry,
            "background_value": self.background_value,
        }
