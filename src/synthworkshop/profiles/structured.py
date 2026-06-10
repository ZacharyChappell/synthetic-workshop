"""Structured analytic scalar profiles beyond simple radial decay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.profiles.base import (
    ScalarProfileEvaluation,
    validate_finite_float,
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


def _validate_unit_fraction(
    value: float,
    *,
    name: str,
    lower_open: bool = True,
    upper_open: bool = False,
) -> float:
    """Validate a finite fraction near the unit interval."""

    out = validate_finite_float(value, name=name)
    lower_ok = out > 0.0 if lower_open else out >= 0.0
    upper_ok = out < 1.0 if upper_open else out <= 1.0
    if not lower_ok or not upper_ok:
        lower = "(0" if lower_open else "[0"
        upper = "1)" if upper_open else "1]"
        raise ValueError(f"{name} must lie in {lower}, {upper}.")
    return out


@dataclass(frozen=True)
class HollowCoreProfile:
    """Radial profile with a low-valued core and brighter surrounding shell.

    This is intended for analytic hollow-core or central-sparing validation
    cases. It is not a biophysical model. The profile is defined by piecewise
    linear interpolation over normalised radius ``rho``:

    ``core_value`` from rho 0 to ``core_radius_fraction``,
    rising to ``shell_value`` at ``shell_radius_fraction``,
    then falling or rising to ``edge_value`` at rho 1.
    """

    core_value: float = 0.1
    shell_value: float = 1.0
    edge_value: float = 0.2
    core_radius_fraction: float = 0.25
    shell_radius_fraction: float = 0.65
    background_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "core_value",
            validate_finite_float(self.core_value, name="core_value"),
        )
        object.__setattr__(
            self,
            "shell_value",
            validate_finite_float(self.shell_value, name="shell_value"),
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

        core_radius = _validate_unit_fraction(
            self.core_radius_fraction,
            name="core_radius_fraction",
            lower_open=True,
            upper_open=True,
        )
        shell_radius = _validate_unit_fraction(
            self.shell_radius_fraction,
            name="shell_radius_fraction",
            lower_open=True,
            upper_open=False,
        )
        if shell_radius <= core_radius:
            raise ValueError(
                "shell_radius_fraction must be greater than core_radius_fraction."
            )

        object.__setattr__(self, "core_radius_fraction", core_radius)
        object.__setattr__(self, "shell_radius_fraction", shell_radius)

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "hollow_core"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate the hollow-core radial profile."""

        rho_arr, _, inside_arr, _, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )

        scaled = np.clip(rho_arr, 0.0, 1.0)
        values = np.interp(
            scaled,
            [
                0.0,
                self.core_radius_fraction,
                self.shell_radius_fraction,
                1.0,
            ],
            [
                self.core_value,
                self.core_value,
                self.shell_value,
                self.edge_value,
            ],
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
            "core_value": self.core_value,
            "shell_value": self.shell_value,
            "edge_value": self.edge_value,
            "core_radius_fraction": self.core_radius_fraction,
            "shell_radius_fraction": self.shell_radius_fraction,
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class SigmoidBoundaryProfile:
    """Smooth radial transition from centre value to boundary value.

    This gives an analytic soft-boundary profile for testing edge localisation
    and boundary sampling. The transition midpoint is ``boundary_fraction`` and
    the transition steepness is controlled by ``width_fraction``.
    """

    centre_value: float = 1.0
    edge_value: float = 0.2
    boundary_fraction: float = 0.75
    width_fraction: float = 0.08
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
            "boundary_fraction",
            _validate_unit_fraction(
                self.boundary_fraction,
                name="boundary_fraction",
                lower_open=True,
                upper_open=True,
            ),
        )
        object.__setattr__(
            self,
            "width_fraction",
            _validate_unit_fraction(
                self.width_fraction,
                name="width_fraction",
                lower_open=True,
                upper_open=False,
            ),
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "sigmoid_boundary"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate the sigmoid-boundary profile."""

        rho_arr, _, inside_arr, _, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )

        scaled = np.clip(rho_arr, 0.0, 1.0)
        exponent = (scaled - self.boundary_fraction) / self.width_fraction
        sigmoid = 1.0 / (1.0 + np.exp(exponent))
        values = self.edge_value + (self.centre_value - self.edge_value) * sigmoid
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
            "boundary_fraction": self.boundary_fraction,
            "width_fraction": self.width_fraction,
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class LongitudinalGradientProfile:
    """Linear scalar gradient along longitudinal object coordinate.

    The profile varies from ``start_value`` at longitudinal position 0 to
    ``end_value`` at ``length_mm``. Values outside this range are clipped. This
    is useful for known longitudinal-effect and profile-sampling validation.
    """

    start_value: float = 0.2
    end_value: float = 1.0
    length_mm: float = 1.0
    background_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_value",
            validate_finite_float(self.start_value, name="start_value"),
        )
        object.__setattr__(
            self,
            "end_value",
            validate_finite_float(self.end_value, name="end_value"),
        )
        object.__setattr__(
            self,
            "length_mm",
            validate_positive_float(self.length_mm, name="length_mm"),
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "longitudinal_gradient"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate the longitudinal gradient profile."""

        _, _, inside_arr, _, longitudinal_arr = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        if longitudinal_arr is None:
            raise ValueError(
                "longitudinal_mm is required for longitudinal_gradient profiles."
            )

        fraction = np.clip(longitudinal_arr / self.length_mm, 0.0, 1.0)
        values = self.start_value + (self.end_value - self.start_value) * fraction
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
            "start_value": self.start_value,
            "end_value": self.end_value,
            "length_mm": self.length_mm,
            "background_value": self.background_value,
        }


def _as_float_tuple(values: ArrayLike, *, name: str) -> tuple[float, ...]:
    """Validate a finite 1D float sequence."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return tuple(float(value) for value in array)


def _as_positive_float_tuple(values: ArrayLike, *, name: str) -> tuple[float, ...]:
    """Validate a finite positive 1D float sequence."""

    out = _as_float_tuple(values, name=name)
    if any(value <= 0.0 for value in out):
        raise ValueError(f"{name} must contain positive values.")
    return out


@dataclass(frozen=True)
class MultiPeakRadialProfile:
    """Radial profile formed from one or more Gaussian radial peaks.

    Peaks are specified in normalised radial coordinate ``rho``. This is useful
    for testing methods against profiles that have more than one local maximum
    across the cross-section.
    """

    base_value: float = 0.0
    peak_centres_fraction: tuple[float, ...] = (0.35, 0.75)
    peak_amplitudes: tuple[float, ...] = (1.0, 0.5)
    peak_widths_fraction: tuple[float, ...] = (0.08, 0.10)
    background_value: float = 0.0

    def __post_init__(self) -> None:
        centres = _as_float_tuple(
            self.peak_centres_fraction,
            name="peak_centres_fraction",
        )
        amplitudes = _as_float_tuple(self.peak_amplitudes, name="peak_amplitudes")
        widths = _as_positive_float_tuple(
            self.peak_widths_fraction,
            name="peak_widths_fraction",
        )

        if not (len(centres) == len(amplitudes) == len(widths)):
            raise ValueError(
                "peak_centres_fraction, peak_amplitudes, and "
                "peak_widths_fraction must have matching lengths."
            )
        if any(centre < 0.0 or centre > 1.0 for centre in centres):
            raise ValueError("peak_centres_fraction values must lie in [0, 1].")

        object.__setattr__(
            self,
            "base_value",
            validate_finite_float(self.base_value, name="base_value"),
        )
        object.__setattr__(self, "peak_centres_fraction", centres)
        object.__setattr__(self, "peak_amplitudes", amplitudes)
        object.__setattr__(self, "peak_widths_fraction", widths)
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "multi_peak_radial"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate the multi-peak radial profile."""

        rho_arr, _, inside_arr, _, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )

        scaled = np.clip(rho_arr, 0.0, 1.0)
        values = np.full(scaled.shape, self.base_value, dtype=float)
        for centre, amplitude, width in zip(
            self.peak_centres_fraction,
            self.peak_amplitudes,
            self.peak_widths_fraction,
            strict=True,
        ):
            values += amplitude * np.exp(-0.5 * ((scaled - centre) / width) ** 2)

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
            "base_value": self.base_value,
            "peak_centres_fraction": list(self.peak_centres_fraction),
            "peak_amplitudes": list(self.peak_amplitudes),
            "peak_widths_fraction": list(self.peak_widths_fraction),
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class OneSidedLesionProfile:
    """One-sided localised scalar elevation or depression.

    The lesion is defined along the signed local ``u`` coordinate, so it can
    create left/right asymmetric lesion-like effects in a tube cross-section.
    """

    baseline_value: float = 0.2
    lesion_delta: float = 0.8
    lesion_side: str = "positive"
    lesion_centre_mm: float = 1.0
    lesion_width_mm: float = 1.0
    background_value: float = 0.0

    def __post_init__(self) -> None:
        side = str(self.lesion_side).strip().lower()
        if side not in {"positive", "negative"}:
            raise ValueError("lesion_side must be 'positive' or 'negative'.")

        object.__setattr__(
            self,
            "baseline_value",
            validate_finite_float(self.baseline_value, name="baseline_value"),
        )
        object.__setattr__(
            self,
            "lesion_delta",
            validate_finite_float(self.lesion_delta, name="lesion_delta"),
        )
        object.__setattr__(self, "lesion_side", side)
        object.__setattr__(
            self,
            "lesion_centre_mm",
            validate_finite_float(self.lesion_centre_mm, name="lesion_centre_mm"),
        )
        object.__setattr__(
            self,
            "lesion_width_mm",
            validate_positive_float(self.lesion_width_mm, name="lesion_width_mm"),
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "one_sided_lesion"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate the one-sided lesion-like profile."""

        _, _, inside_arr, signed_u_arr, _ = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        if signed_u_arr is None:
            raise ValueError("signed_u_mm is required for one_sided_lesion profiles.")

        coordinate = signed_u_arr if self.lesion_side == "positive" else -signed_u_arr
        side_mask = coordinate >= 0.0
        lesion = np.exp(
            -0.5 * ((coordinate - self.lesion_centre_mm) / self.lesion_width_mm) ** 2
        )

        values = self.baseline_value + self.lesion_delta * lesion * side_mask
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
            "baseline_value": self.baseline_value,
            "lesion_delta": self.lesion_delta,
            "lesion_side": self.lesion_side,
            "lesion_centre_mm": self.lesion_centre_mm,
            "lesion_width_mm": self.lesion_width_mm,
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class RadialLongitudinalGradientProfile:
    """Linear radial profile whose centre and edge values vary longitudinally."""

    centre_start_value: float = 1.0
    centre_end_value: float = 0.5
    edge_start_value: float = 0.2
    edge_end_value: float = 0.8
    length_mm: float = 1.0
    background_value: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "centre_start_value",
            "centre_end_value",
            "edge_start_value",
            "edge_end_value",
            "background_value",
        ):
            object.__setattr__(
                self,
                name,
                validate_finite_float(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "length_mm",
            validate_positive_float(self.length_mm, name="length_mm"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "radial_longitudinal_gradient"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate the radial-longitudinal gradient profile."""

        rho_arr, _, inside_arr, _, longitudinal_arr = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        if longitudinal_arr is None:
            raise ValueError(
                "longitudinal_mm is required for radial_longitudinal_gradient profiles."
            )

        fraction = np.clip(longitudinal_arr / self.length_mm, 0.0, 1.0)
        centre = (
            self.centre_start_value
            + (self.centre_end_value - self.centre_start_value) * fraction
        )
        edge = (
            self.edge_start_value
            + (self.edge_end_value - self.edge_start_value) * fraction
        )

        radial_fraction = np.clip(rho_arr, 0.0, 1.0)
        values = centre + (edge - centre) * radial_fraction
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
            "centre_start_value": self.centre_start_value,
            "centre_end_value": self.centre_end_value,
            "edge_start_value": self.edge_start_value,
            "edge_end_value": self.edge_end_value,
            "length_mm": self.length_mm,
            "background_value": self.background_value,
        }


@dataclass(frozen=True)
class PeriodicLongitudinalProfile:
    """Sinusoidal scalar modulation along longitudinal object coordinate."""

    baseline_value: float = 0.5
    amplitude: float = 0.2
    length_mm: float = 1.0
    periods: float = 1.0
    phase_radians: float = 0.0
    background_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_value",
            validate_finite_float(self.baseline_value, name="baseline_value"),
        )
        object.__setattr__(
            self,
            "amplitude",
            validate_finite_float(self.amplitude, name="amplitude"),
        )
        object.__setattr__(
            self,
            "length_mm",
            validate_positive_float(self.length_mm, name="length_mm"),
        )
        object.__setattr__(
            self,
            "periods",
            validate_positive_float(self.periods, name="periods"),
        )
        object.__setattr__(
            self,
            "phase_radians",
            validate_finite_float(self.phase_radians, name="phase_radians"),
        )
        object.__setattr__(
            self,
            "background_value",
            validate_finite_float(self.background_value, name="background_value"),
        )

    @property
    def kind(self) -> str:
        """Profile kind name."""

        return "periodic_longitudinal"

    def evaluate(
        self,
        *,
        rho: ArrayLike,
        radial_distance_mm: ArrayLike,
        inside: ArrayLike | None = None,
        signed_u_mm: ArrayLike | None = None,
        longitudinal_mm: ArrayLike | None = None,
    ) -> ScalarProfileEvaluation:
        """Evaluate periodic longitudinal modulation."""

        _, _, inside_arr, _, longitudinal_arr = validate_profile_inputs(
            rho=rho,
            radial_distance_mm=radial_distance_mm,
            inside=inside,
            signed_u_mm=signed_u_mm,
            longitudinal_mm=longitudinal_mm,
        )
        if longitudinal_arr is None:
            raise ValueError(
                "longitudinal_mm is required for periodic_longitudinal profiles."
            )

        angle = (
            2.0
            * np.pi
            * self.periods
            * np.clip(longitudinal_arr / self.length_mm, 0.0, 1.0)
            + self.phase_radians
        )
        values = self.baseline_value + self.amplitude * np.sin(angle)
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
            "baseline_value": self.baseline_value,
            "amplitude": self.amplitude,
            "length_mm": self.length_mm,
            "periods": self.periods,
            "phase_radians": self.phase_radians,
            "background_value": self.background_value,
        }
