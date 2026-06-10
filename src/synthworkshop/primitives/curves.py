"""Analytic curves and sampled centreline containers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.coordinates import FloatArray, normalise_vectors


def _validate_step_mm(step_mm: float) -> float:
    """Validate a positive sampling step."""

    value = float(step_mm)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("step_mm must be finite and positive.")
    return value


def _validate_parameters(parameters: ArrayLike) -> FloatArray:
    """Validate curve parameters in the closed interval [0, 1]."""

    t = np.asarray(parameters, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("Curve parameters contain non-finite values.")
    if np.any((t < 0.0) | (t > 1.0)):
        raise ValueError("Curve parameters must lie in [0, 1].")
    return t


def _as_point(value: Sequence[float], *, name: str) -> FloatArray:
    """Validate a 2D or 3D point."""

    point = np.asarray(value, dtype=float)
    if point.ndim != 1 or point.shape[0] not in {2, 3}:
        raise ValueError(f"{name} must be a 2D or 3D coordinate.")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} contains non-finite values.")
    return point


def _sample_parameters_by_step(
    *,
    length_mm: float,
    step_mm: float,
    n_samples: int | None,
) -> FloatArray:
    """Return monotonically increasing parameters for curve sampling."""

    step = _validate_step_mm(step_mm)
    if n_samples is not None:
        n = int(n_samples)
        if n < 2:
            raise ValueError("n_samples must be at least 2.")
        return np.linspace(0.0, 1.0, n)
    if not np.isfinite(length_mm) or length_mm <= 0:
        raise ValueError("length_mm must be finite and positive.")
    n = max(2, int(np.ceil(length_mm / step)) + 1)
    return np.linspace(0.0, 1.0, n)


def cumulative_arclength_mm(points_mm: ArrayLike) -> FloatArray:
    """Compute cumulative arclength along sampled points."""

    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("points_mm must have shape (n_points, ndim).")
    if points.shape[0] < 2:
        raise ValueError("At least two points are required.")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_mm contains non-finite values.")
    deltas = np.diff(points, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(distances)])


@dataclass(frozen=True)
class Centreline:
    """Sampled analytic centreline in physical coordinates."""

    coordinates_mm: ArrayLike
    tangents: ArrayLike
    arclength_mm: ArrayLike
    parameters: ArrayLike
    object_id: str | None = None
    segment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates_mm, dtype=float)
        if coords.ndim != 2 or coords.shape[1] not in {2, 3}:
            raise ValueError("coordinates_mm must have shape (n_points, ndim).")
        if coords.shape[0] < 2:
            raise ValueError("A centreline must contain at least two points.")
        if not np.all(np.isfinite(coords)):
            raise ValueError("coordinates_mm contains non-finite values.")
        tangents = np.asarray(self.tangents, dtype=float)
        if tangents.shape != coords.shape:
            raise ValueError("tangents must have the same shape as coordinates_mm.")
        tangents = normalise_vectors(tangents, axis=1, name="tangents")
        arclength = np.asarray(self.arclength_mm, dtype=float)
        if arclength.shape != (coords.shape[0],):
            raise ValueError("arclength_mm must have one value per point.")
        if not np.all(np.isfinite(arclength)):
            raise ValueError("arclength_mm contains non-finite values.")
        if not np.isclose(arclength[0], 0.0):
            raise ValueError("arclength_mm must start at 0.")
        if np.any(np.diff(arclength) < -1e-10):
            raise ValueError("arclength_mm must be monotonically increasing.")
        parameters = np.asarray(self.parameters, dtype=float)
        if parameters.shape != (coords.shape[0],):
            raise ValueError("parameters must have one value per point.")
        _validate_parameters(parameters)
        object.__setattr__(self, "coordinates_mm", coords)
        object.__setattr__(self, "tangents", tangents)
        object.__setattr__(self, "arclength_mm", arclength)
        object.__setattr__(self, "parameters", parameters)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return int(self.coordinates_mm.shape[1])

    @property
    def n_points(self) -> int:
        """Number of sampled centreline points."""

        return int(self.coordinates_mm.shape[0])

    @property
    def length_mm(self) -> float:
        """Sampled centreline length."""

        return float(self.arclength_mm[-1])

    def to_dataframe(self):
        """Return the centreline as a tabular object."""

        import pandas as pd

        axis_names = ("i", "j") if self.ndim == 2 else ("i", "j", "k")
        data: dict[str, object] = {
            "point_index": np.arange(self.n_points, dtype=int),
            "parameter": self.parameters,
            "arclength_mm": self.arclength_mm,
        }
        if self.object_id is not None:
            data["object_id"] = self.object_id
        if self.segment_id is not None:
            data["segment_id"] = self.segment_id
        for axis, name in enumerate(axis_names):
            data[f"{name}_mm"] = self.coordinates_mm[:, axis]
            data[f"tangent_{name}"] = self.tangents[:, axis]
        return pd.DataFrame(data)


@dataclass(frozen=True)
class LineCurve:
    """Straight line segment between two physical points."""

    start_mm: Sequence[float]
    end_mm: Sequence[float]

    def __post_init__(self) -> None:
        start = _as_point(self.start_mm, name="start_mm")
        end = _as_point(self.end_mm, name="end_mm")
        if start.shape != end.shape:
            raise ValueError("start_mm and end_mm must have matching dimensions.")
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 0:
            raise ValueError("LineCurve start_mm and end_mm must differ.")
        object.__setattr__(self, "start_mm", start)
        object.__setattr__(self, "end_mm", end)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return int(np.asarray(self.start_mm).shape[0])

    @property
    def length_mm(self) -> float:
        """Exact line length."""

        return float(
            np.linalg.norm(np.asarray(self.end_mm) - np.asarray(self.start_mm))
        )

    def evaluate(self, parameters: ArrayLike) -> FloatArray:
        """Evaluate physical coordinates along the line."""

        t = _validate_parameters(parameters)
        start = np.asarray(self.start_mm, dtype=float)
        end = np.asarray(self.end_mm, dtype=float)
        return start + np.expand_dims(t, axis=-1) * (end - start)

    def tangent(self, parameters: ArrayLike) -> FloatArray:
        """Evaluate unit tangent vectors along the line."""

        t = _validate_parameters(parameters)
        start = np.asarray(self.start_mm, dtype=float)
        end = np.asarray(self.end_mm, dtype=float)
        tangent = (end - start) / self.length_mm
        return np.broadcast_to(tangent, (*t.shape, self.ndim)).copy()

    def sample(
        self,
        *,
        step_mm: float = 1.0,
        n_samples: int | None = None,
        object_id: str | None = None,
        segment_id: str | None = None,
    ) -> Centreline:
        """Sample the line as a Centreline."""

        parameters = _sample_parameters_by_step(
            length_mm=self.length_mm,
            step_mm=step_mm,
            n_samples=n_samples,
        )
        points = self.evaluate(parameters)
        tangents = self.tangent(parameters)
        arclength = parameters * self.length_mm
        return Centreline(
            coordinates_mm=points,
            tangents=tangents,
            arclength_mm=arclength,
            parameters=parameters,
            object_id=object_id,
            segment_id=segment_id,
            metadata={"curve_kind": "line"},
        )


@dataclass(frozen=True)
class SinusoidalCurve:
    """Endpoint-preserving sinusoidal curve between two physical points."""

    start_mm: Sequence[float]
    end_mm: Sequence[float]
    amplitude_mm: Sequence[float]
    periods: float = 1.0
    phase_radians: float = 0.0

    def __post_init__(self) -> None:
        start = _as_point(self.start_mm, name="start_mm")
        end = _as_point(self.end_mm, name="end_mm")
        amplitude = _as_point(self.amplitude_mm, name="amplitude_mm")
        if start.shape != end.shape or start.shape != amplitude.shape:
            raise ValueError("start_mm, end_mm, and amplitude_mm must match.")
        if np.linalg.norm(end - start) <= 0:
            raise ValueError("SinusoidalCurve start_mm and end_mm must differ.")
        periods = float(self.periods)
        phase = float(self.phase_radians)
        if not np.isfinite(periods) or periods <= 0:
            raise ValueError("periods must be finite and positive.")
        if not np.isfinite(phase):
            raise ValueError("phase_radians must be finite.")
        object.__setattr__(self, "start_mm", start)
        object.__setattr__(self, "end_mm", end)
        object.__setattr__(self, "amplitude_mm", amplitude)
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "phase_radians", phase)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return int(np.asarray(self.start_mm).shape[0])

    def _wave(self, parameters: FloatArray) -> FloatArray:
        """Endpoint-preserving sinusoidal displacement weight."""

        phase = 2.0 * np.pi * self.periods * parameters + self.phase_radians
        return np.sin(np.pi * parameters) * np.sin(phase)

    def _wave_derivative(self, parameters: FloatArray) -> FloatArray:
        """Derivative of the sinusoidal displacement weight."""

        phase = 2.0 * np.pi * self.periods * parameters + self.phase_radians
        return np.pi * np.cos(np.pi * parameters) * np.sin(
            phase
        ) + 2.0 * np.pi * self.periods * np.sin(np.pi * parameters) * np.cos(phase)

    def evaluate(self, parameters: ArrayLike) -> FloatArray:
        """Evaluate physical coordinates along the sinusoidal curve."""

        t = _validate_parameters(parameters)
        start = np.asarray(self.start_mm, dtype=float)
        end = np.asarray(self.end_mm, dtype=float)
        amplitude = np.asarray(self.amplitude_mm, dtype=float)
        baseline = start + np.expand_dims(t, axis=-1) * (end - start)
        displacement = np.expand_dims(self._wave(t), axis=-1) * amplitude
        return baseline + displacement

    def tangent(self, parameters: ArrayLike) -> FloatArray:
        """Evaluate unit tangent vectors along the sinusoidal curve."""

        t = _validate_parameters(parameters)
        start = np.asarray(self.start_mm, dtype=float)
        end = np.asarray(self.end_mm, dtype=float)
        amplitude = np.asarray(self.amplitude_mm, dtype=float)
        derivative = (end - start) + np.expand_dims(
            self._wave_derivative(t),
            axis=-1,
        ) * amplitude
        return normalise_vectors(derivative, axis=-1, name="curve derivative")

    def estimate_length_mm(self, *, n_samples: int = 4096) -> float:
        """Estimate curve length by dense sampling."""

        if n_samples < 2:
            raise ValueError("n_samples must be at least 2.")
        parameters = np.linspace(0.0, 1.0, int(n_samples))
        points = self.evaluate(parameters)
        return float(cumulative_arclength_mm(points)[-1])

    def sample(
        self,
        *,
        step_mm: float = 1.0,
        n_samples: int | None = None,
        object_id: str | None = None,
        segment_id: str | None = None,
    ) -> Centreline:
        """Sample the sinusoidal curve as a Centreline."""

        estimated_length = self.estimate_length_mm()
        parameters = _sample_parameters_by_step(
            length_mm=estimated_length,
            step_mm=step_mm,
            n_samples=n_samples,
        )
        points = self.evaluate(parameters)
        tangents = self.tangent(parameters)
        arclength = cumulative_arclength_mm(points)
        return Centreline(
            coordinates_mm=points,
            tangents=tangents,
            arclength_mm=arclength,
            parameters=parameters,
            object_id=object_id,
            segment_id=segment_id,
            metadata={
                "curve_kind": "sinusoidal",
                "periods": self.periods,
                "phase_radians": self.phase_radians,
            },
        )


@dataclass(frozen=True)
class PolylineCurve:
    """Piecewise-linear curve through two or more physical vertices."""

    vertices_mm: ArrayLike

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_mm, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] not in {2, 3}:
            raise ValueError("vertices_mm must have shape (n_vertices, ndim).")
        if vertices.shape[0] < 2:
            raise ValueError("PolylineCurve requires at least two vertices.")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices_mm contains non-finite values.")

        deltas = np.diff(vertices, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("PolylineCurve contains zero-length segment(s).")

        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

        object.__setattr__(self, "vertices_mm", vertices)
        object.__setattr__(self, "_segment_vectors", deltas)
        object.__setattr__(self, "_segment_lengths", lengths)
        object.__setattr__(self, "_cumulative_lengths", cumulative)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return int(np.asarray(self.vertices_mm).shape[1])

    @property
    def n_vertices(self) -> int:
        """Number of polyline vertices."""

        return int(np.asarray(self.vertices_mm).shape[0])

    @property
    def n_segments(self) -> int:
        """Number of piecewise-linear segments."""

        return self.n_vertices - 1

    @property
    def length_mm(self) -> float:
        """Exact polyline arclength."""

        return float(self._cumulative_lengths[-1])

    @property
    def segment_lengths_mm(self) -> FloatArray:
        """Piecewise segment lengths."""

        return np.asarray(self._segment_lengths, dtype=float)

    def _segment_indices_and_fractions(
        self,
        parameters: ArrayLike,
    ) -> tuple[FloatArray, np.ndarray, FloatArray]:
        """Map curve parameters to segment indices and local fractions."""

        t = _validate_parameters(parameters)
        flat_t = np.ravel(t)
        target_lengths = flat_t * self.length_mm

        segment_indices = np.searchsorted(
            self._cumulative_lengths[1:],
            target_lengths,
            side="left",
        )
        segment_indices = np.clip(segment_indices, 0, self.n_segments - 1)

        segment_start = self._cumulative_lengths[segment_indices]
        segment_length = self._segment_lengths[segment_indices]
        fractions = (target_lengths - segment_start) / segment_length
        fractions = np.clip(fractions, 0.0, 1.0)

        return t, segment_indices, fractions

    def evaluate(self, parameters: ArrayLike) -> FloatArray:
        """Evaluate physical coordinates along the polyline."""

        t, segment_indices, fractions = self._segment_indices_and_fractions(parameters)
        vertices = np.asarray(self.vertices_mm, dtype=float)
        vectors = np.asarray(self._segment_vectors, dtype=float)

        points = (
            vertices[segment_indices] + fractions[:, None] * vectors[segment_indices]
        )
        return points.reshape((*t.shape, self.ndim))

    def tangent(self, parameters: ArrayLike) -> FloatArray:
        """Evaluate unit tangent vectors along the polyline."""

        t, segment_indices, _ = self._segment_indices_and_fractions(parameters)
        vectors = np.asarray(self._segment_vectors, dtype=float)
        lengths = np.asarray(self._segment_lengths, dtype=float)
        tangents = vectors[segment_indices] / lengths[segment_indices, None]
        return tangents.reshape((*t.shape, self.ndim))

    def sample(
        self,
        *,
        step_mm: float = 1.0,
        n_samples: int | None = None,
        object_id: str | None = None,
        segment_id: str | None = None,
    ) -> Centreline:
        """Sample the polyline as a Centreline.

        For step-based sampling, original polyline vertices are always retained.
        This prevents sampled centrelines from cutting across corners and keeps
        arclength tied to the true piecewise-linear path.
        """

        parameters = _sample_parameters_by_step(
            length_mm=self.length_mm,
            step_mm=step_mm,
            n_samples=n_samples,
        )
        if n_samples is None:
            vertex_parameters = self._cumulative_lengths / self.length_mm
            parameters = np.unique(np.concatenate([parameters, vertex_parameters]))

        points = self.evaluate(parameters)
        tangents = self.tangent(parameters)
        arclength = parameters * self.length_mm

        return Centreline(
            coordinates_mm=points,
            tangents=tangents,
            arclength_mm=arclength,
            parameters=parameters,
            object_id=object_id,
            segment_id=segment_id,
            metadata={
                "curve_kind": "polyline",
                "n_vertices": self.n_vertices,
                "n_segments": self.n_segments,
            },
        )
