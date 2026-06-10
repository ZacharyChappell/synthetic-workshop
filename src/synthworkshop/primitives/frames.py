"""Simple deterministic frame construction for sampled 3D centrelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.coordinates import FloatArray, normalise_vectors
from synthworkshop.primitives.curves import Centreline


@dataclass(frozen=True)
class ReferenceFrame:
    """Reference-guided local frame along a 3D centreline."""

    coordinates_mm: ArrayLike
    tangents: ArrayLike
    primary_axes: ArrayLike
    secondary_axes: ArrayLike
    valid: ArrayLike

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates_mm, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coordinates_mm must have shape (n_points, 3).")
        tangents = np.asarray(self.tangents, dtype=float)
        primary = np.asarray(self.primary_axes, dtype=float)
        secondary = np.asarray(self.secondary_axes, dtype=float)
        valid = np.asarray(self.valid, dtype=bool)
        if tangents.shape != coords.shape:
            raise ValueError("tangents must match coordinates_mm shape.")
        if primary.shape != coords.shape:
            raise ValueError("primary_axes must match coordinates_mm shape.")
        if secondary.shape != coords.shape:
            raise ValueError("secondary_axes must match coordinates_mm shape.")
        if valid.shape != (coords.shape[0],):
            raise ValueError("valid must contain one value per point.")
        tangents = normalise_vectors(tangents, axis=1, name="tangents")
        primary = normalise_vectors(primary, axis=1, name="primary_axes")
        secondary = normalise_vectors(secondary, axis=1, name="secondary_axes")
        object.__setattr__(self, "coordinates_mm", coords)
        object.__setattr__(self, "tangents", tangents)
        object.__setattr__(self, "primary_axes", primary)
        object.__setattr__(self, "secondary_axes", secondary)
        object.__setattr__(self, "valid", valid)

    @property
    def n_points(self) -> int:
        """Number of frame points."""

        return int(np.asarray(self.coordinates_mm).shape[0])

    def to_dataframe(self):
        """Return frame vectors as a tabular object."""

        import pandas as pd

        return pd.DataFrame(
            {
                "point_index": np.arange(self.n_points, dtype=int),
                "i_mm": self.coordinates_mm[:, 0],
                "j_mm": self.coordinates_mm[:, 1],
                "k_mm": self.coordinates_mm[:, 2],
                "tangent_i": self.tangents[:, 0],
                "tangent_j": self.tangents[:, 1],
                "tangent_k": self.tangents[:, 2],
                "primary_i": self.primary_axes[:, 0],
                "primary_j": self.primary_axes[:, 1],
                "primary_k": self.primary_axes[:, 2],
                "secondary_i": self.secondary_axes[:, 0],
                "secondary_j": self.secondary_axes[:, 1],
                "secondary_k": self.secondary_axes[:, 2],
                "valid": self.valid,
            }
        )


def _reference_axes_array(reference_axes: Sequence[Sequence[float]]) -> FloatArray:
    """Validate reference axes."""

    axes = np.asarray(reference_axes, dtype=float)
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError("reference_axes must have shape (n_axes, 3).")
    if axes.shape[0] < 1:
        raise ValueError("At least one reference axis is required.")
    return normalise_vectors(axes, axis=1, name="reference_axes")


def build_reference_guided_frame(
    centreline: Centreline,
    *,
    reference_axes: Sequence[Sequence[float]] = (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
    min_projection_norm: float = 1e-8,
) -> ReferenceFrame:
    """Build a deterministic orthonormal frame for a 3D centreline.

    The primary axis is selected by projecting candidate reference axes into the
    plane perpendicular to each tangent. The first sufficiently non-parallel
    reference axis is used.
    """

    if centreline.ndim != 3:
        raise ValueError("Reference-guided frames are currently defined for 3D curves.")
    if not np.isfinite(min_projection_norm) or min_projection_norm <= 0:
        raise ValueError("min_projection_norm must be finite and positive.")
    axes = _reference_axes_array(reference_axes)
    tangents = np.asarray(centreline.tangents, dtype=float)
    primary = np.zeros_like(tangents)
    secondary = np.zeros_like(tangents)
    valid = np.zeros(centreline.n_points, dtype=bool)
    for point_index, tangent in enumerate(tangents):
        chosen = None
        for axis in axes:
            projected = axis - np.dot(axis, tangent) * tangent
            norm = float(np.linalg.norm(projected))
            if norm > min_projection_norm:
                chosen = projected / norm
                break
        if chosen is None:
            continue
        primary[point_index] = chosen
        secondary[point_index] = np.cross(tangent, chosen)
        valid[point_index] = True
    if not np.all(valid):
        raise ValueError("Could not build a valid frame for every centreline point.")
    secondary = normalise_vectors(secondary, axis=1, name="secondary_axes")
    return ReferenceFrame(
        coordinates_mm=centreline.coordinates_mm,
        tangents=tangents,
        primary_axes=primary,
        secondary_axes=secondary,
        valid=valid,
    )
