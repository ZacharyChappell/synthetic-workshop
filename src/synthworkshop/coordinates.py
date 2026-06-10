"""Coordinate, shape, spacing, and vector validation helpers."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


def validate_shape(
    shape: Sequence[int],
    *,
    ndim: int | None = None,
    name: str = "shape",
) -> tuple[int, ...]:
    """Validate and normalise an image/grid shape."""
    values = tuple(int(value) for value in shape)
    if ndim is not None and len(values) != ndim:
        raise ValueError(f"{name} must contain {ndim} values.")
    if len(values) not in {2, 3}:
        raise ValueError(f"{name} must describe a 2D or 3D grid.")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive integers.")
    return values


def validate_spacing(
    spacing: Sequence[float],
    *,
    ndim: int | None = None,
    name: str = "spacing",
) -> tuple[float, ...]:
    """Validate and normalise physical voxel spacing."""
    values = tuple(float(value) for value in spacing)
    if ndim is not None and len(values) != ndim:
        raise ValueError(f"{name} must contain {ndim} values.")
    if len(values) not in {2, 3}:
        raise ValueError(f"{name} must describe a 2D or 3D grid.")
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} values must be finite and positive.")
    return values


def validate_origin(
    origin: Sequence[float],
    *,
    ndim: int,
    name: str = "origin",
) -> tuple[float, ...]:
    """Validate and normalise the physical origin."""
    values = tuple(float(value) for value in origin)
    if len(values) != ndim:
        raise ValueError(f"{name} must contain {ndim} values.")
    if any(not np.isfinite(value) for value in values):
        raise ValueError(f"{name} values must be finite.")
    return values


def validate_axis_names(
    axis_names: Sequence[str],
    *,
    ndim: int,
    name: str = "axis_names",
) -> tuple[str, ...]:
    """Validate axis names for a grid."""
    values = tuple(str(value) for value in axis_names)
    if len(values) != ndim:
        raise ValueError(f"{name} must contain {ndim} values.")
    if any(not value for value in values):
        raise ValueError(f"{name} values must be non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} values must be unique.")
    return values


def as_coordinate_array(
    coordinates: ArrayLike,
    *,
    ndim: int,
    name: str = "coordinates",
) -> FloatArray:
    """Return an array whose final dimension contains coordinate components."""
    arr = np.asarray(coordinates, dtype=float)
    if arr.shape == (ndim,):
        return arr
    if arr.ndim < 2 or arr.shape[-1] != ndim:
        raise ValueError(f"{name} must have final dimension {ndim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def normalise_vectors(
    vectors: ArrayLike,
    *,
    axis: int = -1,
    eps: float = 1e-12,
    name: str = "vectors",
) -> FloatArray:
    """Return unit vectors, rejecting zero-length or non-finite vectors."""
    arr = np.asarray(vectors, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError(f"{name} contains zero-length vectors.")
    return arr / norms


def validate_array_shape(
    array: ArrayLike,
    *,
    shape: tuple[int, ...],
    name: str,
) -> NDArray:
    """Validate that an array has the expected shape."""
    arr = np.asarray(array)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}.")
    return arr
