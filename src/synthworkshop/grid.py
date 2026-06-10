"""Grid and physical-coordinate definitions."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from synthworkshop.coordinates import (
    FloatArray,
    as_coordinate_array,
    validate_axis_names,
    validate_origin,
    validate_shape,
    validate_spacing,
)


@dataclass(frozen=True)
class GridSpec:
    """Regular 2D/3D grid with physical spacing and origin."""

    shape: Sequence[int]
    spacing: Sequence[float]
    origin: Sequence[float] | None = None
    axis_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        shape = validate_shape(self.shape)
        spacing = validate_spacing(self.spacing, ndim=len(shape))
        origin = (
            tuple(0.0 for _ in shape)
            if self.origin is None
            else validate_origin(self.origin, ndim=len(shape))
        )
        axis_names = (
            (("i", "j") if len(shape) == 2 else ("i", "j", "k"))
            if self.axis_names is None
            else validate_axis_names(
                self.axis_names,
                ndim=len(shape),
            )
        )

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axis_names", axis_names)

    @property
    def ndim(self) -> int:
        """Number of grid dimensions."""

        return len(self.shape)

    @property
    def n_voxels(self) -> int:
        """Total number of voxels/pixels."""

        return int(np.prod(self.shape))

    @property
    def spacing_array(self) -> FloatArray:
        """Spacing as a NumPy array."""

        return np.asarray(self.spacing, dtype=float)

    @property
    def origin_array(self) -> FloatArray:
        """Origin as a NumPy array."""

        return np.asarray(self.origin, dtype=float)

    @property
    def physical_extent(self) -> tuple[float, ...]:
        """Physical distance from first to last sample along each axis."""

        return tuple(
            (size - 1) * spacing
            for size, spacing in zip(self.shape, self.spacing, strict=True)
        )

    @property
    def physical_size(self) -> tuple[float, ...]:
        """Physical voxel-covered size along each axis."""

        return tuple(
            size * spacing
            for size, spacing in zip(self.shape, self.spacing, strict=True)
        )

    def index_arrays(self) -> tuple[FloatArray, ...]:
        """Return dense index-coordinate arrays."""

        arrays = np.indices(self.shape, dtype=float)
        return tuple(arrays[axis] for axis in range(self.ndim))

    def world_arrays(self) -> tuple[FloatArray, ...]:
        """Return dense physical-coordinate arrays."""

        return tuple(
            self.origin[axis] + self.spacing[axis] * index_array
            for axis, index_array in enumerate(self.index_arrays())
        )

    def index_to_world(self, coordinates: object) -> FloatArray:
        """Convert index coordinates to physical coordinates."""

        coords = as_coordinate_array(coordinates, ndim=self.ndim)
        return coords * self.spacing_array + self.origin_array

    def world_to_index(self, coordinates: object) -> FloatArray:
        """Convert physical coordinates to floating-point index coordinates."""

        coords = as_coordinate_array(coordinates, ndim=self.ndim)
        return (coords - self.origin_array) / self.spacing_array

    def affine_matrix(self) -> FloatArray:
        """Return a simple diagonal index-to-world affine matrix."""

        affine = np.eye(self.ndim + 1, dtype=float)
        for axis, spacing in enumerate(self.spacing):
            affine[axis, axis] = spacing
            affine[axis, -1] = self.origin[axis]
        return affine

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable grid summary."""

        return {
            "shape": self.shape,
            "spacing": self.spacing,
            "origin": self.origin,
            "axis_names": self.axis_names,
            "ndim": self.ndim,
            "n_voxels": self.n_voxels,
            "physical_extent": self.physical_extent,
            "physical_size": self.physical_size,
        }
