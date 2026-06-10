"""Implicit volumetric object primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.grid import GridSpec
from synthworkshop.ground_truth import SceneTruth
from synthworkshop.profiles import ScalarProfile
from synthworkshop.scenes import (
    CompositionRules,
    ObjectRole,
    OverlapPolicy,
    RenderedScene,
    SceneObjectMetadata,
)


def _validate_object_id(value: object) -> str:
    """Validate a non-empty object identifier."""

    text = str(value)
    if not text:
        raise ValueError("object_id must be a non-empty string.")
    return text


def _validate_map_name(value: object) -> str:
    """Validate a non-empty scalar map name."""

    text = str(value)
    if not text:
        raise ValueError("map_name must be a non-empty string.")
    return text


def _validate_positive_label(value: int) -> int:
    """Validate a positive integer label."""

    label = int(value)
    if label <= 0:
        raise ValueError("label must be a positive integer.")
    return label


def _as_point3(value: Sequence[float], *, name: str) -> np.ndarray:
    """Validate a 3D coordinate."""

    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must be a 3D coordinate.")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} contains non-finite values.")
    return point


def _as_positive_axes3(value: Sequence[float], *, name: str) -> np.ndarray:
    """Validate three positive finite axis lengths."""

    axes = np.asarray(value, dtype=float)
    if axes.shape != (3,):
        raise ValueError(f"{name} must contain three values.")
    if not np.all(np.isfinite(axes)) or np.any(axes <= 0):
        raise ValueError(f"{name} must contain finite positive values.")
    return axes


def _validate_positive_float(value: float, *, name: str) -> float:
    """Validate a positive finite float."""

    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return out


def _world_points(grid: GridSpec) -> np.ndarray:
    """Return flattened physical grid coordinates."""

    return np.column_stack([axis.reshape(-1) for axis in grid.world_arrays()])


def _axis_offset_maps(
    local: np.ndarray,
    shape: tuple[int, ...],
    *,
    object_id: str,
) -> dict[str, np.ndarray]:
    """Return local axis offset maps."""

    return {
        f"{object_id}:i_mm": local[:, 0].reshape(shape),
        f"{object_id}:j_mm": local[:, 1].reshape(shape),
        f"{object_id}:k_mm": local[:, 2].reshape(shape),
    }


@dataclass(frozen=True)
class SphereObject:
    """Axis-aligned implicit sphere."""

    object_id: str
    centre_mm: Sequence[float]
    radius_mm: float
    profile: ScalarProfile
    map_name: str = "scalar"
    role: ObjectRole | str = ObjectRole.INCLUSION
    label: int = 1
    priority: int = 0
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _validate_object_id(self.object_id))
        object.__setattr__(
            self, "centre_mm", _as_point3(self.centre_mm, name="centre_mm")
        )
        object.__setattr__(
            self,
            "radius_mm",
            _validate_positive_float(self.radius_mm, name="radius_mm"),
        )
        object.__setattr__(self, "map_name", _validate_map_name(self.map_name))
        object.__setattr__(self, "role", ObjectRole(self.role))
        object.__setattr__(self, "label", _validate_positive_label(self.label))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def volume_mm3(self) -> float:
        """Analytic sphere volume."""

        return float((4.0 / 3.0) * np.pi * self.radius_mm**3)

    def render(
        self,
        grid: GridSpec,
        *,
        overlap_policy: OverlapPolicy | str = OverlapPolicy.ALLOW,
        chunk_size: int | None = None,
    ) -> RenderedScene:
        """Render the sphere as a single-object scene."""

        _ = chunk_size

        if grid.ndim != 3:
            raise ValueError("SphereObject rendering currently requires a 3D grid.")

        points = _world_points(grid)
        local = points - self.centre_mm[None, :]
        radial = np.linalg.norm(local, axis=1)
        rho = radial / self.radius_mm
        inside = rho <= 1.0

        profile_eval = self.profile.evaluate(
            rho=rho,
            radial_distance_mm=radial,
            inside=inside,
            signed_u_mm=local[:, 0],
            longitudinal_mm=np.zeros_like(radial),
        )

        mask = inside.reshape(grid.shape)
        scalar = np.asarray(profile_eval.values, dtype=float).reshape(grid.shape)
        label_map = np.where(mask, self.label, 0).astype(np.int32)
        radial_map = radial.reshape(grid.shape)
        rho_map = rho.reshape(grid.shape)

        object_metadata = SceneObjectMetadata(
            object_id=self.object_id,
            role=self.role,
            label=self.label,
            priority=self.priority,
            name=self.name,
            description=self.description,
            metadata={
                **self.metadata,
                "implicit_kind": "sphere",
                "centre_mm": self.centre_mm.tolist(),
                "radius_mm": self.radius_mm,
                "profile": self.profile.summary(),
            },
        )

        truth = SceneTruth(
            geometric={
                self.object_id: {
                    "kind": "sphere",
                    "centre_mm": self.centre_mm.tolist(),
                    "radius_mm": self.radius_mm,
                    "volume_mm3": self.volume_mm3,
                }
            },
            objects={
                self.object_id: {
                    "role": self.role.value,
                    "label": self.label,
                    "priority": self.priority,
                    "mask_voxels": int(np.sum(mask)),
                }
            },
            scalar_fields={
                self.map_name: {
                    "object_id": self.object_id,
                    "profile": self.profile.summary(),
                }
            },
            metadata={
                "truth_scope": (
                    "Method-agnostic implicit-object truth: analytic sphere "
                    "geometry, object mask, scalar field, radial distance, local "
                    "axis offsets, and rendering metadata."
                )
            },
        )

        return RenderedScene(
            grid=grid,
            scalar_maps={self.map_name: scalar},
            label_map=label_map,
            object_masks={self.object_id: mask},
            object_metadata={self.object_id: object_metadata},
            truth=truth,
            composition=CompositionRules(overlap_policy=overlap_policy),
            distance_maps={self.object_id: radial_map},
            signed_offset_maps={
                **_axis_offset_maps(local, grid.shape, object_id=self.object_id),
                f"{self.object_id}:rho": rho_map,
            },
            metadata={
                "renderer": "SphereObject.render",
                "object_id": self.object_id,
                "map_name": self.map_name,
            },
            provenance={
                "package": "synthworkshop",
                "stage": "M2e-a",
            },
        )


@dataclass(frozen=True)
class EllipsoidObject:
    """Axis-aligned implicit ellipsoid."""

    object_id: str
    centre_mm: Sequence[float]
    radii_mm: Sequence[float]
    profile: ScalarProfile
    map_name: str = "scalar"
    role: ObjectRole | str = ObjectRole.INCLUSION
    label: int = 1
    priority: int = 0
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _validate_object_id(self.object_id))
        object.__setattr__(
            self, "centre_mm", _as_point3(self.centre_mm, name="centre_mm")
        )
        object.__setattr__(
            self, "radii_mm", _as_positive_axes3(self.radii_mm, name="radii_mm")
        )
        object.__setattr__(self, "map_name", _validate_map_name(self.map_name))
        object.__setattr__(self, "role", ObjectRole(self.role))
        object.__setattr__(self, "label", _validate_positive_label(self.label))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def volume_mm3(self) -> float:
        """Analytic ellipsoid volume."""

        return float((4.0 / 3.0) * np.pi * np.prod(self.radii_mm))

    def render(
        self,
        grid: GridSpec,
        *,
        overlap_policy: OverlapPolicy | str = OverlapPolicy.ALLOW,
        chunk_size: int | None = None,
    ) -> RenderedScene:
        """Render the ellipsoid as a single-object scene."""

        _ = chunk_size

        if grid.ndim != 3:
            raise ValueError("EllipsoidObject rendering currently requires a 3D grid.")

        points = _world_points(grid)
        local = points - self.centre_mm[None, :]
        scaled = local / self.radii_mm[None, :]
        rho = np.sqrt(np.sum(scaled * scaled, axis=1))
        radial = np.linalg.norm(local, axis=1)
        inside = rho <= 1.0

        profile_eval = self.profile.evaluate(
            rho=rho,
            radial_distance_mm=radial,
            inside=inside,
            signed_u_mm=local[:, 0],
            longitudinal_mm=np.zeros_like(radial),
        )

        mask = inside.reshape(grid.shape)
        scalar = np.asarray(profile_eval.values, dtype=float).reshape(grid.shape)
        label_map = np.where(mask, self.label, 0).astype(np.int32)
        radial_map = radial.reshape(grid.shape)
        rho_map = rho.reshape(grid.shape)

        object_metadata = SceneObjectMetadata(
            object_id=self.object_id,
            role=self.role,
            label=self.label,
            priority=self.priority,
            name=self.name,
            description=self.description,
            metadata={
                **self.metadata,
                "implicit_kind": "ellipsoid",
                "centre_mm": self.centre_mm.tolist(),
                "radii_mm": self.radii_mm.tolist(),
                "profile": self.profile.summary(),
            },
        )

        truth = SceneTruth(
            geometric={
                self.object_id: {
                    "kind": "ellipsoid",
                    "centre_mm": self.centre_mm.tolist(),
                    "radii_mm": self.radii_mm.tolist(),
                    "volume_mm3": self.volume_mm3,
                }
            },
            objects={
                self.object_id: {
                    "role": self.role.value,
                    "label": self.label,
                    "priority": self.priority,
                    "mask_voxels": int(np.sum(mask)),
                }
            },
            scalar_fields={
                self.map_name: {
                    "object_id": self.object_id,
                    "profile": self.profile.summary(),
                }
            },
            metadata={
                "truth_scope": (
                    "Method-agnostic implicit-object truth: analytic ellipsoid "
                    "geometry, object mask, scalar field, radial distance, local "
                    "axis offsets, normalised ellipsoid radius, and rendering "
                    "metadata."
                )
            },
        )

        return RenderedScene(
            grid=grid,
            scalar_maps={self.map_name: scalar},
            label_map=label_map,
            object_masks={self.object_id: mask},
            object_metadata={self.object_id: object_metadata},
            truth=truth,
            composition=CompositionRules(overlap_policy=overlap_policy),
            distance_maps={self.object_id: radial_map},
            signed_offset_maps={
                **_axis_offset_maps(local, grid.shape, object_id=self.object_id),
                f"{self.object_id}:rho": rho_map,
            },
            metadata={
                "renderer": "EllipsoidObject.render",
                "object_id": self.object_id,
                "map_name": self.map_name,
            },
            provenance={
                "package": "synthworkshop",
                "stage": "M2e-a",
            },
        )


def _coerce_axis(value: int | str, *, name: str = "normal_axis") -> int:
    """Coerce an axis specifier to an integer axis index."""

    if isinstance(value, str):
        text = value.strip().lower()
        mapping = {
            "0": 0,
            "1": 1,
            "2": 2,
            "i": 0,
            "j": 1,
            "k": 2,
            "x": 0,
            "y": 1,
            "z": 2,
        }
        if text not in mapping:
            raise ValueError(f"{name} must be one of 0, 1, 2, i, j, k, x, y, z.")
        return mapping[text]

    axis = int(value)
    if axis not in {0, 1, 2}:
        raise ValueError(f"{name} must be 0, 1, or 2.")
    return axis


def _as_optional_positive_axes2(
    value: Sequence[float] | None,
    *,
    name: str,
) -> np.ndarray | None:
    """Validate optional two positive finite half-extents."""

    if value is None:
        return None

    axes = np.asarray(value, dtype=float)
    if axes.shape != (2,):
        raise ValueError(f"{name} must contain two values.")
    if not np.all(np.isfinite(axes)) or np.any(axes <= 0):
        raise ValueError(f"{name} must contain finite positive values.")
    return axes


@dataclass(frozen=True)
class SlabObject:
    """Axis-aligned implicit slab or sheet object.

    The slab is finite along its normal axis and optionally finite in the two
    in-plane axes. If ``half_extent_mm`` is omitted, the slab extends across the
    whole rendered grid in-plane.

    The scalar profile is evaluated across slab thickness using:

        rho = |normal_offset| / (thickness_mm / 2)

    so ``rho == 0`` at the slab mid-plane and ``rho == 1`` at the slab faces.
    """

    object_id: str
    centre_mm: Sequence[float]
    normal_axis: int | str
    thickness_mm: float
    profile: ScalarProfile
    half_extent_mm: Sequence[float] | None = None
    map_name: str = "scalar"
    role: ObjectRole | str = ObjectRole.ENVIRONMENT
    label: int = 1
    priority: int = 0
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _validate_object_id(self.object_id))
        object.__setattr__(
            self,
            "centre_mm",
            _as_point3(self.centre_mm, name="centre_mm"),
        )
        object.__setattr__(
            self,
            "normal_axis",
            _coerce_axis(self.normal_axis, name="normal_axis"),
        )
        object.__setattr__(
            self,
            "thickness_mm",
            _validate_positive_float(self.thickness_mm, name="thickness_mm"),
        )
        object.__setattr__(
            self,
            "half_extent_mm",
            _as_optional_positive_axes2(self.half_extent_mm, name="half_extent_mm"),
        )
        object.__setattr__(self, "map_name", _validate_map_name(self.map_name))
        object.__setattr__(self, "role", ObjectRole(self.role))
        object.__setattr__(self, "label", _validate_positive_label(self.label))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def normal_axis_name(self) -> str:
        """Name of the normal axis."""

        return ("i", "j", "k")[self.normal_axis]

    @property
    def in_plane_axes(self) -> tuple[int, int]:
        """Axes spanning the slab plane."""

        return tuple(axis for axis in (0, 1, 2) if axis != self.normal_axis)  # type: ignore[return-value]

    @property
    def in_plane_axis_names(self) -> tuple[str, str]:
        """Names of the slab in-plane axes."""

        names = ("i", "j", "k")
        return tuple(names[axis] for axis in self.in_plane_axes)  # type: ignore[return-value]

    @property
    def volume_mm3(self) -> float:
        """Analytic slab volume if finite in-plane, otherwise infinity."""

        if self.half_extent_mm is None:
            return float("inf")
        return float(
            self.thickness_mm
            * (2.0 * self.half_extent_mm[0])
            * (2.0 * self.half_extent_mm[1])
        )

    def render(
        self,
        grid: GridSpec,
        *,
        overlap_policy: OverlapPolicy | str = OverlapPolicy.ALLOW,
        chunk_size: int | None = None,
    ) -> RenderedScene:
        """Render the slab as a single-object scene."""

        _ = chunk_size

        if grid.ndim != 3:
            raise ValueError("SlabObject rendering currently requires a 3D grid.")

        points = _world_points(grid)
        local = points - self.centre_mm[None, :]

        normal_offset = local[:, self.normal_axis]
        in_plane_axes = self.in_plane_axes
        in_plane_offsets = local[:, in_plane_axes]

        half_thickness = 0.5 * self.thickness_mm
        rho = np.abs(normal_offset) / half_thickness
        normal_inside = rho <= 1.0

        if self.half_extent_mm is None:
            in_plane_inside = np.ones(points.shape[0], dtype=bool)
        else:
            in_plane_inside = np.all(
                np.abs(in_plane_offsets) <= self.half_extent_mm[None, :],
                axis=1,
            )

        inside = normal_inside & in_plane_inside
        normal_distance = np.abs(normal_offset)

        profile_eval = self.profile.evaluate(
            rho=rho,
            radial_distance_mm=normal_distance,
            inside=inside,
            signed_u_mm=normal_offset,
            longitudinal_mm=np.zeros_like(normal_distance),
        )

        mask = inside.reshape(grid.shape)
        scalar = np.asarray(profile_eval.values, dtype=float).reshape(grid.shape)
        label_map = np.where(mask, self.label, 0).astype(np.int32)
        normal_distance_map = normal_distance.reshape(grid.shape)
        rho_map = rho.reshape(grid.shape)

        object_metadata = SceneObjectMetadata(
            object_id=self.object_id,
            role=self.role,
            label=self.label,
            priority=self.priority,
            name=self.name,
            description=self.description,
            metadata={
                **self.metadata,
                "implicit_kind": "slab",
                "centre_mm": self.centre_mm.tolist(),
                "normal_axis": self.normal_axis,
                "normal_axis_name": self.normal_axis_name,
                "thickness_mm": self.thickness_mm,
                "half_extent_mm": (
                    None
                    if self.half_extent_mm is None
                    else self.half_extent_mm.tolist()
                ),
                "profile": self.profile.summary(),
            },
        )

        truth = SceneTruth(
            geometric={
                self.object_id: {
                    "kind": "slab",
                    "centre_mm": self.centre_mm.tolist(),
                    "normal_axis": self.normal_axis,
                    "normal_axis_name": self.normal_axis_name,
                    "in_plane_axes": self.in_plane_axes,
                    "in_plane_axis_names": self.in_plane_axis_names,
                    "thickness_mm": self.thickness_mm,
                    "half_extent_mm": (
                        None
                        if self.half_extent_mm is None
                        else self.half_extent_mm.tolist()
                    ),
                    "volume_mm3": self.volume_mm3,
                }
            },
            objects={
                self.object_id: {
                    "role": self.role.value,
                    "label": self.label,
                    "priority": self.priority,
                    "mask_voxels": int(np.sum(mask)),
                }
            },
            scalar_fields={
                self.map_name: {
                    "object_id": self.object_id,
                    "profile": self.profile.summary(),
                }
            },
            metadata={
                "truth_scope": (
                    "Method-agnostic implicit-object truth: analytic slab geometry, "
                    "object mask, scalar field, normal-distance map, local axis "
                    "offsets, in-plane offsets, and rendering metadata."
                )
            },
        )

        in_plane_u_name, in_plane_v_name = self.in_plane_axis_names

        return RenderedScene(
            grid=grid,
            scalar_maps={self.map_name: scalar},
            label_map=label_map,
            object_masks={self.object_id: mask},
            object_metadata={self.object_id: object_metadata},
            truth=truth,
            composition=CompositionRules(overlap_policy=overlap_policy),
            distance_maps={self.object_id: normal_distance_map},
            signed_offset_maps={
                **_axis_offset_maps(local, grid.shape, object_id=self.object_id),
                f"{self.object_id}:normal_mm": normal_offset.reshape(grid.shape),
                f"{self.object_id}:in_plane_{in_plane_u_name}_mm": in_plane_offsets[
                    :, 0
                ].reshape(grid.shape),
                f"{self.object_id}:in_plane_{in_plane_v_name}_mm": in_plane_offsets[
                    :, 1
                ].reshape(grid.shape),
                f"{self.object_id}:rho": rho_map,
            },
            metadata={
                "renderer": "SlabObject.render",
                "object_id": self.object_id,
                "map_name": self.map_name,
            },
            provenance={
                "package": "synthworkshop",
                "stage": "M2e-c-a",
            },
        )


def _coerce_axis_direction(value: int | float | str = 1) -> int:
    """Coerce an axis direction to +1 or -1."""

    if isinstance(value, str):
        text = value.strip().lower()
        mapping = {
            "+": 1,
            "+1": 1,
            "1": 1,
            "positive": 1,
            "pos": 1,
            "forward": 1,
            "-": -1,
            "-1": -1,
            "negative": -1,
            "neg": -1,
            "backward": -1,
        }
        if text not in mapping:
            raise ValueError("axis_direction must be +1 or -1.")
        return mapping[text]

    direction = int(value)
    if direction not in {-1, 1}:
        raise ValueError("axis_direction must be +1 or -1.")
    return direction


def _perpendicular_axes(axis: int) -> tuple[int, int]:
    """Return the two axes perpendicular to an axis index."""

    return tuple(idx for idx in (0, 1, 2) if idx != axis)  # type: ignore[return-value]


def _radial_distance_from_axis(local: np.ndarray, axis: int) -> np.ndarray:
    """Return Euclidean radial distance from an axis."""

    perp = _perpendicular_axes(axis)
    return np.sqrt(np.sum(local[:, perp] ** 2, axis=1))


def _safe_conical_rho(radial: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Return finite radial/radius values for conical supports.

    At a true cone apex the local support radius is zero. The exact apex point
    has rho == 0. Off-axis points on a zero-radius plane are outside the cone;
    they are assigned a finite outside-support rho value so scalar-profile
    validation never receives infinities.
    """

    rho = np.full(radial.shape, 2.0, dtype=float)
    np.divide(radial, radius, out=rho, where=radius > 0.0)
    rho[(radius == 0.0) & (radial == 0.0)] = 0.0
    return rho


@dataclass(frozen=True)
class ConeObject:
    """Axis-aligned implicit cone object.

    ``apex_mm`` defines the cone apex. ``axis`` and ``axis_direction`` define
    the direction from apex to base. The radius grows linearly from zero at the
    apex to ``base_radius_mm`` at ``height_mm``.
    """

    object_id: str
    apex_mm: Sequence[float]
    axis: int | str
    height_mm: float
    base_radius_mm: float
    profile: ScalarProfile
    axis_direction: int | float | str = 1
    map_name: str = "scalar"
    role: ObjectRole | str = ObjectRole.INCLUSION
    label: int = 1
    priority: int = 0
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _validate_object_id(self.object_id))
        object.__setattr__(self, "apex_mm", _as_point3(self.apex_mm, name="apex_mm"))
        object.__setattr__(self, "axis", _coerce_axis(self.axis, name="axis"))
        object.__setattr__(
            self,
            "axis_direction",
            _coerce_axis_direction(self.axis_direction),
        )
        object.__setattr__(
            self,
            "height_mm",
            _validate_positive_float(self.height_mm, name="height_mm"),
        )
        object.__setattr__(
            self,
            "base_radius_mm",
            _validate_positive_float(self.base_radius_mm, name="base_radius_mm"),
        )
        object.__setattr__(self, "map_name", _validate_map_name(self.map_name))
        object.__setattr__(self, "role", ObjectRole(self.role))
        object.__setattr__(self, "label", _validate_positive_label(self.label))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def axis_name(self) -> str:
        """Name of the cone axis."""

        return ("i", "j", "k")[self.axis]

    @property
    def base_centre_mm(self) -> np.ndarray:
        """Physical coordinate of the cone base centre."""

        out = self.apex_mm.copy()
        out[self.axis] += self.axis_direction * self.height_mm
        return out

    @property
    def volume_mm3(self) -> float:
        """Analytic cone volume."""

        return float((1.0 / 3.0) * np.pi * self.base_radius_mm**2 * self.height_mm)

    def radius_at(self, axial_mm: ArrayLike) -> np.ndarray:
        """Evaluate cone radius at axial distances from the apex."""

        axial = np.asarray(axial_mm, dtype=float)
        if not np.all(np.isfinite(axial)):
            raise ValueError("axial_mm contains non-finite values.")
        fraction = np.clip(axial / self.height_mm, 0.0, 1.0)
        return self.base_radius_mm * fraction

    def render(
        self,
        grid: GridSpec,
        *,
        overlap_policy: OverlapPolicy | str = OverlapPolicy.ALLOW,
        chunk_size: int | None = None,
    ) -> RenderedScene:
        """Render the cone as a single-object scene."""

        _ = chunk_size

        if grid.ndim != 3:
            raise ValueError("ConeObject rendering currently requires a 3D grid.")

        points = _world_points(grid)
        local = points - self.apex_mm[None, :]
        axial = local[:, self.axis] * self.axis_direction
        radial = _radial_distance_from_axis(local, self.axis)
        radius = self.radius_at(axial)
        rho = _safe_conical_rho(radial, radius)
        inside = (axial >= 0.0) & (axial <= self.height_mm) & (rho <= 1.0)

        profile_eval = self.profile.evaluate(
            rho=rho,
            radial_distance_mm=radial,
            inside=inside,
            signed_u_mm=radial,
            longitudinal_mm=axial,
        )

        mask = inside.reshape(grid.shape)
        scalar = np.asarray(profile_eval.values, dtype=float).reshape(grid.shape)
        label_map = np.where(mask, self.label, 0).astype(np.int32)
        radial_map = radial.reshape(grid.shape)
        rho_map = rho.reshape(grid.shape)
        axial_map = axial.reshape(grid.shape)

        object_metadata = SceneObjectMetadata(
            object_id=self.object_id,
            role=self.role,
            label=self.label,
            priority=self.priority,
            name=self.name,
            description=self.description,
            metadata={
                **self.metadata,
                "implicit_kind": "cone",
                "apex_mm": self.apex_mm.tolist(),
                "base_centre_mm": self.base_centre_mm.tolist(),
                "axis": self.axis,
                "axis_name": self.axis_name,
                "axis_direction": self.axis_direction,
                "height_mm": self.height_mm,
                "base_radius_mm": self.base_radius_mm,
                "profile": self.profile.summary(),
            },
        )

        truth = SceneTruth(
            geometric={
                self.object_id: {
                    "kind": "cone",
                    "apex_mm": self.apex_mm.tolist(),
                    "base_centre_mm": self.base_centre_mm.tolist(),
                    "axis": self.axis,
                    "axis_name": self.axis_name,
                    "axis_direction": self.axis_direction,
                    "height_mm": self.height_mm,
                    "base_radius_mm": self.base_radius_mm,
                    "volume_mm3": self.volume_mm3,
                }
            },
            objects={
                self.object_id: {
                    "role": self.role.value,
                    "label": self.label,
                    "priority": self.priority,
                    "mask_voxels": int(np.sum(mask)),
                }
            },
            scalar_fields={
                self.map_name: {
                    "object_id": self.object_id,
                    "profile": self.profile.summary(),
                }
            },
            metadata={
                "truth_scope": (
                    "Method-agnostic implicit-object truth: analytic cone "
                    "geometry, object mask, scalar field, axial coordinate, "
                    "radial distance from cone axis, normalised radial support, "
                    "local axis offsets, and rendering metadata."
                )
            },
        )

        return RenderedScene(
            grid=grid,
            scalar_maps={self.map_name: scalar},
            label_map=label_map,
            object_masks={self.object_id: mask},
            object_metadata={self.object_id: object_metadata},
            truth=truth,
            composition=CompositionRules(overlap_policy=overlap_policy),
            distance_maps={self.object_id: radial_map},
            signed_offset_maps={
                **_axis_offset_maps(local, grid.shape, object_id=self.object_id),
                f"{self.object_id}:axial_mm": axial_map,
                f"{self.object_id}:radial_mm": radial_map,
                f"{self.object_id}:rho": rho_map,
            },
            metadata={
                "renderer": "ConeObject.render",
                "object_id": self.object_id,
                "map_name": self.map_name,
            },
            provenance={
                "package": "synthworkshop",
                "stage": "M2e-c-b",
            },
        )


@dataclass(frozen=True)
class FrustumObject:
    """Axis-aligned conical frustum object.

    ``start_mm`` defines the centre of the first circular face. ``axis`` and
    ``axis_direction`` define the direction towards the second face. The radius
    varies linearly from ``radius_start_mm`` to ``radius_end_mm`` along
    ``height_mm``.
    """

    object_id: str
    start_mm: Sequence[float]
    axis: int | str
    height_mm: float
    radius_start_mm: float
    radius_end_mm: float
    profile: ScalarProfile
    axis_direction: int | float | str = 1
    map_name: str = "scalar"
    role: ObjectRole | str = ObjectRole.INCLUSION
    label: int = 1
    priority: int = 0
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _validate_object_id(self.object_id))
        object.__setattr__(self, "start_mm", _as_point3(self.start_mm, name="start_mm"))
        object.__setattr__(self, "axis", _coerce_axis(self.axis, name="axis"))
        object.__setattr__(
            self,
            "axis_direction",
            _coerce_axis_direction(self.axis_direction),
        )
        object.__setattr__(
            self,
            "height_mm",
            _validate_positive_float(self.height_mm, name="height_mm"),
        )
        object.__setattr__(
            self,
            "radius_start_mm",
            _validate_positive_float(self.radius_start_mm, name="radius_start_mm"),
        )
        object.__setattr__(
            self,
            "radius_end_mm",
            _validate_positive_float(self.radius_end_mm, name="radius_end_mm"),
        )
        object.__setattr__(self, "map_name", _validate_map_name(self.map_name))
        object.__setattr__(self, "role", ObjectRole(self.role))
        object.__setattr__(self, "label", _validate_positive_label(self.label))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def axis_name(self) -> str:
        """Name of the frustum axis."""

        return ("i", "j", "k")[self.axis]

    @property
    def end_mm(self) -> np.ndarray:
        """Physical coordinate of the second face centre."""

        out = self.start_mm.copy()
        out[self.axis] += self.axis_direction * self.height_mm
        return out

    @property
    def volume_mm3(self) -> float:
        """Analytic conical frustum volume."""

        return float(
            (np.pi * self.height_mm / 3.0)
            * (
                self.radius_start_mm**2
                + self.radius_start_mm * self.radius_end_mm
                + self.radius_end_mm**2
            )
        )

    def radius_at(self, axial_mm: ArrayLike) -> np.ndarray:
        """Evaluate frustum radius at axial distances from the start face."""

        axial = np.asarray(axial_mm, dtype=float)
        if not np.all(np.isfinite(axial)):
            raise ValueError("axial_mm contains non-finite values.")
        fraction = np.clip(axial / self.height_mm, 0.0, 1.0)
        return (
            self.radius_start_mm
            + (self.radius_end_mm - self.radius_start_mm) * fraction
        )

    def render(
        self,
        grid: GridSpec,
        *,
        overlap_policy: OverlapPolicy | str = OverlapPolicy.ALLOW,
        chunk_size: int | None = None,
    ) -> RenderedScene:
        """Render the frustum as a single-object scene."""

        _ = chunk_size

        if grid.ndim != 3:
            raise ValueError("FrustumObject rendering currently requires a 3D grid.")

        points = _world_points(grid)
        local = points - self.start_mm[None, :]
        axial = local[:, self.axis] * self.axis_direction
        radial = _radial_distance_from_axis(local, self.axis)
        radius = self.radius_at(axial)
        rho = radial / radius
        inside = (axial >= 0.0) & (axial <= self.height_mm) & (rho <= 1.0)

        profile_eval = self.profile.evaluate(
            rho=rho,
            radial_distance_mm=radial,
            inside=inside,
            signed_u_mm=radial,
            longitudinal_mm=axial,
        )

        mask = inside.reshape(grid.shape)
        scalar = np.asarray(profile_eval.values, dtype=float).reshape(grid.shape)
        label_map = np.where(mask, self.label, 0).astype(np.int32)
        radial_map = radial.reshape(grid.shape)
        rho_map = rho.reshape(grid.shape)
        axial_map = axial.reshape(grid.shape)

        object_metadata = SceneObjectMetadata(
            object_id=self.object_id,
            role=self.role,
            label=self.label,
            priority=self.priority,
            name=self.name,
            description=self.description,
            metadata={
                **self.metadata,
                "implicit_kind": "frustum",
                "start_mm": self.start_mm.tolist(),
                "end_mm": self.end_mm.tolist(),
                "axis": self.axis,
                "axis_name": self.axis_name,
                "axis_direction": self.axis_direction,
                "height_mm": self.height_mm,
                "radius_start_mm": self.radius_start_mm,
                "radius_end_mm": self.radius_end_mm,
                "profile": self.profile.summary(),
            },
        )

        truth = SceneTruth(
            geometric={
                self.object_id: {
                    "kind": "frustum",
                    "start_mm": self.start_mm.tolist(),
                    "end_mm": self.end_mm.tolist(),
                    "axis": self.axis,
                    "axis_name": self.axis_name,
                    "axis_direction": self.axis_direction,
                    "height_mm": self.height_mm,
                    "radius_start_mm": self.radius_start_mm,
                    "radius_end_mm": self.radius_end_mm,
                    "volume_mm3": self.volume_mm3,
                }
            },
            objects={
                self.object_id: {
                    "role": self.role.value,
                    "label": self.label,
                    "priority": self.priority,
                    "mask_voxels": int(np.sum(mask)),
                }
            },
            scalar_fields={
                self.map_name: {
                    "object_id": self.object_id,
                    "profile": self.profile.summary(),
                }
            },
            metadata={
                "truth_scope": (
                    "Method-agnostic implicit-object truth: analytic frustum "
                    "geometry, object mask, scalar field, axial coordinate, "
                    "radial distance from frustum axis, normalised radial support, "
                    "local axis offsets, and rendering metadata."
                )
            },
        )

        return RenderedScene(
            grid=grid,
            scalar_maps={self.map_name: scalar},
            label_map=label_map,
            object_masks={self.object_id: mask},
            object_metadata={self.object_id: object_metadata},
            truth=truth,
            composition=CompositionRules(overlap_policy=overlap_policy),
            distance_maps={self.object_id: radial_map},
            signed_offset_maps={
                **_axis_offset_maps(local, grid.shape, object_id=self.object_id),
                f"{self.object_id}:axial_mm": axial_map,
                f"{self.object_id}:radial_mm": radial_map,
                f"{self.object_id}:rho": rho_map,
            },
            metadata={
                "renderer": "FrustumObject.render",
                "object_id": self.object_id,
                "map_name": self.map_name,
            },
            provenance={
                "package": "synthworkshop",
                "stage": "M2e-c-b",
            },
        )
