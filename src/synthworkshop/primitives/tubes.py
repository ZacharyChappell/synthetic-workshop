"""Single-object tube rendering around sampled 3D centrelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from synthworkshop.cross_sections import CrossSection
from synthworkshop.grid import GridSpec
from synthworkshop.ground_truth import SceneTruth
from synthworkshop.primitives.curves import Centreline
from synthworkshop.primitives.frames import ReferenceFrame, build_reference_guided_frame
from synthworkshop.profiles import ScalarProfile
from synthworkshop.scenes import (
    CompositionRules,
    ObjectRole,
    OverlapPolicy,
    RenderedScene,
    SceneObjectMetadata,
)


def _validate_chunk_size(chunk_size: int) -> int:
    """Validate chunk size for nearest-centreline calculations."""

    value = int(chunk_size)
    if value <= 0:
        raise ValueError("chunk_size must be positive.")
    return value


def _validate_object_id(object_id: str) -> str:
    """Validate an object identifier."""

    value = str(object_id)
    if not value:
        raise ValueError("object_id must be a non-empty string.")
    return value


def _world_points(grid: GridSpec) -> np.ndarray:
    """Return flattened physical grid coordinates."""

    return np.column_stack([axis.reshape(-1) for axis in grid.world_arrays()])


def _nearest_centreline_indices(
    points_mm: np.ndarray,
    centreline: Centreline,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Find nearest sampled centreline point for each physical grid point."""

    n_points = points_mm.shape[0]
    nearest = np.empty(n_points, dtype=np.int32)
    centre = np.asarray(centreline.coordinates_mm, dtype=float)

    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        chunk = points_mm[start:stop]
        diff = chunk[:, None, :] - centre[None, :, :]
        distance2 = np.sum(diff * diff, axis=2)
        nearest[start:stop] = np.argmin(distance2, axis=1).astype(np.int32)

    return nearest


def _skeleton_mask_from_centreline(
    grid: GridSpec, centreline: Centreline
) -> np.ndarray:
    """Rasterise sampled centreline coordinates to a nearest-voxel skeleton mask."""

    coords = np.rint(grid.world_to_index(centreline.coordinates_mm)).astype(int)
    mask = np.zeros(grid.shape, dtype=bool)
    in_bounds = np.ones(coords.shape[0], dtype=bool)
    for axis, size in enumerate(grid.shape):
        in_bounds &= (coords[:, axis] >= 0) & (coords[:, axis] < size)
    valid_coords = coords[in_bounds]
    if valid_coords.size:
        mask[tuple(valid_coords.T)] = True
    return mask


@dataclass(frozen=True)
class TubeObject:
    """Single tube object defined by centreline, cross-section, and profile."""

    object_id: str
    centreline: Centreline
    cross_section: CrossSection
    profile: ScalarProfile
    map_name: str = "scalar"
    role: ObjectRole | str = ObjectRole.TARGET
    label: int = 1
    priority: int = 0
    frame: ReferenceFrame | None = None
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object_id = _validate_object_id(self.object_id)
        map_name = str(self.map_name)
        if not map_name:
            raise ValueError("map_name must be a non-empty string.")
        if self.centreline.ndim != 3:
            raise ValueError("TubeObject currently requires a 3D centreline.")
        if self.frame is not None and self.frame.n_points != self.centreline.n_points:
            raise ValueError("frame must contain one row per centreline point.")
        label = int(self.label)
        if label <= 0:
            raise ValueError("TubeObject label must be a positive integer.")
        priority = int(self.priority)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "map_name", map_name)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def render(
        self,
        grid: GridSpec,
        *,
        chunk_size: int = 65536,
        overlap_policy: OverlapPolicy | str = OverlapPolicy.ALLOW,
    ) -> RenderedScene:
        """Render this tube as a single-object RenderedScene."""

        if grid.ndim != 3:
            raise ValueError("TubeObject rendering currently requires a 3D grid.")
        chunk = _validate_chunk_size(chunk_size)
        frame = self.frame or build_reference_guided_frame(self.centreline)

        points = _world_points(grid)
        nearest_index = _nearest_centreline_indices(
            points,
            self.centreline,
            chunk_size=chunk,
        )
        nearest_coords = self.centreline.coordinates_mm[nearest_index]
        local = points - nearest_coords

        primary = frame.primary_axes[nearest_index]
        secondary = frame.secondary_axes[nearest_index]
        signed_u = np.sum(local * primary, axis=1)
        signed_v = np.sum(local * secondary, axis=1)
        longitudinal = self.centreline.arclength_mm[nearest_index]

        cross_eval = self.cross_section.evaluate(
            signed_u,
            signed_v,
            longitudinal_mm=longitudinal,
        )
        profile_eval = self.profile.evaluate(
            rho=cross_eval.rho,
            radial_distance_mm=cross_eval.radial_distance_mm,
            inside=cross_eval.inside,
            signed_u_mm=signed_u,
            longitudinal_mm=longitudinal,
        )

        mask = np.asarray(cross_eval.inside, dtype=bool).reshape(grid.shape)
        scalar = np.asarray(profile_eval.values, dtype=float).reshape(grid.shape)
        label_map = np.where(mask, self.label, 0).astype(np.int32)
        skeleton = _skeleton_mask_from_centreline(grid, self.centreline)

        radial_distance = np.asarray(
            cross_eval.radial_distance_mm, dtype=float
        ).reshape(grid.shape)
        signed_u_map = signed_u.reshape(grid.shape)
        signed_v_map = signed_v.reshape(grid.shape)
        longitudinal_map = longitudinal.reshape(grid.shape)
        nearest_map = nearest_index.reshape(grid.shape)

        object_metadata = SceneObjectMetadata(
            object_id=self.object_id,
            role=self.role,
            label=self.label,
            priority=self.priority,
            name=self.name,
            description=self.description,
            metadata={
                **self.metadata,
                "tube_kind": "sampled_centreline_tube",
                "cross_section": self.cross_section.summary(),
                "profile": self.profile.summary(),
            },
        )

        centreline_table = self.centreline.to_dataframe()
        frame_table = frame.to_dataframe()
        truth = SceneTruth(
            geometric={
                self.object_id: {
                    "kind": "tube",
                    "centreline_n_points": self.centreline.n_points,
                    "centreline_length_mm": self.centreline.length_mm,
                    "cross_section": self.cross_section.summary(),
                }
            },
            objects={
                self.object_id: {
                    "role": ObjectRole(object_metadata.role).value,
                    "label": self.label,
                    "priority": self.priority,
                    "mask_voxels": int(np.sum(mask)),
                    "skeleton_voxels": int(np.sum(skeleton)),
                }
            },
            scalar_fields={
                self.map_name: {
                    "object_id": self.object_id,
                    "profile": self.profile.summary(),
                }
            },
            tables={
                "centrelines": centreline_table,
                "frames": frame_table,
            },
            metadata={
                "truth_scope": (
                    "Method-agnostic tube geometry, object mask, scalar field, "
                    "distance, signed-offset, centreline, and frame truth."
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
            skeleton_masks={self.object_id: skeleton},
            centrelines={self.object_id: centreline_table},
            frames={self.object_id: frame_table},
            distance_maps={self.object_id: radial_distance},
            signed_offset_maps={
                f"{self.object_id}:u_mm": signed_u_map,
                f"{self.object_id}:v_mm": signed_v_map,
                f"{self.object_id}:longitudinal_mm": longitudinal_map,
                f"{self.object_id}:nearest_centreline_index": nearest_map,
            },
            metadata={
                "renderer": "TubeObject.render",
                "object_id": self.object_id,
                "map_name": self.map_name,
            },
            provenance={
                "package": "synthworkshop",
                "stage": "M1e",
            },
        )
