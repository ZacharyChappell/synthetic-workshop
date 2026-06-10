"""Rendered-scene containers and composition metadata."""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from synthworkshop.coordinates import BoolArray, validate_array_shape
from synthworkshop.grid import GridSpec
from synthworkshop.ground_truth.geometry import SceneTruth


class ObjectRole(str, Enum):
    """Semantic object role within a scene."""

    TARGET = "target"
    ENVIRONMENT = "environment"
    DISTRACTOR = "distractor"
    INCLUSION = "inclusion"
    BACKGROUND = "background"
    ANALYSIS_SUPPORT = "analysis_support"


class LabelMode(str, Enum):
    """Rule for constructing labels in overlapping regions."""

    PRIORITY = "priority"
    FIRST = "first"
    LAST = "last"


class ScalarBlend(str, Enum):
    """Rule for combining scalar values in overlapping regions."""

    OVERWRITE = "overwrite"
    MAX = "max"
    SUM = "sum"
    WEIGHTED_MEAN = "weighted_mean"


class OverlapPolicy(str, Enum):
    """How to handle overlapping object masks."""

    ALLOW = "allow"
    WARN = "warn"
    ERROR = "error"


def _coerce_enum(value: Any, enum_type: type[Enum], *, name: str) -> Enum:
    """Coerce strings to enum values with useful errors."""

    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}.") from exc


def _coerce_roles(
    values: Sequence[ObjectRole | str], *, name: str
) -> tuple[ObjectRole, ...]:
    """Coerce role strings to ObjectRole values."""

    if not values:
        raise ValueError(f"{name} must contain at least one role.")
    return tuple(
        _coerce_enum(value, ObjectRole, name=name)  # type: ignore[arg-type]
        for value in values
    )


@dataclass(frozen=True)
class CompositionRules:
    """Explicit scene-composition semantics."""

    label_mode: LabelMode | str = LabelMode.PRIORITY
    scalar_blend: ScalarBlend | str = ScalarBlend.OVERWRITE
    overlap_policy: OverlapPolicy | str = OverlapPolicy.WARN

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "label_mode",
            _coerce_enum(self.label_mode, LabelMode, name="label_mode"),
        )
        object.__setattr__(
            self,
            "scalar_blend",
            _coerce_enum(self.scalar_blend, ScalarBlend, name="scalar_blend"),
        )
        object.__setattr__(
            self,
            "overlap_policy",
            _coerce_enum(self.overlap_policy, OverlapPolicy, name="overlap_policy"),
        )


@dataclass(frozen=True)
class MaskRules:
    """Rules for deriving default target and analysis masks from object roles."""

    target_roles: Sequence[ObjectRole | str] = (ObjectRole.TARGET,)
    analysis_roles: Sequence[ObjectRole | str] = (
        ObjectRole.TARGET,
        ObjectRole.ANALYSIS_SUPPORT,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_roles",
            _coerce_roles(self.target_roles, name="target_roles"),
        )
        object.__setattr__(
            self,
            "analysis_roles",
            _coerce_roles(self.analysis_roles, name="analysis_roles"),
        )


@dataclass(frozen=True)
class SceneObjectMetadata:
    """Metadata for one rendered object or compartment."""

    object_id: str
    role: ObjectRole | str
    label: int
    priority: int = 0
    name: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = str(self.object_id)
        if not object_id:
            raise ValueError("object_id must be a non-empty string.")
        label = int(self.label)
        if label < 0:
            raise ValueError("label must be non-negative.")
        priority = int(self.priority)

        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(
            self,
            "role",
            _coerce_enum(self.role, ObjectRole, name="role"),
        )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "priority", priority)


@dataclass(frozen=True)
class OverlapReport:
    """Summary of object-mask overlap."""

    n_overlap_voxels: int = 0
    object_pair_counts: tuple[Mapping[str, object], ...] = ()
    policy: OverlapPolicy | str = OverlapPolicy.WARN

    def __post_init__(self) -> None:
        if self.n_overlap_voxels < 0:
            raise ValueError("n_overlap_voxels must be non-negative.")
        object.__setattr__(
            self,
            "policy",
            _coerce_enum(self.policy, OverlapPolicy, name="policy"),
        )

    @property
    def has_overlap(self) -> bool:
        """Whether any voxel belongs to more than one object."""

        return self.n_overlap_voxels > 0

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation."""

        return {
            "n_overlap_voxels": self.n_overlap_voxels,
            "has_overlap": self.has_overlap,
            "object_pair_counts": tuple(dict(row) for row in self.object_pair_counts),
            "policy": self.policy.value,
        }


def detect_mask_overlaps(
    object_masks: Mapping[str, ArrayLike],
    *,
    shape: tuple[int, ...],
    policy: OverlapPolicy | str = OverlapPolicy.WARN,
) -> OverlapReport:
    """Detect pairwise overlaps among object masks."""
    names = tuple(object_masks.keys())
    masks = {
        name: validate_array_shape(
            mask, shape=shape, name=f"object_masks[{name!r}]"
        ).astype(bool)
        for name, mask in object_masks.items()
    }

    if not masks:
        return OverlapReport(policy=policy)

    occupancy = np.zeros(shape, dtype=np.int16)
    for mask in masks.values():
        occupancy += mask.astype(np.int16)

    n_overlap_voxels = int(np.sum(occupancy > 1))
    pair_rows: list[dict[str, object]] = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            count = int(np.sum(masks[left_name] & masks[right_name]))
            if count:
                pair_rows.append(
                    {
                        "object_id_a": left_name,
                        "object_id_b": right_name,
                        "n_overlap_voxels": count,
                    }
                )

    return OverlapReport(
        n_overlap_voxels=n_overlap_voxels,
        object_pair_counts=tuple(pair_rows),
        policy=policy,
    )


@dataclass(frozen=True)
class RenderedScene:
    """Rendered arrays, masks, scene truth, and provenance."""

    grid: GridSpec
    scalar_maps: Mapping[str, ArrayLike]
    label_map: ArrayLike
    object_masks: Mapping[str, ArrayLike]
    object_metadata: Mapping[str, SceneObjectMetadata]
    truth: SceneTruth = field(default_factory=SceneTruth)
    composition: CompositionRules = field(default_factory=CompositionRules)
    mask_rules: MaskRules = field(default_factory=MaskRules)
    target_masks: Mapping[str, ArrayLike] | None = None
    analysis_masks: Mapping[str, ArrayLike] | None = None
    skeleton_masks: Mapping[str, ArrayLike] | None = None
    centrelines: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    frames: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    distance_maps: Mapping[str, ArrayLike] = field(default_factory=dict)
    signed_offset_maps: Mapping[str, ArrayLike] = field(default_factory=dict)
    overlap_report: OverlapReport | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.grid.shape

        scalar_maps = {
            str(name): np.asarray(value, dtype=float)
            for name, value in self.scalar_maps.items()
        }
        if not scalar_maps:
            raise ValueError("RenderedScene requires at least one scalar map.")
        for name, value in scalar_maps.items():
            validate_array_shape(value, shape=shape, name=f"scalar_maps[{name!r}]")

        label_map = validate_array_shape(
            np.asarray(self.label_map),
            shape=shape,
            name="label_map",
        ).astype(np.int32)

        object_masks = {
            str(name): validate_array_shape(
                np.asarray(mask),
                shape=shape,
                name=f"object_masks[{name!r}]",
            ).astype(bool)
            for name, mask in self.object_masks.items()
        }

        object_metadata = {
            str(name): metadata for name, metadata in self.object_metadata.items()
        }
        missing_metadata = sorted(set(object_masks) - set(object_metadata))
        if missing_metadata:
            raise ValueError(
                f"Missing object metadata for object mask(s): {missing_metadata}."
            )
        metadata_without_masks = sorted(set(object_metadata) - set(object_masks))
        if metadata_without_masks:
            raise ValueError(
                "Object metadata exists without object mask(s): "
                f"{metadata_without_masks}."
            )

        target_masks = (
            self._derive_role_masks(
                object_masks,
                object_metadata,
                roles=self.mask_rules.target_roles,
                default_name="target",
            )
            if self.target_masks is None
            else self._validate_mask_mapping(self.target_masks, shape, "target_masks")
        )
        analysis_masks = (
            self._derive_role_masks(
                object_masks,
                object_metadata,
                roles=self.mask_rules.analysis_roles,
                default_name="analysis",
            )
            if self.analysis_masks is None
            else self._validate_mask_mapping(
                self.analysis_masks, shape, "analysis_masks"
            )
        )
        skeleton_masks = self._validate_mask_mapping(
            self.skeleton_masks or {},
            shape,
            "skeleton_masks",
        )

        distance_maps = {
            str(name): validate_array_shape(
                np.asarray(value, dtype=float),
                shape=shape,
                name=f"distance_maps[{name!r}]",
            )
            for name, value in self.distance_maps.items()
        }
        signed_offset_maps = {
            str(name): validate_array_shape(
                np.asarray(value, dtype=float),
                shape=shape,
                name=f"signed_offset_maps[{name!r}]",
            )
            for name, value in self.signed_offset_maps.items()
        }

        overlap_report = self.overlap_report or detect_mask_overlaps(
            object_masks,
            shape=shape,
            policy=self.composition.overlap_policy,
        )
        self._apply_overlap_policy(overlap_report)

        object.__setattr__(self, "scalar_maps", scalar_maps)
        object.__setattr__(self, "label_map", label_map)
        object.__setattr__(self, "object_masks", object_masks)
        object.__setattr__(self, "object_metadata", object_metadata)
        object.__setattr__(self, "target_masks", target_masks)
        object.__setattr__(self, "analysis_masks", analysis_masks)
        object.__setattr__(self, "skeleton_masks", skeleton_masks)
        object.__setattr__(self, "distance_maps", distance_maps)
        object.__setattr__(self, "signed_offset_maps", signed_offset_maps)
        object.__setattr__(self, "overlap_report", overlap_report)

    @staticmethod
    def _validate_mask_mapping(
        masks: Mapping[str, ArrayLike],
        shape: tuple[int, ...],
        name: str,
    ) -> dict[str, BoolArray]:
        """Validate a mapping of named masks."""

        return {
            str(mask_name): validate_array_shape(
                np.asarray(mask),
                shape=shape,
                name=f"{name}[{mask_name!r}]",
            ).astype(bool)
            for mask_name, mask in masks.items()
        }

    @staticmethod
    def _derive_role_masks(
        object_masks: Mapping[str, BoolArray],
        object_metadata: Mapping[str, SceneObjectMetadata],
        *,
        roles: Sequence[ObjectRole],
        default_name: str,
    ) -> dict[str, BoolArray]:
        """Derive one combined mask from objects matching the requested roles."""

        first_mask = next(iter(object_masks.values()), None)
        if first_mask is None:
            raise ValueError("At least one object mask is required.")
        combined = np.zeros(first_mask.shape, dtype=bool)
        role_set = set(roles)
        for object_id, mask in object_masks.items():
            if object_metadata[object_id].role in role_set:
                combined |= mask
        return {default_name: combined}

    @staticmethod
    def _apply_overlap_policy(overlap_report: OverlapReport) -> None:
        """Apply the configured overlap policy."""

        if not overlap_report.has_overlap:
            return
        if overlap_report.policy is OverlapPolicy.ERROR:
            raise ValueError(
                "Object masks overlap and overlap_policy='error'. "
                f"Overlap voxels: {overlap_report.n_overlap_voxels}."
            )
        if overlap_report.policy is OverlapPolicy.WARN:
            warnings.warn(
                "Object masks overlap. Details are recorded in overlap_report.",
                UserWarning,
                stacklevel=3,
            )

    def object_ids_by_role(self, role: ObjectRole | str) -> tuple[str, ...]:
        """Return object IDs matching a role."""

        role_value = _coerce_enum(role, ObjectRole, name="role")
        return tuple(
            object_id
            for object_id, metadata in self.object_metadata.items()
            if metadata.role is role_value
        )

    def combined_object_mask(self, object_ids: Sequence[str]) -> BoolArray:
        """Return the union of selected object masks."""

        combined = np.zeros(self.grid.shape, dtype=bool)
        for object_id in object_ids:
            if object_id not in self.object_masks:
                raise KeyError(f"Unknown object_id: {object_id!r}.")
            combined |= self.object_masks[object_id]
        return combined

    def summary(self) -> dict[str, object]:
        """Return a compact scene summary."""

        return {
            "shape": self.grid.shape,
            "spacing": self.grid.spacing,
            "n_scalar_maps": len(self.scalar_maps),
            "scalar_maps": tuple(self.scalar_maps.keys()),
            "n_objects": len(self.object_masks),
            "object_ids": tuple(self.object_masks.keys()),
            "target_mask_voxels": {
                name: int(np.sum(mask)) for name, mask in self.target_masks.items()
            },
            "analysis_mask_voxels": {
                name: int(np.sum(mask)) for name, mask in self.analysis_masks.items()
            },
            "overlap": self.overlap_report.to_dict(),
            "composition": {
                "label_mode": self.composition.label_mode.value,
                "scalar_blend": self.composition.scalar_blend.value,
                "overlap_policy": self.composition.overlap_policy.value,
            },
        }
