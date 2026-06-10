"""Composition of rendered object scenes into multi-object scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from synthworkshop.coordinates import BoolArray
from synthworkshop.ground_truth import SceneTruth
from synthworkshop.scenes.model import (
    CompositionRules,
    LabelMode,
    MaskRules,
    RenderedScene,
    ScalarBlend,
    SceneObjectMetadata,
    detect_mask_overlaps,
)


@dataclass(frozen=True)
class ObjectContribution:
    """One object contribution from a source rendered scene."""

    source_scene_index: int
    object_id: str
    metadata: SceneObjectMetadata
    mask: BoolArray
    source_scene: RenderedScene


def _coerce_composition(
    composition: CompositionRules | None,
) -> CompositionRules:
    """Return explicit composition rules."""

    return CompositionRules() if composition is None else composition


def _coerce_mask_rules(mask_rules: MaskRules | None) -> MaskRules:
    """Return explicit mask derivation rules."""

    return MaskRules() if mask_rules is None else mask_rules


def _check_source_scenes(scenes: Sequence[RenderedScene]) -> tuple[int, ...]:
    """Validate scene list and return common shape."""

    if not scenes:
        raise ValueError("At least one RenderedScene is required.")
    first = scenes[0].grid
    for index, scene in enumerate(scenes[1:], start=1):
        if scene.grid.shape != first.shape:
            raise ValueError(f"Scene {index} grid shape does not match scene 0.")
        if scene.grid.spacing != first.spacing:
            raise ValueError(f"Scene {index} grid spacing does not match scene 0.")
        if scene.grid.origin != first.origin:
            raise ValueError(f"Scene {index} grid origin does not match scene 0.")
        if scene.grid.axis_names != first.axis_names:
            raise ValueError(f"Scene {index} axis names do not match scene 0.")
    return first.shape


def _collect_contributions(
    scenes: Sequence[RenderedScene],
) -> tuple[ObjectContribution, ...]:
    """Collect object contributions and enforce unique object IDs."""

    contributions: list[ObjectContribution] = []
    seen: set[str] = set()

    for scene_index, scene in enumerate(scenes):
        for object_id, mask in scene.object_masks.items():
            if object_id in seen:
                raise ValueError(f"Duplicate object_id across scenes: {object_id!r}.")
            seen.add(object_id)
            contributions.append(
                ObjectContribution(
                    source_scene_index=scene_index,
                    object_id=object_id,
                    metadata=scene.object_metadata[object_id],
                    mask=mask,
                    source_scene=scene,
                )
            )

    if not contributions:
        raise ValueError("No object contributions were found.")
    return tuple(contributions)


def _map_names(scenes: Sequence[RenderedScene]) -> tuple[str, ...]:
    """Return scalar map names in first-seen order."""

    names: list[str] = []
    for scene in scenes:
        for name in scene.scalar_maps:
            if name not in names:
                names.append(name)
    return tuple(names)


def _compose_label_map(
    contributions: Sequence[ObjectContribution],
    *,
    shape: tuple[int, ...],
    label_mode: LabelMode,
) -> np.ndarray:
    """Compose a label map from object masks and metadata."""

    label_map = np.zeros(shape, dtype=np.int32)

    if label_mode is LabelMode.FIRST:
        occupied = np.zeros(shape, dtype=bool)
        for contribution in contributions:
            fill = contribution.mask & ~occupied
            label_map[fill] = contribution.metadata.label
            occupied |= contribution.mask
        return label_map

    if label_mode is LabelMode.LAST:
        for contribution in contributions:
            label_map[contribution.mask] = contribution.metadata.label
        return label_map

    if label_mode is LabelMode.PRIORITY:
        priority_map = np.full(shape, -np.inf, dtype=float)
        for contribution in contributions:
            priority = float(contribution.metadata.priority)
            fill = contribution.mask & (priority > priority_map)
            label_map[fill] = contribution.metadata.label
            priority_map[fill] = priority
        return label_map

    raise ValueError(f"Unsupported label_mode: {label_mode!r}.")


def _compose_scalar_map(
    map_name: str,
    contributions: Sequence[ObjectContribution],
    *,
    shape: tuple[int, ...],
    scalar_blend: ScalarBlend,
    scalar_weights: Mapping[str, float] | None,
) -> np.ndarray:
    """Compose one scalar map using explicit blending semantics."""

    out = np.zeros(shape, dtype=float)

    if scalar_blend is ScalarBlend.OVERWRITE:
        for contribution in contributions:
            scene = contribution.source_scene
            if map_name not in scene.scalar_maps:
                continue
            values = scene.scalar_maps[map_name]
            out[contribution.mask] = values[contribution.mask]
        return out

    if scalar_blend is ScalarBlend.SUM:
        for contribution in contributions:
            scene = contribution.source_scene
            if map_name not in scene.scalar_maps:
                continue
            values = scene.scalar_maps[map_name]
            out[contribution.mask] += values[contribution.mask]
        return out

    if scalar_blend is ScalarBlend.MAX:
        filled = np.zeros(shape, dtype=bool)
        for contribution in contributions:
            scene = contribution.source_scene
            if map_name not in scene.scalar_maps:
                continue
            values = scene.scalar_maps[map_name]
            first = contribution.mask & ~filled
            later = contribution.mask & filled
            out[first] = values[first]
            out[later] = np.maximum(out[later], values[later])
            filled |= contribution.mask
        return out

    if scalar_blend is ScalarBlend.WEIGHTED_MEAN:
        numerator = np.zeros(shape, dtype=float)
        denominator = np.zeros(shape, dtype=float)
        weights = dict(scalar_weights or {})
        for contribution in contributions:
            scene = contribution.source_scene
            if map_name not in scene.scalar_maps:
                continue
            weight = float(weights.get(contribution.object_id, 1.0))
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("scalar_weights must contain finite positive values.")
            values = scene.scalar_maps[map_name]
            numerator[contribution.mask] += weight * values[contribution.mask]
            denominator[contribution.mask] += weight
        valid = denominator > 0
        out[valid] = numerator[valid] / denominator[valid]
        return out

    raise ValueError(f"Unsupported scalar_blend: {scalar_blend!r}.")


def _compose_scalar_maps(
    scenes: Sequence[RenderedScene],
    contributions: Sequence[ObjectContribution],
    *,
    shape: tuple[int, ...],
    scalar_blend: ScalarBlend,
    scalar_weights: Mapping[str, float] | None,
) -> dict[str, np.ndarray]:
    """Compose all scalar maps in the source scenes."""

    return {
        map_name: _compose_scalar_map(
            map_name,
            contributions,
            shape=shape,
            scalar_blend=scalar_blend,
            scalar_weights=scalar_weights,
        )
        for map_name in _map_names(scenes)
    }


def _object_metadata(
    contributions: Sequence[ObjectContribution],
) -> dict[str, SceneObjectMetadata]:
    """Return composed object metadata."""

    return {
        contribution.object_id: contribution.metadata for contribution in contributions
    }


def _object_masks(
    contributions: Sequence[ObjectContribution],
) -> dict[str, BoolArray]:
    """Return composed object masks."""

    return {contribution.object_id: contribution.mask for contribution in contributions}


def _object_table(contributions: Sequence[ObjectContribution]) -> pd.DataFrame:
    """Create a composition object table."""

    return pd.DataFrame(
        [
            {
                "source_scene_index": contribution.source_scene_index,
                "object_id": contribution.object_id,
                "role": contribution.metadata.role.value,
                "label": contribution.metadata.label,
                "priority": contribution.metadata.priority,
                "mask_voxels": int(np.sum(contribution.mask)),
                "name": contribution.metadata.name,
                "description": contribution.metadata.description,
            }
            for contribution in contributions
        ]
    )


def _overlap_table(overlap_report) -> pd.DataFrame:
    """Create an overlap table from an OverlapReport."""

    if not overlap_report.object_pair_counts:
        return pd.DataFrame(columns=["object_id_a", "object_id_b", "n_overlap_voxels"])
    return pd.DataFrame.from_records(overlap_report.object_pair_counts)


def _merge_truth_tables(
    scenes: Sequence[RenderedScene],
    *,
    object_table: pd.DataFrame,
    overlap_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Merge source truth tables and add composition tables."""

    tables: dict[str, list[pd.DataFrame]] = {}
    for scene_index, scene in enumerate(scenes):
        for table_name, table in scene.truth.tables.items():
            work = table.copy()
            if "source_scene_index" not in work.columns:
                work.insert(0, "source_scene_index", scene_index)
            tables.setdefault(table_name, []).append(work)

    merged = {
        table_name: pd.concat(frames, axis=0, ignore_index=True)
        for table_name, frames in tables.items()
    }
    merged["objects"] = object_table
    merged["overlaps"] = overlap_table
    return merged


def _compose_truth(
    scenes: Sequence[RenderedScene],
    contributions: Sequence[ObjectContribution],
    *,
    composition: CompositionRules,
    scalar_maps: Mapping[str, np.ndarray],
    overlap_report,
) -> SceneTruth:
    """Create method-agnostic truth metadata for the composed scene."""

    geometric: dict[str, Any] = {}
    perturbations: dict[str, Any] = {}
    for scene_index, scene in enumerate(scenes):
        for key, value in scene.truth.geometric.items():
            geometric[str(key)] = value
        if scene.truth.perturbations:
            perturbations[f"source_scene_{scene_index}"] = dict(
                scene.truth.perturbations
            )

    objects = {
        contribution.object_id: {
            "source_scene_index": contribution.source_scene_index,
            "role": contribution.metadata.role.value,
            "label": contribution.metadata.label,
            "priority": contribution.metadata.priority,
            "mask_voxels": int(np.sum(contribution.mask)),
        }
        for contribution in contributions
    }

    scalar_fields = {
        map_name: {
            "blend_rule": composition.scalar_blend.value,
            "contributing_objects": [
                contribution.object_id
                for contribution in contributions
                if map_name in contribution.source_scene.scalar_maps
            ],
        }
        for map_name in scalar_maps
    }

    object_table = _object_table(contributions)
    overlap_table = _overlap_table(overlap_report)
    tables = _merge_truth_tables(
        scenes,
        object_table=object_table,
        overlap_table=overlap_table,
    )

    return SceneTruth(
        geometric=geometric,
        objects=objects,
        scalar_fields=scalar_fields,
        perturbations=perturbations,
        tables=tables,
        metadata={
            "truth_scope": (
                "Method-agnostic composition truth: object roles, labels, masks, "
                "scalar blend rules, source-scene provenance, and overlap report."
            ),
            "n_source_scenes": len(scenes),
            "composition": {
                "label_mode": composition.label_mode.value,
                "scalar_blend": composition.scalar_blend.value,
                "overlap_policy": composition.overlap_policy.value,
            },
        },
    )


def compose_rendered_scenes(
    scenes: Sequence[RenderedScene],
    *,
    composition: CompositionRules | None = None,
    mask_rules: MaskRules | None = None,
    scalar_weights: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> RenderedScene:
    """Compose rendered object scenes into one multi-object RenderedScene."""

    rules = _coerce_composition(composition)
    masks = _coerce_mask_rules(mask_rules)
    shape = _check_source_scenes(scenes)
    contributions = _collect_contributions(scenes)

    object_masks = _object_masks(contributions)
    object_metadata = _object_metadata(contributions)
    overlap_report = detect_mask_overlaps(
        object_masks,
        shape=shape,
        policy=rules.overlap_policy,
    )

    label_map = _compose_label_map(
        contributions,
        shape=shape,
        label_mode=rules.label_mode,
    )
    scalar_maps = _compose_scalar_maps(
        scenes,
        contributions,
        shape=shape,
        scalar_blend=rules.scalar_blend,
        scalar_weights=scalar_weights,
    )

    truth = _compose_truth(
        scenes,
        contributions,
        composition=rules,
        scalar_maps=scalar_maps,
        overlap_report=overlap_report,
    )

    skeleton_masks = {}
    centrelines = {}
    frames = {}
    distance_maps = {}
    signed_offset_maps = {}

    for scene in scenes:
        skeleton_masks.update(scene.skeleton_masks)
        centrelines.update(scene.centrelines)
        frames.update(scene.frames)
        distance_maps.update(scene.distance_maps)
        signed_offset_maps.update(scene.signed_offset_maps)

    return RenderedScene(
        grid=scenes[0].grid,
        scalar_maps=scalar_maps,
        label_map=label_map,
        object_masks=object_masks,
        object_metadata=object_metadata,
        truth=truth,
        composition=rules,
        mask_rules=masks,
        skeleton_masks=skeleton_masks,
        centrelines=centrelines,
        frames=frames,
        distance_maps=distance_maps,
        signed_offset_maps=signed_offset_maps,
        overlap_report=overlap_report,
        metadata={
            "renderer": "compose_rendered_scenes",
            "n_source_scenes": len(scenes),
            **dict(metadata or {}),
        },
        provenance={
            "package": "synthworkshop",
            "stage": "M1f",
            **dict(provenance or {}),
        },
    )
