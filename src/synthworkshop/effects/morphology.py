"""Known morphology and profile-support effects."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from synthworkshop.effects.base import (
    EffectRecord,
    attach_effect_record,
    require_object_mask,
    require_scalar_map,
    validate_finite_float,
    validate_positive_int,
)
from synthworkshop.scenes import RenderedScene


def expand_object_width(
    scene: RenderedScene,
    *,
    object_id: str,
    iterations: int = 1,
) -> RenderedScene:
    """Expand one object mask by binary dilation."""

    n_iter = validate_positive_int(iterations, name="iterations")
    original = require_object_mask(scene, object_id)

    expanded = original.copy()
    for _ in range(n_iter):
        expanded = _dilate_once(expanded)

    added = expanded & ~original
    updated_scene = _replace_object_mask(scene, object_id=object_id, new_mask=expanded)

    record = EffectRecord(
        name="width_expansion",
        target="object_masks",
        parameters={
            "object_id": object_id,
            "iterations": n_iter,
        },
        affected_objects=(object_id,),
        support_voxels=int(added.sum()),
        magnitude=float(n_iter),
        expected_direction="increase",
        clean_null=not bool(added.any()),
        truth_changed=bool(added.any()),
    )
    return attach_effect_record(updated_scene, record)


def contract_object_width(
    scene: RenderedScene,
    *,
    object_id: str,
    iterations: int = 1,
) -> RenderedScene:
    """Contract one object mask by binary erosion."""

    n_iter = validate_positive_int(iterations, name="iterations")
    original = require_object_mask(scene, object_id)

    contracted = original.copy()
    for _ in range(n_iter):
        contracted = _erode_once(contracted)

    removed = original & ~contracted
    updated_scene = _replace_object_mask(
        scene,
        object_id=object_id,
        new_mask=contracted,
    )

    record = EffectRecord(
        name="width_contraction",
        target="object_masks",
        parameters={
            "object_id": object_id,
            "iterations": n_iter,
        },
        affected_objects=(object_id,),
        support_voxels=int(removed.sum()),
        magnitude=float(n_iter),
        expected_direction="decrease",
        clean_null=not bool(removed.any()),
        truth_changed=bool(removed.any()),
    )
    return attach_effect_record(updated_scene, record)


def add_rim_enhancement(
    scene: RenderedScene,
    *,
    object_id: str,
    map_name: str,
    delta: float,
    erosion_iterations: int = 1,
) -> RenderedScene:
    """Add a known scalar change to an object's rim or edge support."""

    return _add_profile_support_shift(
        scene,
        object_id=object_id,
        map_name=map_name,
        delta=delta,
        erosion_iterations=erosion_iterations,
        support_name="rim",
        effect_name="rim_enhancement",
    )


def add_hollow_core_change(
    scene: RenderedScene,
    *,
    object_id: str,
    map_name: str,
    delta: float,
    erosion_iterations: int = 1,
) -> RenderedScene:
    """Add a known scalar change to an object's eroded core support."""

    return _add_profile_support_shift(
        scene,
        object_id=object_id,
        map_name=map_name,
        delta=delta,
        erosion_iterations=erosion_iterations,
        support_name="core",
        effect_name="hollow_core_change",
    )


def _add_profile_support_shift(
    scene: RenderedScene,
    *,
    object_id: str,
    map_name: str,
    delta: float,
    erosion_iterations: int,
    support_name: str,
    effect_name: str,
) -> RenderedScene:
    delta_value = validate_finite_float(delta, name="delta")
    n_erode = validate_positive_int(erosion_iterations, name="erosion_iterations")

    object_mask = require_object_mask(scene, object_id)
    scalar = require_scalar_map(scene, map_name)

    eroded = object_mask.copy()
    for _ in range(n_erode):
        eroded = _erode_once(eroded)

    if support_name == "core":
        support = eroded if eroded.any() else object_mask.copy()
    elif support_name == "rim":
        support = object_mask & ~eroded
        if not support.any():
            support = object_mask.copy()
    else:  # pragma: no cover
        raise ValueError(f"Unknown support_name: {support_name!r}.")

    shifted = scalar.copy()
    shifted[support] = shifted[support] + delta_value

    scalar_maps = dict(scene.scalar_maps)
    scalar_maps[map_name] = shifted

    record = EffectRecord(
        name=effect_name,
        target="scalar_maps",
        parameters={
            "object_id": object_id,
            "map_name": map_name,
            "delta": delta_value,
            "support": support_name,
            "erosion_iterations": n_erode,
        },
        affected_maps=(map_name,),
        affected_objects=(object_id,),
        support_voxels=int(support.sum()),
        magnitude=abs(delta_value),
        expected_direction=_direction(delta_value),
        clean_null=delta_value == 0.0,
        truth_changed=delta_value != 0.0,
    )
    return attach_effect_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )


def _replace_object_mask(
    scene: RenderedScene,
    *,
    object_id: str,
    new_mask: np.ndarray,
) -> RenderedScene:
    metadata = scene.object_metadata.get(object_id)
    if metadata is None:
        raise ValueError(f"Missing object metadata for object_id: {object_id!r}.")

    old_mask = require_object_mask(scene, object_id)
    object_masks = dict(scene.object_masks)
    object_masks[object_id] = np.asarray(new_mask, dtype=bool)

    label_map = np.asarray(scene.label_map).copy()
    label = int(metadata.label)

    removed = old_mask & ~object_masks[object_id]
    added = object_masks[object_id] & ~old_mask

    label_map[removed & (label_map == label)] = 0
    label_map[added] = label

    return replace(
        scene,
        object_masks=object_masks,
        label_map=label_map,
        target_masks=None,
        analysis_masks=None,
    )


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)

    for offset in product(range(3), repeat=mask.ndim):
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(offset, mask.shape, strict=True)
        )
        out |= padded[slices]

    return out


def _erode_once(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)

    for offset in product(range(3), repeat=mask.ndim):
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(offset, mask.shape, strict=True)
        )
        out &= padded[slices]

    return out


def _direction(delta: float) -> str:
    if delta > 0:
        return "increase"
    if delta < 0:
        return "decrease"
    return "none"
