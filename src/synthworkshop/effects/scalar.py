"""Known scalar-effect injection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import product

import numpy as np

from synthworkshop.effects.base import (
    EffectRecord,
    attach_effect_record,
    require_object_mask,
    require_scalar_map,
    selected_object_ids,
    validate_finite_float,
    validate_positive_int,
)
from synthworkshop.scenes import RenderedScene


def inject_no_effect(
    scene: RenderedScene,
    *,
    name: str = "no_effect_null",
    note: str | None = "Explicit no-effect null.",
) -> RenderedScene:
    """Record an explicit null effect without changing arrays."""

    record = EffectRecord(
        name=name,
        target="scene",
        parameters={},
        support_voxels=0,
        magnitude=0.0,
        expected_direction="none",
        clean_null=True,
        truth_changed=False,
        note=note,
    )
    return attach_effect_record(scene, record)


def add_object_value_shift(
    scene: RenderedScene,
    *,
    object_id: str,
    map_name: str,
    delta: float,
) -> RenderedScene:
    """Add a known scalar shift across one object mask."""

    return _add_scalar_shift(
        scene,
        object_id=object_id,
        map_name=map_name,
        delta=delta,
        support="object",
        erosion_iterations=1,
        effect_name="object_value_shift",
    )


def add_centre_value_shift(
    scene: RenderedScene,
    *,
    object_id: str,
    map_name: str,
    delta: float,
    erosion_iterations: int = 1,
) -> RenderedScene:
    """Add a known scalar shift to the eroded centre of an object mask."""

    return _add_scalar_shift(
        scene,
        object_id=object_id,
        map_name=map_name,
        delta=delta,
        support="centre",
        erosion_iterations=erosion_iterations,
        effect_name="centre_value_shift",
    )


def add_edge_value_shift(
    scene: RenderedScene,
    *,
    object_id: str,
    map_name: str,
    delta: float,
    erosion_iterations: int = 1,
) -> RenderedScene:
    """Add a known scalar shift to the boundary region of an object mask."""

    return _add_scalar_shift(
        scene,
        object_id=object_id,
        map_name=map_name,
        delta=delta,
        support="edge",
        erosion_iterations=erosion_iterations,
        effect_name="edge_value_shift",
    )


def add_multi_object_value_shift(
    scene: RenderedScene,
    *,
    object_ids: Sequence[str] | None,
    map_name: str,
    delta: float,
) -> RenderedScene:
    """Add a known scalar shift across several object masks."""

    selected = selected_object_ids(scene, object_ids)
    current = scene
    for object_id in selected:
        current = add_object_value_shift(
            current,
            object_id=object_id,
            map_name=map_name,
            delta=delta,
        )
    return current


def _add_scalar_shift(
    scene: RenderedScene,
    *,
    object_id: str,
    map_name: str,
    delta: float,
    support: str,
    erosion_iterations: int,
    effect_name: str,
) -> RenderedScene:
    delta_value = validate_finite_float(delta, name="delta")
    n_erode = validate_positive_int(erosion_iterations, name="erosion_iterations")

    object_mask = require_object_mask(scene, object_id)
    scalar = require_scalar_map(scene, map_name)

    support_mask = _support_mask(
        object_mask,
        support=support,
        erosion_iterations=n_erode,
    )
    shifted = scalar.copy()
    shifted[support_mask] = shifted[support_mask] + delta_value

    scalar_maps = dict(scene.scalar_maps)
    scalar_maps[map_name] = shifted

    record = EffectRecord(
        name=effect_name,
        target="scalar_maps",
        parameters={
            "object_id": object_id,
            "map_name": map_name,
            "delta": delta_value,
            "support": support,
            "erosion_iterations": n_erode,
        },
        affected_maps=(map_name,),
        affected_objects=(object_id,),
        support_voxels=int(support_mask.sum()),
        magnitude=abs(delta_value),
        expected_direction=_direction(delta_value),
        clean_null=delta_value == 0.0,
        truth_changed=delta_value != 0.0,
    )

    return attach_effect_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )


def _support_mask(
    object_mask: np.ndarray,
    *,
    support: str,
    erosion_iterations: int,
) -> np.ndarray:
    if support == "object":
        return object_mask.copy()

    eroded = object_mask.copy()
    for _ in range(erosion_iterations):
        eroded = _erode_once(eroded)

    if support == "centre":
        if eroded.any():
            return eroded
        return object_mask.copy()

    if support == "edge":
        edge = object_mask & ~eroded
        if edge.any():
            return edge
        return object_mask.copy()

    raise ValueError(f"Unknown scalar-effect support: {support!r}.")


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
