"""Known localised scalar effects."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from synthworkshop.effects.base import (
    EffectRecord,
    attach_effect_record,
    require_object_mask,
    require_scalar_map,
    validate_finite_float,
)
from synthworkshop.scenes import RenderedScene


def add_axis_interval_value_shift(
    scene: RenderedScene,
    *,
    object_id: str,
    map_name: str,
    delta: float,
    axis: int,
    start_mm: float,
    end_mm: float,
) -> RenderedScene:
    """Add a scalar shift within an object and physical axis interval."""

    axis_index = int(axis)
    if axis_index < 0 or axis_index >= scene.grid.ndim:
        raise ValueError(f"axis must be in [0, {scene.grid.ndim - 1}].")

    start = validate_finite_float(start_mm, name="start_mm")
    end = validate_finite_float(end_mm, name="end_mm")
    if end < start:
        raise ValueError("end_mm must be greater than or equal to start_mm.")

    delta_value = validate_finite_float(delta, name="delta")
    object_mask = require_object_mask(scene, object_id)
    scalar = require_scalar_map(scene, map_name)

    axis_support = _axis_interval_mask(
        scene.grid.shape,
        spacing=scene.grid.spacing,
        origin=scene.grid.origin,
        axis=axis_index,
        start_mm=start,
        end_mm=end,
    )
    support = object_mask & axis_support

    shifted = scalar.copy()
    shifted[support] = shifted[support] + delta_value

    scalar_maps = dict(scene.scalar_maps)
    scalar_maps[map_name] = shifted

    record = EffectRecord(
        name="axis_interval_value_shift",
        target="scalar_maps",
        parameters={
            "object_id": object_id,
            "map_name": map_name,
            "delta": delta_value,
            "axis": axis_index,
            "start_mm": start,
            "end_mm": end,
        },
        affected_maps=(map_name,),
        affected_objects=(object_id,),
        support_voxels=int(support.sum()),
        magnitude=abs(delta_value),
        expected_direction=_direction(delta_value),
        clean_null=delta_value == 0.0 or not bool(support.any()),
        truth_changed=delta_value != 0.0 and bool(support.any()),
        note="Axis-local support is defined in physical grid coordinates.",
    )

    return attach_effect_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )


def add_branch_value_shift(
    scene: RenderedScene,
    *,
    map_name: str,
    delta: float,
    branch_object_id: str | None = None,
    graph_object_id: str | None = None,
    edge_id: str | None = None,
) -> RenderedScene:
    """Add a scalar shift to one rendered graph-edge object."""

    object_id = _branch_object_id(
        branch_object_id=branch_object_id,
        graph_object_id=graph_object_id,
        edge_id=edge_id,
    )

    delta_value = validate_finite_float(delta, name="delta")
    object_mask = require_object_mask(scene, object_id)
    scalar = require_scalar_map(scene, map_name)

    shifted = scalar.copy()
    shifted[object_mask] = shifted[object_mask] + delta_value

    scalar_maps = dict(scene.scalar_maps)
    scalar_maps[map_name] = shifted

    record = EffectRecord(
        name="branch_value_shift",
        target="scalar_maps",
        parameters={
            "branch_object_id": object_id,
            "graph_object_id": graph_object_id,
            "edge_id": edge_id,
            "map_name": map_name,
            "delta": delta_value,
        },
        affected_maps=(map_name,),
        affected_objects=(object_id,),
        support_voxels=int(object_mask.sum()),
        magnitude=abs(delta_value),
        expected_direction=_direction(delta_value),
        clean_null=delta_value == 0.0 or not bool(object_mask.any()),
        truth_changed=delta_value != 0.0 and bool(object_mask.any()),
        note=(
            "Branch support is defined by the rendered graph-edge object mask. "
            "Graph geometry is not modified."
        ),
    )

    return attach_effect_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )


def _axis_interval_mask(
    shape: tuple[int, ...],
    *,
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    axis: int,
    start_mm: float,
    end_mm: float,
) -> np.ndarray:
    coordinates = origin[axis] + np.arange(shape[axis], dtype=float) * spacing[axis]
    axis_line = (coordinates >= start_mm) & (coordinates <= end_mm)

    view_shape = [1] * len(shape)
    view_shape[axis] = shape[axis]
    return np.broadcast_to(axis_line.reshape(view_shape), shape)


def _branch_object_id(
    *,
    branch_object_id: str | None,
    graph_object_id: str | None,
    edge_id: str | None,
) -> str:
    if branch_object_id is not None:
        object_id = str(branch_object_id)
    else:
        if graph_object_id is None or edge_id is None:
            raise ValueError(
                "Provide either branch_object_id or both graph_object_id and edge_id."
            )
        object_id = f"{graph_object_id}__edge__{edge_id}"

    if not object_id:
        raise ValueError("branch object_id must be a non-empty string.")
    return object_id


def _direction(delta: float) -> str:
    if delta > 0:
        return "increase"
    if delta < 0:
        return "decrease"
    return "none"
