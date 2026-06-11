"""Intensity perturbations for rendered scalar maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.perturbations.base import (
    PerturbationRecord,
    attach_perturbation_record,
    selected_names,
)
from synthworkshop.scenes import RenderedScene


def scale_scalar_maps(
    scene: RenderedScene,
    *,
    factor: float,
    map_names: Sequence[str] | None = None,
    mask: ArrayLike | None = None,
) -> RenderedScene:
    """Multiply selected scalar maps by a constant factor."""

    factor_value = _validate_finite_float(factor, name="factor")
    selected = selected_names(scene.scalar_maps, map_names, label="scalar map")
    mask_array = _optional_mask(mask, shape=scene.grid.shape)

    scalar_maps = dict(scene.scalar_maps)
    affected: list[str] = []

    for name in selected:
        array = np.asarray(scene.scalar_maps[name], dtype=float)
        if mask_array is None:
            scalar_maps[name] = array * factor_value
        else:
            scalar_maps[name] = np.where(mask_array, array * factor_value, array)
        affected.append(f"scalar_maps:{name}")

    record = PerturbationRecord(
        name="intensity_scaling",
        target="scalar_maps",
        parameters={
            "factor": factor_value,
            "map_names": selected,
            "masked": mask_array is not None,
        },
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(affected),
    )

    return attach_perturbation_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )


def add_linear_bias_field(
    scene: RenderedScene,
    *,
    axis: int,
    start_scale: float,
    end_scale: float,
    map_names: Sequence[str] | None = None,
    mask: ArrayLike | None = None,
) -> RenderedScene:
    """Apply a multiplicative linear bias-like field."""

    axis_index = int(axis)
    if axis_index < 0 or axis_index >= scene.grid.ndim:
        raise ValueError(f"axis must be in [0, {scene.grid.ndim - 1}].")

    start = _validate_finite_float(start_scale, name="start_scale")
    end = _validate_finite_float(end_scale, name="end_scale")
    selected = selected_names(scene.scalar_maps, map_names, label="scalar map")
    mask_array = _optional_mask(mask, shape=scene.grid.shape)

    bias = _linear_field(scene.grid.shape, axis=axis_index, start=start, end=end)
    scalar_maps = dict(scene.scalar_maps)
    affected: list[str] = []

    for name in selected:
        array = np.asarray(scene.scalar_maps[name], dtype=float)
        if mask_array is None:
            scalar_maps[name] = array * bias
        else:
            scalar_maps[name] = np.where(mask_array, array * bias, array)
        affected.append(f"scalar_maps:{name}")

    record = PerturbationRecord(
        name="linear_bias_field",
        target="scalar_maps",
        parameters={
            "axis": axis_index,
            "start_scale": start,
            "end_scale": end,
            "map_names": selected,
            "masked": mask_array is not None,
        },
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(affected),
        note=(
            "This is a simple analytic multiplicative field, not an MRI "
            "acquisition or bias-correction model."
        ),
    )

    return attach_perturbation_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )


def _linear_field(
    shape: tuple[int, ...],
    *,
    axis: int,
    start: float,
    end: float,
) -> np.ndarray:
    values = np.linspace(start, end, shape[axis], dtype=float)
    view_shape = [1] * len(shape)
    view_shape[axis] = shape[axis]
    return np.broadcast_to(values.reshape(view_shape), shape)


def _optional_mask(
    mask: ArrayLike | None, *, shape: tuple[int, ...]
) -> np.ndarray | None:
    if mask is None:
        return None

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != shape:
        raise ValueError("mask must match the scene grid shape.")
    return mask_array


def _validate_finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out
