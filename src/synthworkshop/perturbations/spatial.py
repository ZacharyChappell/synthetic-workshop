"""Spatial perturbations for rendered scene arrays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import product

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.perturbations.base import (
    PerturbationRecord,
    attach_perturbation_record,
    selected_names,
    validate_positive_int,
    validate_shift,
)
from synthworkshop.scenes import RenderedScene


def shift_array_integer(
    array: ArrayLike,
    shift_voxels: Sequence[int],
    *,
    fill_value: float | int | bool = 0,
) -> np.ndarray:
    """Shift an array by integer voxels without wrap-around."""

    source = np.asarray(array)
    shift = validate_shift(shift_voxels, ndim=source.ndim)

    result = np.full(source.shape, fill_value, dtype=source.dtype)
    source_slices = []
    target_slices = []

    for axis, offset in enumerate(shift):
        size = source.shape[axis]
        if abs(offset) >= size:
            return result

        if offset >= 0:
            source_slices.append(slice(0, size - offset))
            target_slices.append(slice(offset, size))
        else:
            source_slices.append(slice(-offset, size))
            target_slices.append(slice(0, size + offset))

    result[tuple(target_slices)] = source[tuple(source_slices)]
    return result


def shift_scalar_maps(
    scene: RenderedScene,
    *,
    shift_voxels: Sequence[int],
    map_names: Sequence[str] | None = None,
    fill_value: float = 0.0,
) -> RenderedScene:
    """Shift selected scalar maps by integer voxels."""

    shift = validate_shift(shift_voxels, ndim=scene.grid.ndim)
    selected = selected_names(scene.scalar_maps, map_names, label="scalar map")

    scalar_maps = dict(scene.scalar_maps)
    affected: list[str] = []

    for name in selected:
        scalar_maps[name] = shift_array_integer(
            scene.scalar_maps[name],
            shift,
            fill_value=fill_value,
        )
        affected.append(f"scalar_maps:{name}")

    record = PerturbationRecord(
        name="integer_scalar_shift",
        target="scalar_maps",
        parameters={
            "shift_voxels": shift,
            "map_names": selected,
            "fill_value": float(fill_value),
        },
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(affected),
    )

    return attach_perturbation_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )


def mean_blur_scalar_maps(
    scene: RenderedScene,
    *,
    radius_voxels: int = 1,
    iterations: int = 1,
    map_names: Sequence[str] | None = None,
) -> RenderedScene:
    """Apply a simple local mean blur to selected scalar maps."""

    radius = validate_positive_int(radius_voxels, name="radius_voxels")
    n_iter = validate_positive_int(iterations, name="iterations")
    selected = selected_names(scene.scalar_maps, map_names, label="scalar map")

    scalar_maps = dict(scene.scalar_maps)
    affected: list[str] = []

    for name in selected:
        array = np.asarray(scene.scalar_maps[name], dtype=float)
        for _ in range(n_iter):
            array = _mean_filter(array, radius=radius)
        scalar_maps[name] = array
        affected.append(f"scalar_maps:{name}")

    record = PerturbationRecord(
        name="mean_blur",
        target="scalar_maps",
        parameters={
            "radius_voxels": radius,
            "iterations": n_iter,
            "map_names": selected,
        },
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(affected),
    )

    return attach_perturbation_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )


def _mean_filter(array: np.ndarray, *, radius: int) -> np.ndarray:
    """Mean filter implemented with NumPy only."""

    padded = np.pad(array, pad_width=radius, mode="edge")
    out = np.zeros_like(array, dtype=float)
    n_terms = 0

    offsets = range(2 * radius + 1)
    for offset in product(offsets, repeat=array.ndim):
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(offset, array.shape, strict=True)
        )
        out += padded[slices]
        n_terms += 1

    return out / float(n_terms)
