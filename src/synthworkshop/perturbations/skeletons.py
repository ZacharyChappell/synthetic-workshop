"""Skeleton-mask perturbations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from synthworkshop.perturbations.base import (
    PerturbationRecord,
    attach_perturbation_record,
    selected_names,
    validate_positive_int,
    validate_shift,
)
from synthworkshop.perturbations.spatial import shift_array_integer
from synthworkshop.scenes import RenderedScene


def shift_skeleton_masks(
    scene: RenderedScene,
    *,
    shift_voxels: Sequence[int],
    mask_names: Sequence[str] | None = None,
) -> RenderedScene:
    """Shift skeleton masks by integer voxels."""

    shift = validate_shift(shift_voxels, ndim=scene.grid.ndim)
    selected = selected_names(scene.skeleton_masks, mask_names, label="skeleton mask")

    skeleton_masks = dict(scene.skeleton_masks)
    for name in selected:
        skeleton_masks[name] = shift_array_integer(
            scene.skeleton_masks[name],
            shift,
            fill_value=False,
        ).astype(bool)

    record = PerturbationRecord(
        name="integer_skeleton_shift",
        target="skeleton_masks",
        parameters={
            "shift_voxels": shift,
            "mask_names": selected,
            "centrelines_updated": False,
            "frames_updated": False,
        },
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(f"skeleton_masks:{name}" for name in selected),
        note=(
            "This perturbation shifts skeleton masks only. Centreline and frame "
            "tables are left unchanged."
        ),
    )

    return attach_perturbation_record(
        replace(scene, skeleton_masks=skeleton_masks),
        record,
    )


def break_skeleton_masks(
    scene: RenderedScene,
    *,
    mask_names: Sequence[str] | None = None,
    n_breaks: int = 1,
    radius_voxels: int = 1,
    seed: int | None = None,
    centres_voxels: Sequence[Sequence[int]] | None = None,
) -> RenderedScene:
    """Remove local blocks from skeleton masks."""

    n_blocks = validate_positive_int(n_breaks, name="n_breaks")
    radius = _validate_non_negative_int(radius_voxels, name="radius_voxels")
    selected = selected_names(scene.skeleton_masks, mask_names, label="skeleton mask")
    rng = np.random.default_rng(seed)

    skeleton_masks = dict(scene.skeleton_masks)
    centres_by_mask: dict[str, list[list[int]]] = {}

    for name in selected:
        mask = np.asarray(skeleton_masks[name], dtype=bool).copy()
        centres = _choose_centres(
            mask,
            n_centres=n_blocks,
            rng=rng,
            centres_voxels=centres_voxels,
        )
        for centre in centres:
            _remove_local_block(mask, centre=centre, radius=radius)
        skeleton_masks[name] = mask
        centres_by_mask[name] = [list(centre) for centre in centres]

    record = PerturbationRecord(
        name="broken_skeleton",
        target="skeleton_masks",
        parameters={
            "mask_names": selected,
            "n_breaks": n_blocks,
            "radius_voxels": radius,
            "centres_voxels": centres_by_mask,
            "centrelines_updated": False,
            "frames_updated": False,
        },
        seed=seed,
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(f"skeleton_masks:{name}" for name in selected),
        note=(
            "This perturbation removes voxels from skeleton masks only. "
            "Centreline and frame tables are left unchanged."
        ),
    )

    return attach_perturbation_record(
        replace(scene, skeleton_masks=skeleton_masks),
        record,
    )


def _choose_centres(
    mask: np.ndarray,
    *,
    n_centres: int,
    rng: np.random.Generator,
    centres_voxels: Sequence[Sequence[int]] | None,
) -> list[tuple[int, ...]]:
    if centres_voxels is not None:
        centres = [tuple(int(value) for value in centre) for centre in centres_voxels]
        for centre in centres:
            if len(centre) != mask.ndim:
                raise ValueError(
                    f"centres_voxels entries must have {mask.ndim} values."
                )
            if any(
                index < 0 or index >= size
                for index, size in zip(centre, mask.shape, strict=True)
            ):
                raise ValueError(
                    "centres_voxels entries must lie within the mask shape."
                )
        return centres

    coords = np.argwhere(mask)
    if coords.size == 0:
        return []

    n_choose = min(n_centres, coords.shape[0])
    chosen = rng.choice(coords.shape[0], size=n_choose, replace=False)
    return [tuple(int(value) for value in coords[index]) for index in chosen]


def _remove_local_block(
    mask: np.ndarray,
    *,
    centre: tuple[int, ...],
    radius: int,
) -> None:
    slices = tuple(
        slice(max(0, index - radius), min(size, index + radius + 1))
        for index, size in zip(centre, mask.shape, strict=True)
    )
    mask[slices] = False


def _validate_non_negative_int(value: int, *, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative.")
    return out
