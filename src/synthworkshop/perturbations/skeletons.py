"""Skeleton-mask perturbations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from synthworkshop.perturbations.base import (
    PerturbationRecord,
    attach_perturbation_record,
    selected_names,
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
