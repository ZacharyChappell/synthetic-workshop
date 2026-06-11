"""Binary-mask perturbations for rendered scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product

import numpy as np

from synthworkshop.perturbations.base import (
    PerturbationRecord,
    attach_perturbation_record,
    selected_names,
    validate_positive_int,
)
from synthworkshop.scenes import RenderedScene

MASK_GROUPS = {
    "object_masks",
    "target_masks",
    "analysis_masks",
    "skeleton_masks",
}


def erode_masks(
    scene: RenderedScene,
    *,
    mask_group: str = "object_masks",
    mask_names: Sequence[str] | None = None,
    iterations: int = 1,
) -> RenderedScene:
    """Erode selected binary masks."""

    return _perturb_masks(
        scene,
        operation="erosion",
        mask_group=mask_group,
        mask_names=mask_names,
        iterations=iterations,
    )


def dilate_masks(
    scene: RenderedScene,
    *,
    mask_group: str = "object_masks",
    mask_names: Sequence[str] | None = None,
    iterations: int = 1,
) -> RenderedScene:
    """Dilate selected binary masks."""

    return _perturb_masks(
        scene,
        operation="dilation",
        mask_group=mask_group,
        mask_names=mask_names,
        iterations=iterations,
    )


def _perturb_masks(
    scene: RenderedScene,
    *,
    operation: str,
    mask_group: str,
    mask_names: Sequence[str] | None,
    iterations: int,
) -> RenderedScene:
    n_iter = validate_positive_int(iterations, name="iterations")
    masks = _mask_mapping(scene, mask_group)
    selected = selected_names(masks, mask_names, label=mask_group)

    updated = dict(masks)
    for name in selected:
        mask = np.asarray(updated[name], dtype=bool)
        for _ in range(n_iter):
            if operation == "erosion":
                mask = _erode_once(mask)
            elif operation == "dilation":
                mask = _dilate_once(mask)
            else:  # pragma: no cover
                raise ValueError(f"Unknown mask operation: {operation}")
        updated[name] = mask

    scene_with_masks = _replace_mask_group(scene, mask_group, updated)

    record = PerturbationRecord(
        name=f"mask_{operation}",
        target=mask_group,
        parameters={
            "mask_group": mask_group,
            "mask_names": selected,
            "iterations": n_iter,
        },
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(f"{mask_group}:{name}" for name in selected),
        note=(
            "Object-mask perturbations rederive target and analysis masks "
            "from the perturbed object masks."
            if mask_group == "object_masks"
            else None
        ),
    )
    return attach_perturbation_record(scene_with_masks, record)


def _mask_mapping(scene: RenderedScene, mask_group: str) -> Mapping[str, np.ndarray]:
    if mask_group not in MASK_GROUPS:
        allowed = ", ".join(sorted(MASK_GROUPS))
        raise ValueError(f"mask_group must be one of: {allowed}.")
    return getattr(scene, mask_group)


def _replace_mask_group(
    scene: RenderedScene,
    mask_group: str,
    masks: Mapping[str, np.ndarray],
) -> RenderedScene:
    if mask_group == "object_masks":
        return replace(
            scene,
            object_masks=dict(masks),
            target_masks=None,
            analysis_masks=None,
        )
    return replace(scene, **{mask_group: dict(masks)})


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
