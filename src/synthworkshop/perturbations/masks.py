"""Binary-mask perturbations for rendered scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product

import numpy as np
from numpy.typing import ArrayLike

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


def add_mask_holes(
    scene: RenderedScene,
    *,
    mask_group: str = "object_masks",
    mask_names: Sequence[str] | None = None,
    n_holes: int = 1,
    radius_voxels: int = 1,
    seed: int | None = None,
    centres_voxels: Sequence[Sequence[int]] | None = None,
) -> RenderedScene:
    """Remove local cuboid holes from selected binary masks."""

    n_blocks = validate_positive_int(n_holes, name="n_holes")
    radius = _validate_non_negative_int(radius_voxels, name="radius_voxels")
    masks = _mask_mapping(scene, mask_group)
    selected = selected_names(masks, mask_names, label=mask_group)
    rng = np.random.default_rng(seed)

    updated = dict(masks)
    centres_by_mask: dict[str, list[list[int]]] = {}

    for name in selected:
        mask = np.asarray(updated[name], dtype=bool).copy()
        centres = _choose_centres(
            mask,
            n_centres=n_blocks,
            rng=rng,
            centres_voxels=centres_voxels,
        )
        for centre in centres:
            _set_local_block(mask, centre=centre, radius=radius, value=False)
        updated[name] = mask
        centres_by_mask[name] = [list(centre) for centre in centres]

    scene_with_masks = _replace_mask_group(scene, mask_group, updated)

    record = PerturbationRecord(
        name="mask_holes",
        target=mask_group,
        parameters={
            "mask_group": mask_group,
            "mask_names": selected,
            "n_holes": n_blocks,
            "radius_voxels": radius,
            "centres_voxels": centres_by_mask,
        },
        seed=seed,
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(f"{mask_group}:{name}" for name in selected),
    )

    return attach_perturbation_record(scene_with_masks, record)


def add_mask_contamination(
    scene: RenderedScene,
    *,
    mask_group: str = "object_masks",
    mask_names: Sequence[str] | None = None,
    contamination_mask: ArrayLike | None = None,
    source_mask_group: str | None = None,
    source_mask_name: str | None = None,
    fraction: float = 1.0,
    seed: int | None = None,
) -> RenderedScene:
    """Add contaminating voxels to selected binary masks."""

    fraction_value = float(fraction)
    if not np.isfinite(fraction_value) or not 0.0 <= fraction_value <= 1.0:
        raise ValueError("fraction must be finite and between 0 and 1.")

    masks = _mask_mapping(scene, mask_group)
    selected = selected_names(masks, mask_names, label=mask_group)
    contamination = _resolve_contamination_mask(
        scene,
        contamination_mask=contamination_mask,
        source_mask_group=source_mask_group,
        source_mask_name=source_mask_name,
    )
    contamination = _sample_mask(contamination, fraction=fraction_value, seed=seed)

    updated = dict(masks)
    for name in selected:
        updated[name] = np.asarray(updated[name], dtype=bool) | contamination

    scene_with_masks = _replace_mask_group(scene, mask_group, updated)

    record = PerturbationRecord(
        name="mask_contamination",
        target=mask_group,
        parameters={
            "mask_group": mask_group,
            "mask_names": selected,
            "source_mask_group": source_mask_group,
            "source_mask_name": source_mask_name,
            "fraction": fraction_value,
            "n_contamination_voxels": int(contamination.sum()),
        },
        seed=seed,
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(f"{mask_group}:{name}" for name in selected),
    )

    return attach_perturbation_record(scene_with_masks, record)


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


def _resolve_contamination_mask(
    scene: RenderedScene,
    *,
    contamination_mask: ArrayLike | None,
    source_mask_group: str | None,
    source_mask_name: str | None,
) -> np.ndarray:
    if contamination_mask is not None:
        mask = np.asarray(contamination_mask, dtype=bool)
        if mask.shape != scene.grid.shape:
            raise ValueError("contamination_mask must match the scene grid shape.")
        return mask

    if source_mask_group is None or source_mask_name is None:
        raise ValueError(
            "Provide either contamination_mask or both source_mask_group "
            "and source_mask_name."
        )

    source_masks = _mask_mapping(scene, source_mask_group)
    if source_mask_name not in source_masks:
        raise ValueError(
            f"Unknown source mask {source_mask_name!r} in {source_mask_group}."
        )

    return np.asarray(source_masks[source_mask_name], dtype=bool)


def _sample_mask(mask: np.ndarray, *, fraction: float, seed: int | None) -> np.ndarray:
    if fraction == 1.0:
        return mask

    coords = np.argwhere(mask)
    out = np.zeros_like(mask, dtype=bool)
    if coords.size == 0 or fraction == 0.0:
        return out

    rng = np.random.default_rng(seed)
    n_keep = int(np.ceil(coords.shape[0] * fraction))
    chosen = rng.choice(coords.shape[0], size=n_keep, replace=False)
    out[tuple(coords[chosen].T)] = True
    return out


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


def _set_local_block(
    mask: np.ndarray,
    *,
    centre: tuple[int, ...],
    radius: int,
    value: bool,
) -> None:
    slices = tuple(
        slice(max(0, index - radius), min(size, index + radius + 1))
        for index, size in zip(centre, mask.shape, strict=True)
    )
    mask[slices] = value


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


def _validate_non_negative_int(value: int, *, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative.")
    return out
