"""Apply configured perturbations to rendered scenes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from synthworkshop.perturbations.intensity import (
    add_linear_bias_field,
    scale_scalar_maps,
)
from synthworkshop.perturbations.masks import (
    add_mask_contamination,
    add_mask_holes,
    dilate_masks,
    erode_masks,
)
from synthworkshop.perturbations.noise import add_gaussian_noise
from synthworkshop.perturbations.skeletons import (
    break_skeleton_masks,
    shift_skeleton_masks,
)
from synthworkshop.perturbations.spatial import (
    mean_blur_scalar_maps,
    shift_scalar_maps,
)
from synthworkshop.scenes import RenderedScene

PerturbationFunction = Callable[..., RenderedScene]


PERTURBATION_REGISTRY: dict[str, PerturbationFunction] = {
    "gaussian_noise": add_gaussian_noise,
    "add_gaussian_noise": add_gaussian_noise,
    "mean_blur": mean_blur_scalar_maps,
    "mean_blur_scalar_maps": mean_blur_scalar_maps,
    "blur": mean_blur_scalar_maps,
    "integer_scalar_shift": shift_scalar_maps,
    "shift_scalar_maps": shift_scalar_maps,
    "scalar_shift": shift_scalar_maps,
    "mask_erosion": erode_masks,
    "erode_masks": erode_masks,
    "mask_dilation": dilate_masks,
    "dilate_masks": dilate_masks,
    "mask_holes": add_mask_holes,
    "add_mask_holes": add_mask_holes,
    "mask_contamination": add_mask_contamination,
    "add_mask_contamination": add_mask_contamination,
    "integer_skeleton_shift": shift_skeleton_masks,
    "shift_skeleton_masks": shift_skeleton_masks,
    "skeleton_shift": shift_skeleton_masks,
    "broken_skeleton": break_skeleton_masks,
    "break_skeleton_masks": break_skeleton_masks,
    "intensity_scaling": scale_scalar_maps,
    "scale_scalar_maps": scale_scalar_maps,
    "linear_bias_field": add_linear_bias_field,
    "add_linear_bias_field": add_linear_bias_field,
}


def available_perturbations() -> tuple[str, ...]:
    """Return known perturbation kinds."""

    return tuple(sorted(PERTURBATION_REGISTRY))


def apply_perturbation(
    scene: RenderedScene,
    perturbation: Mapping[str, Any],
) -> RenderedScene:
    """Apply one perturbation specification to a scene."""

    if not isinstance(perturbation, Mapping):
        raise TypeError("perturbation must be a mapping.")

    spec = dict(perturbation)
    kind = spec.pop("kind", None)
    if kind is None:
        raise ValueError("perturbation specification must include a 'kind' field.")
    if not isinstance(kind, str) or not kind:
        raise ValueError("perturbation 'kind' must be a non-empty string.")

    try:
        perturbation_function = PERTURBATION_REGISTRY[kind]
    except KeyError as error:
        known = ", ".join(available_perturbations())
        raise ValueError(
            f"Unknown perturbation kind {kind!r}. Known kinds: {known}."
        ) from error

    return perturbation_function(scene, **spec)


def apply_perturbations(
    scene: RenderedScene,
    perturbations: Iterable[Mapping[str, Any]] | None,
) -> RenderedScene:
    """Apply perturbation specifications in order."""

    if perturbations is None:
        return scene

    current = scene
    for perturbation in perturbations:
        current = apply_perturbation(current, perturbation)
    return current
