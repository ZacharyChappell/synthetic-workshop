"""Controlled perturbations for rendered synthetic scenes."""

from synthworkshop.perturbations.base import PerturbationRecord
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
    shift_array_integer,
    shift_scalar_maps,
)

__all__ = [
    "PerturbationRecord",
    "add_gaussian_noise",
    "add_linear_bias_field",
    "add_mask_contamination",
    "add_mask_holes",
    "break_skeleton_masks",
    "dilate_masks",
    "erode_masks",
    "mean_blur_scalar_maps",
    "scale_scalar_maps",
    "shift_array_integer",
    "shift_scalar_maps",
    "shift_skeleton_masks",
]
