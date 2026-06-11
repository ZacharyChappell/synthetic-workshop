"""Controlled perturbations for rendered synthetic scenes."""

from synthworkshop.perturbations.base import PerturbationRecord
from synthworkshop.perturbations.masks import dilate_masks, erode_masks
from synthworkshop.perturbations.noise import add_gaussian_noise
from synthworkshop.perturbations.skeletons import shift_skeleton_masks
from synthworkshop.perturbations.spatial import (
    mean_blur_scalar_maps,
    shift_array_integer,
    shift_scalar_maps,
)

__all__ = [
    "PerturbationRecord",
    "add_gaussian_noise",
    "dilate_masks",
    "erode_masks",
    "mean_blur_scalar_maps",
    "shift_array_integer",
    "shift_scalar_maps",
    "shift_skeleton_masks",
]
