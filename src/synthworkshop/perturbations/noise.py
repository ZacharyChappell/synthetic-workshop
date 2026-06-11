"""Noise perturbations for rendered scenes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike

from synthworkshop.perturbations.base import (
    PerturbationRecord,
    attach_perturbation_record,
    selected_names,
    validate_non_negative_finite,
)
from synthworkshop.scenes import RenderedScene


def add_gaussian_noise(
    scene: RenderedScene,
    *,
    sigma: float,
    seed: int | None = None,
    map_names: Sequence[str] | None = None,
    mask: ArrayLike | None = None,
) -> RenderedScene:
    """Add reproducible Gaussian noise to selected scalar maps."""

    sigma_value = validate_non_negative_finite(sigma, name="sigma")
    selected = selected_names(scene.scalar_maps, map_names, label="scalar map")

    mask_array = None
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != scene.grid.shape:
            raise ValueError("mask must match the scene grid shape.")

    rng = np.random.default_rng(seed)
    scalar_maps = dict(scene.scalar_maps)
    affected: list[str] = []

    for name in selected:
        array = np.asarray(scene.scalar_maps[name], dtype=float)
        noise = rng.normal(0.0, sigma_value, size=array.shape)
        if mask_array is not None:
            noise = np.where(mask_array, noise, 0.0)
        scalar_maps[name] = array + noise
        affected.append(f"scalar_maps:{name}")

    record = PerturbationRecord(
        name="gaussian_noise",
        target="scalar_maps",
        parameters={
            "sigma": sigma_value,
            "map_names": selected,
            "masked": mask_array is not None,
        },
        seed=seed,
        truth_changed=False,
        observed_changed=True,
        affected_arrays=tuple(affected),
    )

    return attach_perturbation_record(
        replace(scene, scalar_maps=scalar_maps),
        record,
    )
