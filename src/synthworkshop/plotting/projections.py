"""Projection-based scene plots."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from synthworkshop.scenes import RenderedScene


def project_image(
    image: ArrayLike,
    *,
    axis: int = 2,
    mode: str = "max",
) -> np.ndarray:
    """Project an image along one axis."""

    arr = np.asarray(image, dtype=float)
    if arr.ndim != 3:
        raise ValueError("Projection plotting currently requires a 3D image.")
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("axis is out of bounds for image.")

    if mode == "max":
        return np.nanmax(arr, axis=axis)
    if mode == "mean":
        return np.nanmean(arr, axis=axis)
    if mode == "sum":
        return np.nansum(arr, axis=axis)

    raise ValueError("mode must be one of: max, mean, sum.")


def choose_projection_axis(mask_or_image: ArrayLike) -> int:
    """Choose a projection axis that gives a broad non-empty footprint."""

    arr = np.asarray(mask_or_image)
    if arr.ndim != 3:
        raise ValueError("choose_projection_axis currently requires a 3D array.")

    work = np.asarray(arr != 0, dtype=bool)
    if not np.any(work):
        return 2

    scores = []
    for axis in range(3):
        projection = np.max(work, axis=axis)
        coords = np.argwhere(projection)
        if coords.size == 0:
            scores.append(-1)
            continue
        extent = coords.max(axis=0) - coords.min(axis=0) + 1
        scores.append(int(np.prod(extent)))
    return int(np.argmax(scores))


def _default_overlay_masks(scene: RenderedScene) -> Mapping[str, np.ndarray]:
    """Return default masks to overlay on projection plots."""

    masks: dict[str, np.ndarray] = {}
    for name, mask in scene.analysis_masks.items():
        masks[f"analysis:{name}"] = mask
    for name, mask in scene.target_masks.items():
        masks[f"target:{name}"] = mask
    return masks


def plot_projection(
    scene: RenderedScene,
    *,
    map_name: str,
    drop_axis: int | None = None,
    projection_mode: str = "max",
    overlay_masks: bool = True,
    overlay_skeletons: bool = True,
    with_colorbar: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a projected scalar map with optional masks and skeleton overlays."""

    if map_name not in scene.scalar_maps:
        raise KeyError(f"Unknown scalar map: {map_name!r}.")

    image = scene.scalar_maps[map_name]
    if drop_axis is None:
        if scene.target_masks:
            first_mask = next(iter(scene.target_masks.values()))
            drop_axis = choose_projection_axis(first_mask)
        else:
            drop_axis = choose_projection_axis(image)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
    else:
        fig = ax.figure

    projected = project_image(image, axis=drop_axis, mode=projection_mode)
    artist = ax.imshow(projected.T, origin="lower", interpolation="nearest")

    if overlay_masks:
        for _, mask in _default_overlay_masks(scene).items():
            projected_mask = np.max(mask.astype(float), axis=drop_axis)
            if np.any(projected_mask):
                ax.contour(
                    projected_mask.T,
                    levels=[0.5],
                    linewidths=0.8,
                    alpha=0.75,
                )

    if overlay_skeletons:
        for _, skeleton in scene.skeleton_masks.items():
            projected_skeleton = np.max(skeleton.astype(float), axis=drop_axis)
            coords = np.argwhere(projected_skeleton > 0)
            if coords.size:
                ax.scatter(coords[:, 0], coords[:, 1], s=8, marker=".")

    ax.set_title(title or f"{map_name}: best projection")
    ax.set_xlabel("projected axis 0")
    ax.set_ylabel("projected axis 1")
    ax.set_xticks([])
    ax.set_yticks([])

    if with_colorbar:
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax
