"""Orthogonal slice plots for rendered scenes."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from synthworkshop.scenes import RenderedScene


def default_slice_indices(shape: Sequence[int]) -> tuple[int, int, int]:
    """Return central slice indices for a 3D shape."""

    if len(shape) != 3:
        raise ValueError("Orthogonal slices currently require a 3D shape.")
    return tuple(int(size // 2) for size in shape)


def slice_2d(image: np.ndarray, *, axis: int, index: int) -> np.ndarray:
    """Extract a 2D slice from a 3D image."""

    if image.ndim != 3:
        raise ValueError("slice_2d requires a 3D image.")
    if axis == 0:
        return image[index, :, :]
    if axis == 1:
        return image[:, index, :]
    if axis == 2:
        return image[:, :, index]
    raise ValueError("axis must be 0, 1, or 2.")


def plot_orthogonal_slices(
    scene: RenderedScene,
    *,
    map_name: str,
    indices: Sequence[int] | None = None,
    with_colorbar: bool = False,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot three orthogonal scalar-map slices."""

    if map_name not in scene.scalar_maps:
        raise KeyError(f"Unknown scalar map: {map_name!r}.")
    image = scene.scalar_maps[map_name]
    if image.ndim != 3:
        raise ValueError("Orthogonal slice plotting currently requires a 3D image.")

    slice_indices = (
        default_slice_indices(image.shape)
        if indices is None
        else tuple(int(value) for value in indices)
    )
    if len(slice_indices) != 3:
        raise ValueError("indices must contain three values.")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for axis, ax in enumerate(axes):
        panel = slice_2d(image, axis=axis, index=slice_indices[axis])
        artist = ax.imshow(panel.T, origin="lower", interpolation="nearest")
        ax.set_title(f"axis {axis} index {slice_indices[axis]}")
        ax.set_xticks([])
        ax.set_yticks([])
        if with_colorbar:
            fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontweight="bold")

    return fig, axes
