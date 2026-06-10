"""3D centreline and skeleton visualisation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from synthworkshop.scenes import RenderedScene


def _plot_centreline_table(ax: Axes, table, *, label: str) -> None:
    """Plot one centreline table on a 3D axis."""

    required = {"i_mm", "j_mm", "k_mm"}
    if not required.issubset(table.columns):
        return
    ax.plot(
        table["i_mm"].to_numpy(),
        table["j_mm"].to_numpy(),
        table["k_mm"].to_numpy(),
        marker=".",
        linewidth=1.2,
        markersize=3.0,
        label=label,
    )


def _plot_skeleton_mask(ax: Axes, mask: np.ndarray, *, label: str) -> None:
    """Plot a sparse skeleton mask on a 3D axis."""

    coords = np.argwhere(mask)
    if coords.size == 0:
        return
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        marker=".",
        s=8,
        label=label,
    )


def plot_3d_skeletons(
    scene: RenderedScene,
    *,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot available centreline tables or skeleton masks in 3D."""

    fig = plt.figure(figsize=(6.0, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    plotted = False
    for name, table in scene.centrelines.items():
        _plot_centreline_table(ax, table, label=name)
        plotted = True

    if not plotted:
        for name, mask in scene.skeleton_masks.items():
            _plot_skeleton_mask(ax, mask, label=name)
            plotted = True

    ax.set_title(title or "3D centrelines / skeletons")
    ax.set_xlabel("i / x")
    ax.set_ylabel("j / y")
    ax.set_zlabel("k / z")

    if plotted:
        ax.legend(loc="best")

    return fig, ax
