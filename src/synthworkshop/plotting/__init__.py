"""Plotting utilities for rendered synthetic scenes."""

from synthworkshop.plotting.galleries import write_scene_gallery
from synthworkshop.plotting.legends import scene_legend_table, write_scene_legend
from synthworkshop.plotting.projections import (
    choose_projection_axis,
    plot_projection,
    project_image,
)
from synthworkshop.plotting.skeletons import plot_3d_skeletons
from synthworkshop.plotting.slices import (
    default_slice_indices,
    plot_orthogonal_slices,
    slice_2d,
)
from synthworkshop.plotting.style import clean_title, close_figure, save_figure

__all__ = [
    "choose_projection_axis",
    "clean_title",
    "close_figure",
    "default_slice_indices",
    "plot_3d_skeletons",
    "plot_orthogonal_slices",
    "plot_projection",
    "project_image",
    "save_figure",
    "scene_legend_table",
    "slice_2d",
    "write_scene_gallery",
    "write_scene_legend",
]
