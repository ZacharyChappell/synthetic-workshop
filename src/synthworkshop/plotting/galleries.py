"""Small gallery writer for rendered-scene inspection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from synthworkshop.plotting.legends import write_scene_legend
from synthworkshop.plotting.projections import plot_projection
from synthworkshop.plotting.skeletons import plot_3d_skeletons
from synthworkshop.plotting.slices import plot_orthogonal_slices
from synthworkshop.plotting.style import close_figure, save_figure
from synthworkshop.scenes import RenderedScene


def write_scene_gallery(
    scene: RenderedScene,
    output_dir: str | Path,
    *,
    map_name: str,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    overwrite: bool = False,
    with_colorbar: bool = False,
) -> dict[str, list[Path] | Path]:
    """Write a compact inspection gallery for one rendered scene."""

    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path] | Path] = {}

    fig, _ = plot_projection(
        scene,
        map_name=map_name,
        with_colorbar=with_colorbar,
    )
    written["best_projection"] = save_figure(
        fig,
        fig_dir,
        stem="best_projection",
        formats=formats,
        dpi=dpi,
        overwrite=overwrite,
    )
    close_figure(fig)

    fig, _ = plot_orthogonal_slices(
        scene,
        map_name=map_name,
        with_colorbar=with_colorbar,
        title=f"{map_name}: orthogonal slices",
    )
    written["orthogonal_slices"] = save_figure(
        fig,
        fig_dir,
        stem="orthogonal_slices",
        formats=formats,
        dpi=dpi,
        overwrite=overwrite,
    )
    close_figure(fig)

    fig, _ = plot_3d_skeletons(scene)
    written["skeletons_3d"] = save_figure(
        fig,
        fig_dir,
        stem="skeletons_3d",
        formats=formats,
        dpi=dpi,
        overwrite=overwrite,
    )
    close_figure(fig)

    written["legend"] = write_scene_legend(
        scene,
        table_dir / "scene_legend.tsv",
        overwrite=overwrite,
    )

    return written
