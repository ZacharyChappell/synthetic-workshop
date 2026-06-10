from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from synthworkshop import GridSpec
from synthworkshop.cross_sections import CircularCrossSection
from synthworkshop.plotting import (
    choose_projection_axis,
    plot_3d_skeletons,
    plot_orthogonal_slices,
    plot_projection,
    project_image,
    scene_legend_table,
    write_scene_gallery,
)
from synthworkshop.primitives import LineCurve, TubeObject
from synthworkshop.profiles import LinearRadialProfile


def _small_scene():
    grid = GridSpec(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 8.0, 8.0),
        end_mm=(12.0, 8.0, 8.0),
    ).sample(step_mm=1.0, object_id="target")
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(centre_value=1.0, edge_value=0.2),
        map_name="fa_like",
        role="target",
        label=1,
        priority=10,
    )
    return tube.render(grid)


def test_project_image_max_mean_sum() -> None:
    image = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)

    assert project_image(image, axis=2, mode="max").shape == (2, 3)
    assert np.allclose(project_image(image, axis=2, mode="sum"), image.sum(axis=2))
    assert np.allclose(project_image(image, axis=1, mode="mean"), image.mean(axis=1))


def test_project_image_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        project_image(np.zeros((2, 2, 2)), mode="median")


def test_choose_projection_axis_returns_valid_axis() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:7, 4, 4] = True

    axis = choose_projection_axis(mask)

    assert axis in {0, 1, 2}


def test_plot_projection_returns_figure_and_axis() -> None:
    scene = _small_scene()
    fig, ax = plot_projection(scene, map_name="fa_like")

    assert fig is ax.figure
    assert ax.get_title()
    assert len(ax.images) == 1


def test_plot_projection_rejects_unknown_map() -> None:
    scene = _small_scene()

    with pytest.raises(KeyError, match="Unknown scalar map"):
        plot_projection(scene, map_name="missing")


def test_plot_orthogonal_slices_returns_three_axes() -> None:
    scene = _small_scene()
    fig, axes = plot_orthogonal_slices(scene, map_name="fa_like")

    assert fig is axes[0].figure
    assert len(axes) == 3
    assert all(len(ax.images) == 1 for ax in axes)


def test_plot_3d_skeletons_returns_3d_axis() -> None:
    scene = _small_scene()
    fig, ax = plot_3d_skeletons(scene)

    assert fig is ax.figure
    assert hasattr(ax, "get_zlim")


def test_scene_legend_table_contains_object_metadata() -> None:
    scene = _small_scene()
    legend = scene_legend_table(scene)

    assert legend.shape[0] == 1
    assert legend.loc[0, "object_id"] == "target"
    assert legend.loc[0, "role"] == "target"
    assert legend.loc[0, "label"] == 1


def test_write_scene_gallery_writes_expected_files(tmp_path: Path) -> None:
    scene = _small_scene()
    written = write_scene_gallery(
        scene,
        tmp_path / "gallery",
        map_name="fa_like",
        formats=("png", "pdf"),
        dpi=100,
    )

    assert "best_projection" in written
    assert "orthogonal_slices" in written
    assert "skeletons_3d" in written
    assert "legend" in written

    for key in ("best_projection", "orthogonal_slices", "skeletons_3d"):
        paths = written[key]
        assert isinstance(paths, list)
        assert len(paths) == 2
        assert all(path.exists() for path in paths)

    legend_path = written["legend"]
    assert isinstance(legend_path, Path)
    assert legend_path.exists()


def test_write_scene_gallery_respects_overwrite_guard(tmp_path: Path) -> None:
    scene = _small_scene()
    write_scene_gallery(scene, tmp_path / "gallery", map_name="fa_like")

    with pytest.raises(FileExistsError):
        write_scene_gallery(scene, tmp_path / "gallery", map_name="fa_like")


def test_top_level_exports_plotting_helpers() -> None:
    import synthworkshop

    assert synthworkshop.plot_projection is plot_projection
    assert synthworkshop.write_scene_gallery is write_scene_gallery
