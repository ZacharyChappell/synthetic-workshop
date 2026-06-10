from __future__ import annotations

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.placement import (
    geometry_controls_for_object,
    grid_extent_mm,
    update_object_geometry,
)
from synthworkshop.gui.state import read_scene_text
from synthworkshop.gui.yaml_editor import (
    get_object,
    parse_scene_text,
)
from synthworkshop.scenes.config import render_scene_from_dict


def _basic_tube_text() -> str:
    return read_scene_text(get_catalogue_entry("basic_tube"))


def test_grid_extent_mm_for_basic_tube() -> None:
    assert grid_extent_mm(_basic_tube_text()) == [32.0, 32.0, 32.0]


def test_geometry_controls_for_tube_include_start_end_and_radius() -> None:
    obj = get_object(_basic_tube_text(), "target")
    controls = geometry_controls_for_object(
        obj,
        extent_mm=[32.0, 32.0, 32.0],
    )
    paths = {control.path for control in controls}

    assert "curve.start_mm" in paths
    assert "curve.end_mm" in paths
    assert "cross_section.radius_mm" in paths


def test_update_object_geometry_updates_tube_radius_and_renders() -> None:
    text = update_object_geometry(
        _basic_tube_text(),
        "target",
        {
            "cross_section.radius_mm": 4.5,
            "curve.start_mm": [6.0, 16.0, 16.0],
            "curve.end_mm": [26.0, 16.0, 16.0],
        },
    )

    obj = get_object(text, "target")
    assert obj["cross_section"]["radius_mm"] == 4.5
    assert obj["curve"]["start_mm"] == [6.0, 16.0, 16.0]

    scene = render_scene_from_dict(parse_scene_text(text))
    assert "target" in scene.object_masks
