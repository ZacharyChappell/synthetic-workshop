from __future__ import annotations

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.profiles import (
    PROFILE_KINDS,
    apply_profile_updates,
    profile_controls_for_profile,
    profile_for_object,
    profile_template,
    replace_object_profile,
    update_object_profile,
)
from synthworkshop.gui.state import read_scene_text
from synthworkshop.gui.yaml_editor import get_object, parse_scene_text
from synthworkshop.scenes.config import render_scene_from_dict


def _basic_tube_text() -> str:
    return read_scene_text(get_catalogue_entry("basic_tube"))


def test_profile_kinds_include_core_profiles() -> None:
    assert "constant" in PROFILE_KINDS
    assert "linear_radial" in PROFILE_KINDS
    assert "gaussian_radial" in PROFILE_KINDS


def test_profile_for_object_returns_profile_mapping() -> None:
    obj = get_object(_basic_tube_text(), "target")
    profile = profile_for_object(obj)

    assert "kind" in profile


def test_profile_template_preserves_compatible_values() -> None:
    template = profile_template(
        "linear_radial",
        existing={
            "kind": "linear_radial",
            "centre_value": 2.0,
            "edge_value": 0.1,
        },
    )

    assert template["centre_value"] == 2.0
    assert template["edge_value"] == 0.1


def test_profile_controls_for_linear_radial_profile() -> None:
    profile = profile_template("linear_radial")
    controls = profile_controls_for_profile(profile)
    keys = {control.key for control in controls}

    assert "centre_value" in keys
    assert "edge_value" in keys
    assert "background_value" in keys


def test_apply_profile_updates_changes_values() -> None:
    profile = profile_template("linear_radial")
    updated = apply_profile_updates(
        profile,
        {
            "centre_value": 1.5,
            "edge_value": 0.4,
        },
    )

    assert updated["centre_value"] == 1.5
    assert updated["edge_value"] == 0.4


def test_replace_object_profile_updates_yaml_and_renders() -> None:
    text = replace_object_profile(
        _basic_tube_text(),
        "target",
        profile_template(
            "linear_radial",
            existing={
                "centre_value": 1.5,
                "edge_value": 0.4,
                "background_value": 0.0,
            },
        ),
    )

    obj = get_object(text, "target")
    assert obj["profile"]["centre_value"] == 1.5

    scene = render_scene_from_dict(parse_scene_text(text))
    assert "target" in scene.object_masks


def test_update_object_profile_can_switch_to_constant_and_render() -> None:
    text = update_object_profile(
        _basic_tube_text(),
        "target",
        kind="constant",
        updates={
            "value": 0.75,
            "background_value": 0.0,
        },
    )

    obj = get_object(text, "target")
    assert obj["profile"]["kind"] == "constant"
    assert obj["profile"]["value"] == 0.75

    scene = render_scene_from_dict(parse_scene_text(text))
    assert "target" in scene.object_masks
