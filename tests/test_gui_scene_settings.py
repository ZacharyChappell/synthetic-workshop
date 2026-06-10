from __future__ import annotations

import pytest

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.scene_settings import (
    LABEL_MODES,
    OVERLAP_POLICIES,
    SCALAR_BLEND_MODES,
    format_numeric_list,
    format_string_list,
    scene_settings_from_text,
    update_scene_settings,
)
from synthworkshop.gui.state import read_scene_text
from synthworkshop.gui.yaml_editor import parse_scene_text
from synthworkshop.scenes.config import render_scene_from_dict


def _basic_tube_text() -> str:
    return read_scene_text(get_catalogue_entry("basic_tube"))


def test_scene_settings_constants_include_expected_values() -> None:
    assert "priority" in LABEL_MODES
    assert "overwrite" in SCALAR_BLEND_MODES
    assert "warn" in OVERLAP_POLICIES


def test_scene_settings_from_text_reads_basic_tube() -> None:
    settings = scene_settings_from_text(_basic_tube_text())

    assert settings["scene_id"] == "basic_tube"
    assert settings["shape"] == [32, 32, 32]
    assert settings["spacing"] == [1.0, 1.0, 1.0]
    assert settings["label_mode"] == "priority"


def test_update_scene_settings_changes_metadata_grid_and_composition() -> None:
    text = update_scene_settings(
        _basic_tube_text(),
        scene_id="updated_scene",
        description="Updated through tests.",
        shape="[40, 32, 32]",
        spacing="[1.0, 1.5, 1.0]",
        label_mode="priority",
        scalar_blend="max",
        overlap_policy="error",
        target_roles="[target]",
        analysis_roles="[target, analysis_support]",
    )

    payload = parse_scene_text(text)

    assert payload["scene"]["id"] == "updated_scene"
    assert payload["grid"]["shape"] == [40, 32, 32]
    assert payload["grid"]["spacing"] == [1.0, 1.5, 1.0]
    assert payload["composition"]["scalar_blend"] == "max"
    assert payload["composition"]["overlap_policy"] == "error"

    scene = render_scene_from_dict(payload)
    assert scene.metadata["scene_id"] == "updated_scene"


def test_update_scene_settings_accepts_comma_separated_roles() -> None:
    text = update_scene_settings(
        _basic_tube_text(),
        scene_id="roles_scene",
        description="Roles scene.",
        shape="[32, 32, 32]",
        spacing="[1.0, 1.0, 1.0]",
        label_mode="priority",
        scalar_blend="overwrite",
        overlap_policy="warn",
        target_roles="target",
        analysis_roles="target, analysis_support",
    )

    payload = parse_scene_text(text)

    assert payload["mask_rules"]["target_roles"] == ["target"]
    assert payload["mask_rules"]["analysis_roles"] == [
        "target",
        "analysis_support",
    ]


def test_update_scene_settings_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        update_scene_settings(
            _basic_tube_text(),
            scene_id="bad",
            description="bad",
            shape="[32, 0, 32]",
            spacing="[1.0, 1.0, 1.0]",
            label_mode="priority",
            scalar_blend="overwrite",
            overlap_policy="warn",
            target_roles="[target]",
            analysis_roles="[target]",
        )


def test_update_scene_settings_rejects_bad_composition_value() -> None:
    with pytest.raises(ValueError, match="Unknown scalar_blend"):
        update_scene_settings(
            _basic_tube_text(),
            scene_id="bad",
            description="bad",
            shape="[32, 32, 32]",
            spacing="[1.0, 1.0, 1.0]",
            label_mode="priority",
            scalar_blend="not_a_mode",
            overlap_policy="warn",
            target_roles="[target]",
            analysis_roles="[target]",
        )


def test_format_helpers() -> None:
    assert format_numeric_list([1, 2, 3]) == "[1, 2, 3]"
    assert format_string_list(["target", "analysis_support"]) == (
        "[target, analysis_support]"
    )
