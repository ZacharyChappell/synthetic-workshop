from __future__ import annotations

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.state import read_scene_text
from synthworkshop.gui.yaml_editor import (
    add_object_to_scene_text,
    apply_field_edits,
    duplicate_scene_text,
    flatten_editable_fields,
    format_edit_value,
    get_object,
    make_minimal_tube_scene_text,
    make_sphere_object,
    make_tube_object,
    object_ids,
    object_summary_rows,
    parse_edit_value,
    parse_scene_text,
    replace_object,
    scene_grid_centre,
    suggest_next_label,
    suggest_object_id,
)
from synthworkshop.scenes.config import render_scene_from_dict


def _basic_tube_text() -> str:
    return read_scene_text(get_catalogue_entry("basic_tube"))


def test_object_summary_rows_finds_basic_tube_target() -> None:
    rows = object_summary_rows(_basic_tube_text())

    assert rows
    assert any(row["id"] == "target" for row in rows)


def test_object_ids_returns_scene_order_ids() -> None:
    ids = object_ids(_basic_tube_text())

    assert "target" in ids


def test_get_object_returns_copy_of_target() -> None:
    obj = get_object(_basic_tube_text(), "target")

    assert obj["id"] == "target"
    assert obj["kind"] == "tube"


def test_flatten_editable_fields_exposes_nested_tube_fields() -> None:
    obj = get_object(_basic_tube_text(), "target")
    fields = flatten_editable_fields(obj)

    assert "role" in fields
    assert "label" in fields
    assert "map_name" in fields
    assert any(key.startswith("curve.") for key in fields)
    assert any(key.startswith("cross_section.") for key in fields)
    assert any(key.startswith("profile.") for key in fields)


def test_apply_field_edits_updates_common_and_nested_fields() -> None:
    obj = get_object(_basic_tube_text(), "target")

    updated = apply_field_edits(
        obj,
        {
            "role": "analysis_support",
            "label": "7",
            "cross_section.radius_mm": "4.5",
            "profile.centre_value": "2.0",
        },
    )

    assert updated["role"] == "analysis_support"
    assert updated["label"] == 7
    assert updated["cross_section"]["radius_mm"] == 4.5
    assert updated["profile"]["centre_value"] == 2.0


def test_replace_object_round_trips_updated_object() -> None:
    text = _basic_tube_text()
    obj = get_object(text, "target")
    updated = apply_field_edits(obj, {"label": "9"})

    new_text = replace_object(text, "target", updated)
    new_obj = get_object(new_text, "target")

    assert new_obj["label"] == 9


def test_parse_edit_value_handles_scalars_and_lists() -> None:
    assert parse_edit_value("3.5") == 3.5
    assert parse_edit_value("7") == 7
    assert parse_edit_value("true") is True
    assert parse_edit_value("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]
    assert parse_edit_value("target") == "target"


def test_format_edit_value_formats_lists_for_text_input() -> None:
    assert format_edit_value([1.0, 2.0, 3.0]) == "[1.0, 2.0, 3.0]"


def test_make_minimal_tube_scene_text_renders() -> None:
    text = make_minimal_tube_scene_text(scene_id="unit_new_scene")
    payload = parse_scene_text(text)
    scene = render_scene_from_dict(payload)

    assert scene.metadata["scene_id"] == "unit_new_scene"
    assert "target" in scene.object_masks
    assert "fa_like" in scene.scalar_maps


def test_duplicate_scene_text_updates_scene_id() -> None:
    text = duplicate_scene_text(
        _basic_tube_text(),
        new_scene_id="basic_tube_copy",
    )
    payload = parse_scene_text(text)

    assert payload["scene"]["id"] == "basic_tube_copy"


def test_suggest_next_label_and_object_id() -> None:
    text = _basic_tube_text()

    assert suggest_next_label(text) == 2
    assert suggest_object_id(text, "target") == "target_2"
    assert suggest_object_id(text, "new_tube") == "new_tube"


def test_scene_grid_centre_from_basic_tube() -> None:
    centre = scene_grid_centre(_basic_tube_text())

    assert centre == [16.0, 16.0, 16.0]


def test_add_tube_object_to_scene_text_renders() -> None:
    text = _basic_tube_text()
    obj = make_tube_object(
        object_id="support_tube",
        role="analysis_support",
        label=2,
        priority=1,
        map_name="fa_like",
        start_mm=[8.0, 22.0, 16.0],
        end_mm=[24.0, 22.0, 16.0],
        radius_mm=2.0,
        profile_kind="constant",
    )

    updated_text = add_object_to_scene_text(text, obj)
    payload = parse_scene_text(updated_text)
    scene = render_scene_from_dict(payload)

    assert "support_tube" in scene.object_masks


def test_add_sphere_object_to_scene_text_renders() -> None:
    text = _basic_tube_text()
    obj = make_sphere_object(
        object_id="sphere_inclusion",
        role="inclusion",
        label=2,
        priority=5,
        map_name="qsm_like",
        centre_mm=[16.0, 21.0, 16.0],
        radius_mm=2.5,
        value=1.25,
    )

    updated_text = add_object_to_scene_text(text, obj)
    payload = parse_scene_text(updated_text)
    scene = render_scene_from_dict(payload)

    assert "sphere_inclusion" in scene.object_masks
    assert "qsm_like" in scene.scalar_maps


def test_duplicate_object_in_scene_text_adds_translated_copy() -> None:
    from synthworkshop.gui.yaml_editor import duplicate_object_in_scene_text

    text = duplicate_object_in_scene_text(
        _basic_tube_text(),
        "target",
        new_object_id="target_copy",
        new_label=2,
        offset_mm=[0.0, 4.0, 0.0],
    )

    ids = object_ids(text)
    copied = get_object(text, "target_copy")

    assert "target_copy" in ids
    assert copied["label"] == 2
    assert (
        copied["curve"]["start_mm"][1]
        == get_object(
            _basic_tube_text(),
            "target",
        )["curve"]["start_mm"][1]
        + 4.0
    )

    scene = render_scene_from_dict(parse_scene_text(text))
    assert "target_copy" in scene.object_masks


def test_delete_object_from_scene_text_removes_object() -> None:
    from synthworkshop.gui.yaml_editor import (
        delete_object_from_scene_text,
        duplicate_object_in_scene_text,
    )

    text = duplicate_object_in_scene_text(
        _basic_tube_text(),
        "target",
        new_object_id="target_copy",
        new_label=2,
        offset_mm=[0.0, 4.0, 0.0],
    )
    text = delete_object_from_scene_text(text, "target_copy")

    assert "target_copy" not in object_ids(text)


def test_delete_object_from_scene_text_rejects_final_object() -> None:
    import pytest

    from synthworkshop.gui.yaml_editor import delete_object_from_scene_text

    with pytest.raises(ValueError, match="Cannot delete the final object"):
        delete_object_from_scene_text(_basic_tube_text(), "target")


def test_duplicate_object_in_scene_text_rejects_duplicate_id() -> None:
    import pytest

    from synthworkshop.gui.yaml_editor import duplicate_object_in_scene_text

    with pytest.raises(ValueError, match="already exists"):
        duplicate_object_in_scene_text(
            _basic_tube_text(),
            "target",
            new_object_id="target",
            new_label=2,
        )
