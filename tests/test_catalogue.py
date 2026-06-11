from __future__ import annotations

import pytest

from synthworkshop.datasets import (
    get_catalogue_entry,
    list_catalogue_entries,
)


def test_list_catalogue_entries_returns_built_in_examples() -> None:
    entries = list_catalogue_entries()

    scene_ids = {entry.scene_id for entry in entries}

    assert "basic_tube" in scene_ids
    assert "curved_elliptic_tube" in scene_ids
    assert "tube_with_implicit_objects" in scene_ids


def test_catalogue_entries_have_existing_config_paths() -> None:
    entries = list_catalogue_entries()

    for entry in entries:
        assert entry.config_path.exists(), entry.config_path
        assert entry.config_path.suffix in {".yml", ".yaml", ".json"}


def test_get_catalogue_entry_returns_one_scene() -> None:
    entry = get_catalogue_entry("basic_tube")

    assert entry.scene_id == "basic_tube"
    assert entry.family == "control"
    assert entry.config_path.name == "basic_tube.yml"
    assert "tube" in entry.title.lower()


def test_get_catalogue_entry_rejects_unknown_scene() -> None:
    with pytest.raises(KeyError, match="Unknown catalogue scene_id"):
        get_catalogue_entry("not_a_scene")


def test_catalogue_entries_have_unique_scene_ids() -> None:
    entries = list_catalogue_entries()

    scene_ids = [entry.scene_id for entry in entries]

    assert len(scene_ids) == len(set(scene_ids))


def test_catalogue_entries_have_validation_metadata() -> None:
    entries = list_catalogue_entries()

    for entry in entries:
        assert entry.purpose
        assert entry.expected_appearance
        assert entry.expected_failure_mode
        assert entry.recommended_use
        assert entry.validation_focus
        assert entry.tags
        assert entry.default_output_name == entry.scene_id


def test_catalogue_rows_include_phase_three_metadata() -> None:
    entry = get_catalogue_entry("known_effect_tube")
    row = entry.to_row()

    assert row["expected_failure_mode"]
    assert row["recommended_use"]
    assert "known-effect" in row["tags"]
    assert row["default_output_name"] == "known_effect_tube"


def test_catalogue_includes_perturbation_and_known_effect_examples() -> None:
    scene_ids = {entry.scene_id for entry in list_catalogue_entries()}

    assert "perturbed_tube" in scene_ids
    assert "known_effect_tube" in scene_ids
