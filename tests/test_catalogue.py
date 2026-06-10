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
