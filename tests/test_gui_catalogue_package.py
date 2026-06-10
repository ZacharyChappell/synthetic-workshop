from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.catalogue_package import (
    CATALOGUE_FAMILIES,
    default_catalogue_package_dir,
    export_catalogue_package,
    format_focus_items,
    humanise_scene_id,
    parse_focus_items,
    read_catalogue_package,
    safe_slug,
    scene_id_from_text,
)
from synthworkshop.gui.state import read_scene_text


def _basic_tube_text() -> str:
    return read_scene_text(get_catalogue_entry("basic_tube"))


def test_catalogue_families_include_expected_values() -> None:
    assert "control" in CATALOGUE_FAMILIES
    assert "morphology" in CATALOGUE_FAMILIES
    assert "environment" in CATALOGUE_FAMILIES


def test_scene_id_from_text_reads_basic_tube() -> None:
    assert scene_id_from_text(_basic_tube_text()) == "basic_tube"


def test_safe_slug_and_humanise_scene_id() -> None:
    assert safe_slug("My scene / test") == "My_scene_test"
    assert humanise_scene_id("basic_tube") == "Basic Tube"


def test_parse_and_format_focus_items() -> None:
    items = parse_focus_items("tube rendering, overlap reporting\nprofile recovery")

    assert items == [
        "tube rendering",
        "overlap reporting",
        "profile recovery",
    ]
    assert format_focus_items(items) == (
        "tube rendering, overlap reporting, profile recovery"
    )


def test_default_catalogue_package_dir() -> None:
    path = default_catalogue_package_dir(
        output_root="outputs/gui/basic_tube",
        scene_id="basic_tube",
    )

    assert path == Path("outputs/gui/basic_tube/catalogue_entries/basic_tube")


def test_export_catalogue_package_writes_expected_files(tmp_path: Path) -> None:
    package = export_catalogue_package(
        text=_basic_tube_text(),
        package_dir=tmp_path / "package",
        title="Basic tube package",
        family="control",
        purpose="Test package export.",
        expected_appearance="A straight tube.",
        validation_focus="tube rendering, scalar profile",
        notes="Unit test.",
        overwrite=False,
        render_summary=True,
    )

    assert package.scene_path.exists()
    assert package.readme_path.exists()
    assert package.entry_path.exists()
    assert package.summary_path.exists()

    entry = json.loads(package.entry_path.read_text(encoding="utf-8"))
    assert entry["scene_id"] == "basic_tube"
    assert entry["family"] == "control"
    assert entry["validation_focus"] == [
        "tube rendering",
        "scalar profile",
    ]

    summary = json.loads(package.summary_path.read_text(encoding="utf-8"))
    assert summary["scene"]["id"] == "basic_tube"
    assert summary["render"]["attempted"]
    assert summary["render"]["passed"]


def test_export_catalogue_package_refuses_non_empty_overwrite(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "existing.txt").write_text("exists", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_catalogue_package(
            text=_basic_tube_text(),
            package_dir=package_dir,
            title="Basic tube package",
            family="control",
            purpose="Test package export.",
            expected_appearance="A straight tube.",
            overwrite=False,
        )


def test_export_catalogue_package_rejects_unknown_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown catalogue family"):
        export_catalogue_package(
            text=_basic_tube_text(),
            package_dir=tmp_path / "package",
            title="Bad family",
            family="not_a_family",
            purpose="Test.",
            expected_appearance="Test.",
        )


def test_read_catalogue_package_round_trip(tmp_path: Path) -> None:
    written = export_catalogue_package(
        text=_basic_tube_text(),
        package_dir=tmp_path / "package",
        title="Basic tube package",
        family="control",
        purpose="Test package export.",
        expected_appearance="A straight tube.",
        validation_focus=["tube rendering"],
        overwrite=False,
    )

    loaded = read_catalogue_package(written.package_dir)

    assert loaded.scene_text == written.scene_text
    assert loaded.entry["scene_id"] == "basic_tube"
    assert loaded.summary["scene"]["id"] == "basic_tube"


def test_read_catalogue_package_requires_scene_file(tmp_path: Path) -> None:
    package_dir = tmp_path / "empty_package"
    package_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="scene file is missing"):
        read_catalogue_package(package_dir)
