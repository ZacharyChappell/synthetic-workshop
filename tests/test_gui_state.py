from __future__ import annotations

from pathlib import Path

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.state import (
    catalogue_rows,
    catalogue_scene_ids,
    default_output_root,
    gallery_png_paths,
    read_scene_text,
    validate_catalogue_scene,
)


def test_gui_catalogue_rows_include_basic_tube() -> None:
    rows = catalogue_rows()
    scene_ids = {row["scene_id"] for row in rows}

    assert "basic_tube" in scene_ids


def test_gui_catalogue_scene_ids_include_basic_tube() -> None:
    scene_ids = catalogue_scene_ids()

    assert "basic_tube" in scene_ids


def test_gui_default_output_root_is_under_outputs_gui() -> None:
    path = default_output_root("basic_tube")

    assert path == Path("outputs") / "gui" / "basic_tube"


def test_gui_read_scene_text_reads_catalogue_config() -> None:
    entry = get_catalogue_entry("basic_tube")
    text = read_scene_text(entry)

    assert "basic_tube" in text
    assert "objects" in text


def test_gui_validate_catalogue_scene() -> None:
    report = validate_catalogue_scene("basic_tube", render=False)

    assert report.passed
    assert report.scene_id == "basic_tube"


def test_gui_gallery_png_paths_returns_sorted_pngs(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    second = gallery / "b.png"
    first = gallery / "a.png"
    other = gallery / "ignore.txt"

    second.write_text("", encoding="utf-8")
    first.write_text("", encoding="utf-8")
    other.write_text("", encoding="utf-8")

    assert gallery_png_paths(tmp_path) == [first, second]


def test_gui_render_preview_scene_config_text_writes_gallery_only(
    tmp_path: Path,
) -> None:
    from synthworkshop.gui.state import render_preview_scene_config_text

    entry = get_catalogue_entry("basic_tube")
    text = read_scene_text(entry)

    result = render_preview_scene_config_text(
        scene_id="basic_tube",
        text=text,
        output_root=tmp_path,
        formats=("png",),
        overwrite=True,
    )

    preview_root = tmp_path / "placement_preview"

    assert result.gallery_written
    assert not result.exported
    assert (preview_root / "gallery").exists()
    assert not (preview_root / "export").exists()
    assert gallery_png_paths(preview_root)
