from __future__ import annotations

from pathlib import Path

import pytest

from synthworkshop.workflows import (
    SceneWorkflowResult,
    choose_default_map_name,
    render_export_gallery,
)


def test_choose_default_map_name_prefers_fa_like() -> None:
    result = render_export_gallery(
        "examples/basic_tube.yml",
        export=False,
        gallery=False,
    )

    assert choose_default_map_name(result.scene) == "fa_like"


def test_render_export_gallery_can_render_without_writing_outputs() -> None:
    result = render_export_gallery(
        "examples/basic_tube.yml",
        export=False,
        gallery=False,
    )

    assert isinstance(result, SceneWorkflowResult)
    assert result.map_name == "fa_like"
    assert result.scene.metadata["scene_id"] == "basic_tube"
    assert not result.exported
    assert not result.gallery_written


def test_render_export_gallery_writes_export_and_gallery(tmp_path: Path) -> None:
    result = render_export_gallery(
        "examples/basic_tube.yml",
        output_root=tmp_path,
        export=True,
        gallery=True,
        formats=("png",),
        overwrite=True,
    )

    assert result.exported
    assert result.gallery_written

    assert result.export_manifest is not None
    assert (tmp_path / "export" / "tables" / "export_manifest.tsv").exists()
    assert (tmp_path / "export" / "metadata" / "export_manifest.json").exists()

    assert result.gallery_paths is not None
    assert "best_projection" in result.gallery_paths
    assert "orthogonal_slices" in result.gallery_paths
    assert "legend" in result.gallery_paths


def test_render_export_gallery_requires_output_root_when_exporting() -> None:
    with pytest.raises(ValueError, match="output_root is required"):
        render_export_gallery(
            "examples/basic_tube.yml",
            export=True,
            gallery=False,
        )


def test_render_export_gallery_rejects_unknown_map_name() -> None:
    with pytest.raises(ValueError, match="Unknown map_name"):
        render_export_gallery(
            "examples/basic_tube.yml",
            map_name="not_a_map",
            export=False,
            gallery=False,
        )
