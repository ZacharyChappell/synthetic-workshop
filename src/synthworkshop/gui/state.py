"""Pure helper functions for the optional GUI workbench.

This module deliberately avoids importing Streamlit so that the core package and
test suite can use GUI-adjacent helpers without requiring optional GUI
dependencies.
"""

from __future__ import annotations

from pathlib import Path

from synthworkshop.datasets import (
    CatalogueEntry,
    get_catalogue_entry,
    list_catalogue_entries,
)
from synthworkshop.scenes.validation import SceneValidationReport, validate_scene_config
from synthworkshop.workflows import SceneWorkflowResult, render_export_gallery


def catalogue_rows() -> list[dict[str, str]]:
    """Return catalogue entries as display-friendly rows."""

    return [entry.to_row() for entry in list_catalogue_entries()]


def catalogue_scene_ids() -> list[str]:
    """Return built-in catalogue scene IDs."""

    return [entry.scene_id for entry in list_catalogue_entries()]


def default_output_root(scene_id: str) -> Path:
    """Return the default GUI output directory for a scene."""

    return Path("outputs") / "gui" / scene_id


def read_scene_text(entry: CatalogueEntry) -> str:
    """Read a catalogue scene configuration as text."""

    return entry.config_path.read_text(encoding="utf-8")


def write_gui_scene_config(
    *,
    scene_id: str,
    text: str,
    output_root: str | Path,
) -> Path:
    """Write edited scene YAML to the GUI output directory."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{scene_id}__gui_edit.yml"
    path.write_text(text, encoding="utf-8")
    return path


def validate_catalogue_scene(
    scene_id: str,
    *,
    render: bool = False,
) -> SceneValidationReport:
    """Validate a built-in catalogue scene."""

    entry = get_catalogue_entry(scene_id)
    return validate_scene_config(entry.config_path, render=render)


def render_catalogue_scene(
    scene_id: str,
    *,
    output_root: str | Path,
    map_name: str | None = None,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 200,
    overwrite: bool = False,
    with_colorbar: bool = False,
) -> SceneWorkflowResult:
    """Render a built-in catalogue scene through the shared workflow layer."""

    entry = get_catalogue_entry(scene_id)
    return render_export_gallery(
        entry.config_path,
        output_root=output_root,
        map_name=map_name,
        export=True,
        gallery=True,
        formats=formats,
        dpi=dpi,
        overwrite=overwrite,
        with_colorbar=with_colorbar,
    )


def render_scene_config_text(
    *,
    scene_id: str,
    text: str,
    output_root: str | Path,
    map_name: str | None = None,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 200,
    overwrite: bool = False,
    with_colorbar: bool = False,
) -> SceneWorkflowResult:
    """Render edited scene YAML text through the shared workflow layer."""

    config_path = write_gui_scene_config(
        scene_id=scene_id,
        text=text,
        output_root=output_root,
    )
    return render_export_gallery(
        config_path,
        output_root=output_root,
        map_name=map_name,
        export=True,
        gallery=True,
        formats=formats,
        dpi=dpi,
        overwrite=overwrite,
        with_colorbar=with_colorbar,
    )


def render_preview_scene_config_text(
    *,
    scene_id: str,
    text: str,
    output_root: str | Path,
    map_name: str | None = None,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 160,
    overwrite: bool = True,
    with_colorbar: bool = False,
    preview_name: str = "placement_preview",
) -> SceneWorkflowResult:
    """Render edited scene YAML text for GUI preview only.

    This writes gallery figures but deliberately skips array/table/metadata
    export so interactive checks stay lightweight.
    """

    preview_root = Path(output_root) / preview_name
    config_path = write_gui_scene_config(
        scene_id=scene_id,
        text=text,
        output_root=preview_root,
    )
    return render_export_gallery(
        config_path,
        output_root=preview_root,
        map_name=map_name,
        export=False,
        gallery=True,
        formats=formats,
        dpi=dpi,
        overwrite=overwrite,
        with_colorbar=with_colorbar,
    )


def validate_scene_config_text(
    *,
    scene_id: str,
    text: str,
    output_root: str | Path,
    render: bool = False,
) -> SceneValidationReport:
    """Validate edited scene YAML text."""

    config_path = write_gui_scene_config(
        scene_id=scene_id,
        text=text,
        output_root=output_root,
    )
    return validate_scene_config(config_path, render=render)


def gallery_png_paths(output_root: str | Path) -> list[Path]:
    """Return PNG gallery outputs, sorted for deterministic display."""

    gallery_dir = Path(output_root) / "gallery"
    if not gallery_dir.exists():
        return []
    return sorted(gallery_dir.rglob("*.png"))
