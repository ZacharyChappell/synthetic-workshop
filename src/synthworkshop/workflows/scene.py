"""High-level scene rendering, export, and gallery workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synthworkshop.io import SceneExportManifest, export_scene
from synthworkshop.plotting import write_scene_gallery
from synthworkshop.scenes import RenderedScene
from synthworkshop.scenes.config import render_scene_from_path


@dataclass(frozen=True)
class SceneWorkflowResult:
    """Outputs from a high-level scene workflow run."""

    scene: RenderedScene
    map_name: str
    export_manifest: SceneExportManifest | None = None
    gallery_paths: dict[str, list[Path] | Path] | None = None

    @property
    def exported(self) -> bool:
        """Return True if array/table/metadata export was requested."""

        return self.export_manifest is not None

    @property
    def gallery_written(self) -> bool:
        """Return True if gallery output was requested."""

        return self.gallery_paths is not None


def choose_default_map_name(scene: RenderedScene) -> str:
    """Choose a deterministic scalar map for preview/gallery output."""

    if not scene.scalar_maps:
        raise ValueError("Rendered scene contains no scalar maps.")

    preferred = (
        "fa_like",
        "scalar",
        "md_like",
        "qsm_like",
        "r2star_like",
        "wm_pve_like",
    )
    for name in preferred:
        if name in scene.scalar_maps:
            return name

    return sorted(scene.scalar_maps)[0]


def render_export_gallery(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    map_name: str | None = None,
    export: bool = True,
    gallery: bool = True,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    overwrite: bool = False,
    with_colorbar: bool = False,
) -> SceneWorkflowResult:
    """Render a scene spec and optionally write export and gallery outputs.

    This workflow is intentionally small. It is the shared service layer that a
    future CLI and GUI can call without duplicating render/export/gallery logic.
    """

    scene = render_scene_from_path(config_path)
    selected_map = map_name or choose_default_map_name(scene)

    if selected_map not in scene.scalar_maps:
        available = ", ".join(sorted(scene.scalar_maps))
        raise ValueError(
            f"Unknown map_name {selected_map!r}. Available scalar maps: {available}"
        )

    root = Path(output_root) if output_root is not None else None

    export_manifest = None
    if export:
        if root is None:
            raise ValueError("output_root is required when export=True.")
        export_manifest = export_scene(
            scene,
            root / "export",
            overwrite=overwrite,
            extra_metadata={
                "source_config": str(Path(config_path)),
                "workflow": "render_export_gallery",
                "gallery_map_name": selected_map,
            },
        )

    gallery_paths = None
    if gallery:
        if root is None:
            raise ValueError("output_root is required when gallery=True.")
        gallery_paths = write_scene_gallery(
            scene,
            root / "gallery",
            map_name=selected_map,
            formats=formats,
            dpi=dpi,
            overwrite=overwrite,
            with_colorbar=with_colorbar,
        )

    return SceneWorkflowResult(
        scene=scene,
        map_name=selected_map,
        export_manifest=export_manifest,
        gallery_paths=gallery_paths,
    )
