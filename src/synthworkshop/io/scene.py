"""Export rendered synthetic scenes to arrays, tables, and metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from synthworkshop.io.arrays import write_array
from synthworkshop.io.json import write_json
from synthworkshop.io.tables import write_table
from synthworkshop.scenes import RenderedScene

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(value: object) -> str:
    """Return a filesystem-safe identifier."""

    text = str(value).strip()
    text = _SAFE_NAME_RE.sub("_", text)
    text = text.strip("._-")
    if not text:
        raise ValueError("Cannot create a safe name from an empty value.")
    return text


@dataclass(frozen=True)
class SceneExportManifest:
    """Paths written by a scene export operation."""

    output_root: Path
    arrays: Mapping[str, Path]
    tables: Mapping[str, Path]
    metadata: Mapping[str, Path]

    def to_dataframe(self) -> pd.DataFrame:
        """Return export paths as a manifest table."""

        rows: list[dict[str, str]] = []
        for group_name, paths in (
            ("array", self.arrays),
            ("table", self.tables),
            ("metadata", self.metadata),
        ):
            for role, path in paths.items():
                rows.append(
                    {
                        "group": group_name,
                        "role": role,
                        "path": str(path),
                    }
                )
        return pd.DataFrame(rows, columns=["group", "role", "path"])

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable manifest dictionary."""

        return {
            "output_root": self.output_root,
            "arrays": dict(self.arrays),
            "tables": dict(self.tables),
            "metadata": dict(self.metadata),
        }


def _scene_arrays(scene: RenderedScene) -> dict[str, object]:
    """Collect arrays to write for a rendered scene."""

    arrays: dict[str, object] = {"labels": scene.label_map}

    for name, array in scene.scalar_maps.items():
        arrays[f"scalar__{safe_name(name)}"] = array

    for name, array in scene.object_masks.items():
        arrays[f"object_mask__{safe_name(name)}"] = array

    for name, array in scene.target_masks.items():
        arrays[f"target_mask__{safe_name(name)}"] = array

    for name, array in scene.analysis_masks.items():
        arrays[f"analysis_mask__{safe_name(name)}"] = array

    for name, array in scene.skeleton_masks.items():
        arrays[f"skeleton_mask__{safe_name(name)}"] = array

    for name, array in scene.distance_maps.items():
        arrays[f"distance__{safe_name(name)}"] = array

    for name, array in scene.signed_offset_maps.items():
        arrays[f"signed_offset__{safe_name(name)}"] = array

    return arrays


def _scene_tables(scene: RenderedScene) -> dict[str, pd.DataFrame]:
    """Collect tables to write for a rendered scene."""

    tables = {safe_name(name): table for name, table in scene.truth.tables.items()}

    for name, table in scene.centrelines.items():
        tables.setdefault(f"centreline__{safe_name(name)}", table)

    for name, table in scene.frames.items():
        tables.setdefault(f"frame__{safe_name(name)}", table)

    return tables


def _write_arrays(
    scene: RenderedScene,
    arrays_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    """Write scene arrays."""

    written: dict[str, Path] = {}
    for role, array in _scene_arrays(scene).items():
        written[role] = write_array(
            array,
            arrays_dir / f"{role}.npy",
            overwrite=overwrite,
        )
    return written


def _write_tables(
    scene: RenderedScene,
    tables_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    """Write scene truth tables."""

    written: dict[str, Path] = {}
    for role, table in _scene_tables(scene).items():
        written[role] = write_table(
            table,
            tables_dir / f"{role}.tsv",
            overwrite=overwrite,
        )
    return written


def _write_metadata(
    scene: RenderedScene,
    metadata_dir: Path,
    *,
    overwrite: bool,
    extra_metadata: Mapping[str, Any] | None,
) -> dict[str, Path]:
    """Write scene metadata files."""

    metadata = {
        "grid": scene.grid.summary(),
        "scene_summary": scene.summary(),
        "truth_summary": scene.truth.summary(),
        "render_metadata": dict(scene.metadata),
        "provenance": dict(scene.provenance),
    }
    if extra_metadata:
        metadata["extra_metadata"] = dict(extra_metadata)

    written: dict[str, Path] = {}
    for role, payload in metadata.items():
        written[role] = write_json(
            payload,
            metadata_dir / f"{role}.json",
            overwrite=overwrite,
        )
    return written


def export_scene(
    scene: RenderedScene,
    output_root: str | Path,
    *,
    overwrite: bool = False,
    include_arrays: bool = True,
    include_tables: bool = True,
    include_metadata: bool = True,
    extra_metadata: Mapping[str, Any] | None = None,
) -> SceneExportManifest:
    """Export a rendered scene to a standard directory layout."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Path] = {}
    tables: dict[str, Path] = {}
    metadata: dict[str, Path] = {}

    if include_arrays:
        arrays = _write_arrays(scene, root / "arrays", overwrite=overwrite)

    if include_tables:
        tables = _write_tables(scene, root / "tables", overwrite=overwrite)

    if include_metadata:
        metadata = _write_metadata(
            scene,
            root / "metadata",
            overwrite=overwrite,
            extra_metadata=extra_metadata,
        )

    manifest = SceneExportManifest(
        output_root=root,
        arrays=arrays,
        tables=tables,
        metadata=metadata,
    )

    manifest_table = manifest.to_dataframe()
    manifest_path = write_table(
        manifest_table,
        root / "tables" / "export_manifest.tsv",
        overwrite=overwrite,
    )
    tables = {**tables, "export_manifest": manifest_path}

    manifest = SceneExportManifest(
        output_root=root,
        arrays=arrays,
        tables=tables,
        metadata=metadata,
    )

    if include_metadata:
        manifest_json = write_json(
            manifest.to_dict(),
            root / "metadata" / "export_manifest.json",
            overwrite=overwrite,
        )
        metadata = {**metadata, "export_manifest": manifest_json}
        manifest = SceneExportManifest(
            output_root=root,
            arrays=arrays,
            tables=tables,
            metadata=metadata,
        )

    return manifest
