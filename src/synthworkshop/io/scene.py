"""Export rendered synthetic scenes to arrays, tables, and metadata."""

from __future__ import annotations

import json as _json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthworkshop.io.arrays import write_array
from synthworkshop.io.json import to_jsonable, write_json
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
                        "relative_path": _relative_path(path, self.output_root),
                        "format": Path(path).suffix.lstrip("."),
                    }
                )

        return pd.DataFrame(
            rows,
            columns=["group", "role", "path", "relative_path", "format"],
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable manifest dictionary."""

        return {
            "output_root": self.output_root,
            "arrays": dict(self.arrays),
            "tables": dict(self.tables),
            "metadata": dict(self.metadata),
        }


def _relative_path(path: str | Path, root: str | Path) -> str:
    """Return a display-friendly path relative to the export root when possible."""

    out_path = Path(path)
    out_root = Path(root)
    try:
        return str(out_path.relative_to(out_root))
    except ValueError:
        return str(out_path)


def _json_text(value: Any) -> str:
    """Return a compact JSON representation for TSV cells."""

    return _json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False)


def _cell_value(value: Any) -> object:
    """Return a stable scalar or compact JSON text for table cells."""

    value = to_jsonable(value)

    if value is None or isinstance(value, str | int | float | bool):
        return value

    return _json_text(value)


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
    """Collect canonical and scene-specific tables to write for a rendered scene."""

    tables: dict[str, pd.DataFrame] = {
        "object_table": _object_table(scene),
        "scene_manifest": _scene_manifest_table(scene),
        "scalar_map_manifest": _scalar_map_manifest_table(scene),
        "distance_map_manifest": _distance_map_manifest_table(scene),
        "perturbation_table": _perturbation_table(scene),
        "effect_table": _effect_table(scene),
    }

    if scene.centrelines:
        tables["centrelines"] = _combined_named_tables(
            scene.centrelines, id_name="object_id"
        )

    if scene.frames:
        tables["frames"] = _combined_named_tables(scene.frames, id_name="object_id")

    for name, table in scene.truth.tables.items():
        tables.setdefault(safe_name(name), table)

    for name, table in scene.centrelines.items():
        tables.setdefault(f"centreline__{safe_name(name)}", table)

    for name, table in scene.frames.items():
        tables.setdefault(f"frame__{safe_name(name)}", table)

    return tables


def _object_table(scene: RenderedScene) -> pd.DataFrame:
    """Return one row per rendered object mask."""

    rows: list[dict[str, object]] = []

    for object_id, metadata in scene.object_metadata.items():
        mask = scene.object_masks[object_id]
        role = getattr(metadata.role, "value", metadata.role)

        rows.append(
            {
                "object_id": object_id,
                "role": str(role),
                "label": int(metadata.label),
                "priority": int(metadata.priority),
                "name": metadata.name or "",
                "description": metadata.description or "",
                "n_voxels": int(np.sum(mask)),
                "metadata": _json_text(metadata.metadata),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "object_id",
            "role",
            "label",
            "priority",
            "name",
            "description",
            "n_voxels",
            "metadata",
        ],
    )


def _scene_manifest_table(scene: RenderedScene) -> pd.DataFrame:
    """Return compact scene-level export metadata."""

    rows = [
        ("scene_id", scene.metadata.get("scene_id", "")),
        ("ndim", len(scene.grid.shape)),
        ("shape", scene.grid.shape),
        ("spacing", scene.grid.spacing),
        ("scalar_maps", tuple(scene.scalar_maps)),
        ("object_ids", tuple(scene.object_masks)),
        ("target_masks", tuple(scene.target_masks)),
        ("analysis_masks", tuple(scene.analysis_masks)),
        ("skeleton_masks", tuple(scene.skeleton_masks)),
        ("distance_maps", tuple(scene.distance_maps)),
        ("signed_offset_maps", tuple(scene.signed_offset_maps)),
        ("n_objects", len(scene.object_masks)),
        ("n_scalar_maps", len(scene.scalar_maps)),
        ("overlap_voxels", scene.overlap_report.n_overlap_voxels),
        ("overlap_policy", scene.overlap_report.policy.value),
    ]

    return pd.DataFrame(
        [{"field": key, "value": _cell_value(value)} for key, value in rows],
        columns=["field", "value"],
    )


def _scalar_map_manifest_table(scene: RenderedScene) -> pd.DataFrame:
    """Return one row per scalar map."""

    rows: list[dict[str, object]] = []

    for map_name, array in scene.scalar_maps.items():
        data = np.asarray(array, dtype=float)
        finite = np.isfinite(data)

        rows.append(
            {
                "map_name": map_name,
                "array_name": f"scalar__{safe_name(map_name)}",
                "shape": _json_text(data.shape),
                "dtype": str(data.dtype),
                "finite_voxels": int(finite.sum()),
                "nan_voxels": int(np.isnan(data).sum()),
                "min": float(np.nanmin(data)) if finite.any() else np.nan,
                "max": float(np.nanmax(data)) if finite.any() else np.nan,
                "mean": float(np.nanmean(data)) if finite.any() else np.nan,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "map_name",
            "array_name",
            "shape",
            "dtype",
            "finite_voxels",
            "nan_voxels",
            "min",
            "max",
            "mean",
        ],
    )


def _distance_map_manifest_table(scene: RenderedScene) -> pd.DataFrame:
    """Return one row per distance-like array."""

    rows: list[dict[str, object]] = []

    for map_name, array in scene.distance_maps.items():
        rows.append(_distance_row(map_name, array, kind="distance"))

    for map_name, array in scene.signed_offset_maps.items():
        rows.append(_distance_row(map_name, array, kind="signed_offset"))

    return pd.DataFrame(
        rows,
        columns=[
            "kind",
            "map_name",
            "array_name",
            "shape",
            "dtype",
            "finite_voxels",
            "min",
            "max",
        ],
    )


def _distance_row(map_name: str, array: Any, *, kind: str) -> dict[str, object]:
    """Return a manifest row for one distance-like map."""

    data = np.asarray(array, dtype=float)
    finite = np.isfinite(data)
    prefix = "distance" if kind == "distance" else "signed_offset"

    return {
        "kind": kind,
        "map_name": map_name,
        "array_name": f"{prefix}__{safe_name(map_name)}",
        "shape": _json_text(data.shape),
        "dtype": str(data.dtype),
        "finite_voxels": int(finite.sum()),
        "min": float(np.nanmin(data)) if finite.any() else np.nan,
        "max": float(np.nanmax(data)) if finite.any() else np.nan,
    }


def _perturbation_table(scene: RenderedScene) -> pd.DataFrame:
    """Return perturbation metadata as a tabular manifest."""

    records: Any = scene.truth.perturbations
    if not records:
        records = scene.metadata.get("perturbations", ())

    return _record_table(
        records,
        default_columns=[
            "record_id",
            "name",
            "target",
            "affected_arrays",
            "affected_maps",
            "affected_objects",
            "parameters",
            "seed",
            "truth_changed",
            "observation_changed",
            "metadata",
        ],
    )


def _effect_table(scene: RenderedScene) -> pd.DataFrame:
    """Return known-effect metadata as a tabular manifest."""

    records: Any = {}
    if isinstance(scene.truth.metadata, Mapping):
        records = scene.truth.metadata.get("effects", {})

    if not records:
        records = scene.metadata.get("effects", ())

    return _record_table(
        records,
        default_columns=[
            "record_id",
            "name",
            "target",
            "affected_maps",
            "affected_objects",
            "support_voxels",
            "magnitude",
            "expected_direction",
            "clean_null",
            "truth_changed",
            "parameters",
            "metadata",
        ],
    )


def _record_table(
    records: Mapping[str, Any] | Sequence[Any],
    *,
    default_columns: Sequence[str],
) -> pd.DataFrame:
    """Normalise a mapping or sequence of metadata records to a DataFrame."""

    rows: list[dict[str, object]] = []

    if isinstance(records, Mapping):
        iterator = records.items()
    else:
        iterator = ((f"{index + 1:03d}", item) for index, item in enumerate(records))

    for record_id, record in iterator:
        payload = dict(record) if isinstance(record, Mapping) else {"metadata": record}
        row: dict[str, object] = {"record_id": str(record_id)}
        for key, value in payload.items():
            row[str(key)] = _cell_value(value)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(default_columns))

    seen = set(default_columns)
    extra_columns = sorted(
        {column for row in rows for column in row} - seen,
    )
    columns = list(default_columns) + extra_columns

    return pd.DataFrame(rows).reindex(columns=columns)


def _combined_named_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    id_name: str,
) -> pd.DataFrame:
    """Combine named tables while preserving their source identifier."""

    rows: list[pd.DataFrame] = []

    for name, table in tables.items():
        current = table.copy()
        if id_name not in current.columns:
            current.insert(0, id_name, name)
        rows.append(current)

    if not rows:
        return pd.DataFrame(columns=[id_name])

    return pd.concat(rows, ignore_index=True, sort=False)


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
    """Write scene tables."""

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
        "composition_metadata": {
            "composition": {
                "label_mode": scene.composition.label_mode.value,
                "scalar_blend": scene.composition.scalar_blend.value,
                "overlap_policy": scene.composition.overlap_policy.value,
            },
            "mask_rules": {
                "target_roles": tuple(
                    role.value for role in scene.mask_rules.target_roles
                ),
                "analysis_roles": tuple(
                    role.value for role in scene.mask_rules.analysis_roles
                ),
            },
        },
        "overlap_report": scene.overlap_report.to_dict(),
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
