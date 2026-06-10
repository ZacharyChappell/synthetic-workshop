"""Scene-summary helpers for the optional GUI workbench."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from synthworkshop.gui.yaml_editor import object_id_for, parse_scene_text
from synthworkshop.scenes.config import render_scene_from_dict


def build_scene_summary(
    text: str,
    *,
    render: bool = False,
) -> dict[str, Any]:
    """Build a compact scene summary from YAML/JSON scene text."""

    payload = parse_scene_text(text)

    scene_meta = payload.get("scene", {})
    if not isinstance(scene_meta, dict):
        scene_meta = {}

    grid = payload.get("grid", {})
    if not isinstance(grid, dict):
        grid = {}

    composition = payload.get("composition", {})
    if not isinstance(composition, dict):
        composition = {}

    mask_rules = payload.get("mask_rules", {})
    if not isinstance(mask_rules, dict):
        mask_rules = {}

    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Scene field 'objects' must be a list.")

    object_rows = [_object_summary_row(index, obj) for index, obj in enumerate(objects)]
    valid_rows = [row for row in object_rows if row["valid_mapping"]]

    object_ids = [row["id"] for row in valid_rows if row["id"]]
    labels = [row["label"] for row in valid_rows if row["label"] is not None]
    map_names = sorted({row["map_name"] for row in valid_rows if row["map_name"]})

    summary: dict[str, Any] = {
        "schema_version": str(payload.get("schema_version", "")),
        "scene": {
            "id": str(scene_meta.get("id", "")),
            "description": str(scene_meta.get("description", "")),
        },
        "grid": {
            "shape": grid.get("shape"),
            "spacing": grid.get("spacing"),
            "origin": grid.get("origin"),
            "axis_names": grid.get("axis_names"),
            "n_dimensions": len(grid.get("shape", []))
            if isinstance(grid.get("shape"), list)
            else None,
        },
        "composition": {
            "label_mode": composition.get("label_mode"),
            "scalar_blend": composition.get("scalar_blend"),
            "overlap_policy": composition.get("overlap_policy"),
        },
        "mask_rules": {
            "target_roles": mask_rules.get("target_roles", ["target"]),
            "analysis_roles": mask_rules.get(
                "analysis_roles",
                ["target", "analysis_support"],
            ),
        },
        "objects": {
            "n_objects": len(objects),
            "n_valid_object_mappings": len(valid_rows),
            "ids": object_ids,
            "labels": labels,
            "map_names": map_names,
            "kind_counts": _counter_dict(row["kind"] for row in valid_rows),
            "role_counts": _counter_dict(row["role"] for row in valid_rows),
            "map_counts": _counter_dict(row["map_name"] for row in valid_rows),
            "duplicate_ids": _duplicates(object_ids),
            "duplicate_labels": _duplicates(labels),
            "rows": object_rows,
        },
        "render": {
            "attempted": False,
            "passed": None,
            "error": None,
        },
    }

    if render:
        summary["render"]["attempted"] = True
        try:
            rendered_scene = render_scene_from_dict(payload)
        except Exception as exc:
            summary["render"]["passed"] = False
            summary["render"]["error"] = f"{type(exc).__name__}: {exc}"
        else:
            summary["render"].update(_rendered_scene_summary(rendered_scene))

    return summary


def summary_to_json(summary: dict[str, Any]) -> str:
    """Serialise a scene summary as indented JSON."""

    return json.dumps(summary, indent=2, sort_keys=False)


def save_scene_summary_json(
    summary: dict[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a scene summary JSON file."""

    out_path = Path(path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing summary: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary_to_json(summary) + "\n", encoding="utf-8")
    return out_path


def default_summary_path(
    *,
    output_root: str | Path,
    scene_id: str,
) -> Path:
    """Return the default GUI summary JSON path."""

    return Path(output_root) / "metadata" / f"{scene_id}_scene_summary.json"


def _object_summary_row(index: int, obj: Any) -> dict[str, Any]:
    """Summarise one raw object mapping."""

    if not isinstance(obj, dict):
        return {
            "index": index,
            "valid_mapping": False,
            "id": "",
            "kind": "<invalid>",
            "role": "",
            "label": None,
            "priority": None,
            "map_name": "",
            "has_curve": False,
            "has_cross_section": False,
            "has_profile": False,
        }

    return {
        "index": index,
        "valid_mapping": True,
        "id": object_id_for(obj),
        "kind": str(obj.get("kind", "")),
        "role": str(obj.get("role", "")),
        "label": obj.get("label"),
        "priority": obj.get("priority"),
        "map_name": str(obj.get("map_name", "")),
        "has_curve": isinstance(obj.get("curve"), dict),
        "has_cross_section": isinstance(obj.get("cross_section"), dict),
        "has_profile": isinstance(obj.get("profile"), dict),
    }


def _rendered_scene_summary(scene: Any) -> dict[str, Any]:
    """Return a summary of a RenderedScene using stable/public attributes."""

    scalar_maps = getattr(scene, "scalar_maps", {})
    object_masks = getattr(scene, "object_masks", {})
    label_map = getattr(scene, "label_map", None)
    grid = getattr(scene, "grid", None)

    rendered: dict[str, Any] = {
        "passed": True,
        "error": None,
        "scalar_maps": sorted(scalar_maps),
        "n_scalar_maps": len(scalar_maps),
        "object_masks": sorted(object_masks),
        "n_object_masks": len(object_masks),
    }

    if label_map is not None:
        rendered["label_map_shape"] = list(label_map.shape)

    if grid is not None and hasattr(grid, "shape"):
        rendered["grid_shape"] = list(grid.shape)

    for attr in ("target_masks", "analysis_masks", "skeleton_masks"):
        value = getattr(scene, attr, None)
        if isinstance(value, dict):
            rendered[attr] = sorted(value)
            rendered[f"n_{attr}"] = len(value)

    metadata = getattr(scene, "metadata", None)
    if isinstance(metadata, dict):
        rendered["metadata_keys"] = sorted(str(key) for key in metadata)

    return rendered


def _counter_dict(values: list[Any] | tuple[Any, ...] | Any) -> dict[str, int]:
    """Return a JSON-friendly count dictionary."""

    return {
        str(key): int(value)
        for key, value in Counter(
            item for item in values if item not in {"", None}
        ).items()
    }


def _duplicates(values: list[Any]) -> list[Any]:
    """Return duplicated values in deterministic order."""

    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)
