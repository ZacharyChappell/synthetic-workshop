"""YAML editing helpers for the optional GUI workbench.

These helpers deliberately operate on plain dictionaries and YAML text. The GUI
therefore remains a thin editor over the public scene schema rather than a
separate hidden scene model.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "YAML editing requires PyYAML. Install the package with the YAML/IO "
        "dependencies enabled."
    ) from exc


EditableValue = str | int | float | bool | None | list[Any]


def parse_scene_text(text: str) -> dict[str, Any]:
    """Parse YAML scene text into a dictionary."""

    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Scene YAML must parse to a mapping at the top level.")
    return data


def dump_scene_dict(data: Mapping[str, Any]) -> str:
    """Serialise a scene dictionary as stable YAML text."""

    return yaml.safe_dump(
        dict(data),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def make_minimal_tube_scene_text(
    *,
    scene_id: str = "new_tube_scene",
    description: str = "New straight circular tube scene.",
    shape: list[int] | None = None,
    spacing: list[float] | None = None,
    map_name: str = "fa_like",
    radius_mm: float = 3.0,
) -> str:
    """Create a minimal renderable single-tube scene as YAML text."""

    shape = [32, 32, 32] if shape is None else shape
    spacing = [1.0, 1.0, 1.0] if spacing is None else spacing

    if len(shape) != 3:
        raise ValueError("shape must contain three values.")
    if len(spacing) != 3:
        raise ValueError("spacing must contain three values.")

    centre_j = 0.5 * float(shape[1])
    centre_k = 0.5 * float(shape[2])
    start_i = 0.25 * float(shape[0])
    end_i = 0.75 * float(shape[0])

    payload: dict[str, Any] = {
        "schema_version": "0.1",
        "scene": {
            "id": scene_id,
            "description": description,
        },
        "grid": {
            "shape": [int(value) for value in shape],
            "spacing": [float(value) for value in spacing],
        },
        "composition": {
            "label_mode": "priority",
            "scalar_blend": "overwrite",
            "overlap_policy": "warn",
        },
        "mask_rules": {
            "target_roles": ["target"],
            "analysis_roles": ["target", "analysis_support"],
        },
        "objects": [
            make_tube_object(
                object_id="target",
                role="target",
                label=1,
                priority=10,
                map_name=map_name,
                start_mm=[start_i, centre_j, centre_k],
                end_mm=[end_i, centre_j, centre_k],
                radius_mm=radius_mm,
                profile_kind="linear_radial",
            )
        ],
    }
    return dump_scene_dict(payload)


def duplicate_scene_text(
    text: str,
    *,
    new_scene_id: str,
    description_suffix: str = "Duplicated scene.",
) -> str:
    """Duplicate a scene YAML document with a new scene ID."""

    data = parse_scene_text(text)
    scene = data.get("scene")
    if not isinstance(scene, dict):
        scene = {}
        data["scene"] = scene

    old_description = str(scene.get("description", "")).strip()
    scene["id"] = new_scene_id
    scene["description"] = (
        f"{old_description} {description_suffix}".strip()
        if old_description
        else description_suffix
    )

    return dump_scene_dict(data)


def make_tube_object(
    *,
    object_id: str,
    role: str,
    label: int,
    priority: int,
    map_name: str,
    start_mm: list[float],
    end_mm: list[float],
    radius_mm: float = 2.0,
    profile_kind: str = "constant",
) -> dict[str, Any]:
    """Create a tube object mapping."""

    profile: dict[str, Any]
    if profile_kind == "linear_radial":
        profile = {
            "kind": "linear_radial",
            "centre_value": 1.0,
            "edge_value": 0.2,
            "background_value": 0.0,
        }
    elif profile_kind == "gaussian_radial":
        profile = {
            "kind": "gaussian_radial",
            "centre_value": 1.0,
            "edge_value": 0.2,
            "sigma_fraction": 0.5,
            "background_value": 0.0,
        }
    elif profile_kind == "edge_enhanced":
        profile = {
            "kind": "edge_enhanced",
            "centre_value": 0.2,
            "edge_value": 1.0,
            "edge_width_fraction": 0.2,
            "background_value": 0.0,
        }
    else:
        profile = {
            "kind": "constant",
            "value": 0.5,
            "background_value": 0.0,
        }

    return {
        "id": object_id,
        "kind": "tube",
        "role": role,
        "label": int(label),
        "priority": int(priority),
        "map_name": map_name,
        "curve": {
            "kind": "line",
            "start_mm": [float(value) for value in start_mm],
            "end_mm": [float(value) for value in end_mm],
            "step_mm": 1.0,
        },
        "cross_section": {
            "kind": "circle",
            "radius_mm": float(radius_mm),
        },
        "profile": profile,
    }


def make_sphere_object(
    *,
    object_id: str,
    role: str,
    label: int,
    priority: int,
    map_name: str,
    centre_mm: list[float],
    radius_mm: float = 2.5,
    value: float = 1.25,
) -> dict[str, Any]:
    """Create a sphere object mapping."""

    return {
        "id": object_id,
        "kind": "sphere",
        "role": role,
        "label": int(label),
        "priority": int(priority),
        "map_name": map_name,
        "centre_mm": [float(value) for value in centre_mm],
        "radius_mm": float(radius_mm),
        "profile": {
            "kind": "constant",
            "value": float(value),
            "background_value": 0.0,
        },
    }


def make_ellipsoid_object(
    *,
    object_id: str,
    role: str,
    label: int,
    priority: int,
    map_name: str,
    centre_mm: list[float],
    radii_mm: list[float] | None = None,
    value: float = 0.35,
) -> dict[str, Any]:
    """Create an ellipsoid object mapping."""

    radii_mm = [4.0, 2.0, 3.0] if radii_mm is None else radii_mm
    return {
        "id": object_id,
        "kind": "ellipsoid",
        "role": role,
        "label": int(label),
        "priority": int(priority),
        "map_name": map_name,
        "centre_mm": [float(value) for value in centre_mm],
        "radii_mm": [float(value) for value in radii_mm],
        "profile": {
            "kind": "constant",
            "value": float(value),
            "background_value": 0.0,
        },
    }


def add_object_to_scene_text(text: str, obj: Mapping[str, Any]) -> str:
    """Append an object to a scene YAML document and return updated YAML."""

    data = parse_scene_text(text)
    objects = data.get("objects")
    if objects is None:
        objects = []
        data["objects"] = objects
    if not isinstance(objects, list):
        raise ValueError("Scene field 'objects' must be a list.")

    object_id = object_id_for(obj)
    existing_ids = {
        object_id_for(existing) for existing in objects if isinstance(existing, dict)
    }
    if object_id in existing_ids:
        raise ValueError(f"Object id {object_id!r} already exists.")

    labels = [
        existing.get("label")
        for existing in objects
        if isinstance(existing, dict) and "label" in existing
    ]
    if obj.get("label") in labels:
        raise ValueError(f"Object label {obj.get('label')!r} already exists.")

    objects.append(deepcopy(dict(obj)))
    return dump_scene_dict(data)


def suggest_next_label(text: str) -> int:
    """Suggest the next positive integer object label."""

    data = parse_scene_text(text)
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Scene field 'objects' must be a list.")

    labels = [
        int(obj["label"]) for obj in objects if isinstance(obj, dict) and "label" in obj
    ]
    return max(labels, default=0) + 1


def suggest_object_id(text: str, prefix: str) -> str:
    """Suggest a unique object ID using a prefix."""

    data = parse_scene_text(text)
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Scene field 'objects' must be a list.")

    existing = {object_id_for(obj) for obj in objects if isinstance(obj, dict)}
    if prefix not in existing:
        return prefix

    index = 2
    while f"{prefix}_{index}" in existing:
        index += 1
    return f"{prefix}_{index}"


def scene_grid_centre(text: str) -> list[float]:
    """Return an approximate physical centre from the scene grid."""

    data = parse_scene_text(text)
    grid = data.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("Scene field 'grid' must be a mapping.")

    shape = grid.get("shape", [32, 32, 32])
    spacing = grid.get("spacing", [1.0, 1.0, 1.0])
    if len(shape) != 3 or len(spacing) != 3:
        raise ValueError("grid.shape and grid.spacing must contain three values.")

    return [0.5 * float(shape[idx]) * float(spacing[idx]) for idx in range(3)]


def object_id_for(obj: Mapping[str, Any]) -> str:
    """Return the stable ID for a scene object mapping."""

    value = obj.get("id", obj.get("object_id"))
    if value is None:
        return ""
    return str(value)


def object_summary_rows(text: str) -> list[dict[str, str]]:
    """Return display rows for objects in a scene YAML document."""

    data = parse_scene_text(text)
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Scene field 'objects' must be a list.")

    rows: list[dict[str, str]] = []
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            rows.append(
                {
                    "index": str(idx),
                    "id": "",
                    "kind": "<invalid>",
                    "role": "",
                    "label": "",
                    "priority": "",
                    "map_name": "",
                }
            )
            continue

        rows.append(
            {
                "index": str(idx),
                "id": object_id_for(obj),
                "kind": str(obj.get("kind", "")),
                "role": str(obj.get("role", "")),
                "label": str(obj.get("label", "")),
                "priority": str(obj.get("priority", "")),
                "map_name": str(obj.get("map_name", "")),
            }
        )

    return rows


def object_ids(text: str) -> list[str]:
    """Return object IDs in scene order."""

    return [row["id"] for row in object_summary_rows(text) if row["id"]]


def get_object(text: str, object_id: str) -> dict[str, Any]:
    """Return a copy of one object mapping by ID."""

    data = parse_scene_text(text)
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Scene field 'objects' must be a list.")

    for obj in objects:
        if isinstance(obj, dict) and object_id_for(obj) == object_id:
            return deepcopy(obj)

    raise KeyError(f"Object {object_id!r} was not found.")


def replace_object(text: str, object_id: str, new_object: Mapping[str, Any]) -> str:
    """Replace one object mapping and return updated YAML text."""

    data = parse_scene_text(text)
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Scene field 'objects' must be a list.")

    replaced = False
    new_objects: list[Any] = []
    for obj in objects:
        if isinstance(obj, dict) and object_id_for(obj) == object_id:
            new_objects.append(deepcopy(dict(new_object)))
            replaced = True
        else:
            new_objects.append(obj)

    if not replaced:
        raise KeyError(f"Object {object_id!r} was not found.")

    data["objects"] = new_objects
    return dump_scene_dict(data)


def flatten_editable_fields(
    obj: Mapping[str, Any],
    *,
    include_id_and_kind: bool = False,
) -> dict[str, EditableValue]:
    """Flatten common object fields into dot-paths for form editing."""

    fields: dict[str, EditableValue] = {}

    common_keys = [
        "id",
        "object_id",
        "kind",
        "role",
        "label",
        "priority",
        "map_name",
        "name",
        "description",
    ]

    for key in common_keys:
        if key not in obj:
            continue
        if not include_id_and_kind and key in {"id", "object_id", "kind"}:
            continue
        fields[key] = _coerce_editable_value(obj[key])

    for section in ("curve", "cross_section", "profile"):
        value = obj.get(section)
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                fields[f"{section}.{nested_key}"] = _coerce_editable_value(nested_value)

    geometry_keys = [
        "centre_mm",
        "radius_mm",
        "radii_mm",
        "normal",
        "thickness_mm",
        "apex_mm",
        "axis",
        "axis_direction",
        "height_mm",
        "base_radius_mm",
        "start_mm",
        "radius_start_mm",
        "radius_end_mm",
    ]
    for key in geometry_keys:
        if key in obj:
            fields[key] = _coerce_editable_value(obj[key])

    return fields


def apply_field_edits(
    obj: Mapping[str, Any],
    edits: Mapping[str, str],
) -> dict[str, Any]:
    """Apply dot-path string edits to an object mapping."""

    updated = deepcopy(dict(obj))

    for path, raw_value in edits.items():
        value = parse_edit_value(raw_value)
        parts = path.split(".")
        if not parts:
            continue

        cursor: dict[str, Any] = updated
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child

        cursor[parts[-1]] = value

    return updated


def parse_edit_value(text: str) -> EditableValue:
    """Parse a GUI text field into a simple YAML-compatible value."""

    stripped = text.strip()
    if stripped == "":
        return ""

    try:
        parsed = yaml.safe_load(stripped)
    except yaml.YAMLError:
        return text

    if isinstance(parsed, (str, int, float, bool)) or parsed is None:
        return parsed
    if isinstance(parsed, list):
        return parsed

    return text


def format_edit_value(value: Any) -> str:
    """Format a YAML-compatible value for a GUI text input."""

    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if value is None:
        return ""
    return str(value)


def _coerce_editable_value(value: Any) -> EditableValue:
    """Keep only values that can be edited safely in a simple text field."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return value
    return str(value)


def delete_object_from_scene_text(
    text: str,
    object_id: str,
    *,
    require_remaining_object: bool = True,
) -> str:
    """Delete one object from a scene YAML document."""

    data = parse_scene_text(text)
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Scene field 'objects' must be a list.")

    remaining: list[Any] = []
    deleted = False
    for obj in objects:
        if isinstance(obj, dict) and object_id_for(obj) == object_id:
            deleted = True
            continue
        remaining.append(obj)

    if not deleted:
        raise KeyError(f"Object {object_id!r} was not found.")

    if require_remaining_object and not remaining:
        raise ValueError(
            "Cannot delete the final object from the scene. Create or duplicate "
            "another object first, or disable require_remaining_object."
        )

    data["objects"] = remaining
    return dump_scene_dict(data)


def duplicate_object_in_scene_text(
    text: str,
    object_id: str,
    *,
    new_object_id: str | None = None,
    new_label: int | None = None,
    offset_mm: list[float] | None = None,
) -> str:
    """Duplicate one scene object with a new ID and label."""

    source = get_object(text, object_id)

    if new_object_id is None:
        new_object_id = suggest_object_id(text, f"{object_id}_copy")
    if new_label is None:
        new_label = suggest_next_label(text)
    if offset_mm is None:
        offset_mm = [0.0, 3.0, 0.0]

    if len(offset_mm) != 3:
        raise ValueError("offset_mm must contain three values.")

    duplicate = deepcopy(source)
    duplicate["id"] = new_object_id
    duplicate.pop("object_id", None)
    duplicate["label"] = int(new_label)

    _translate_object_geometry(duplicate, [float(value) for value in offset_mm])

    return add_object_to_scene_text(text, duplicate)


def _translate_object_geometry(obj: dict[str, Any], offset_mm: list[float]) -> None:
    """Translate common object geometry fields in-place."""

    for key in ("centre_mm", "apex_mm", "start_mm"):
        if key in obj:
            obj[key] = _translated_point(obj[key], offset_mm)

    curve = obj.get("curve")
    if isinstance(curve, dict):
        for key in ("start_mm", "end_mm"):
            if key in curve:
                curve[key] = _translated_point(curve[key], offset_mm)


def _translated_point(value: Any, offset_mm: list[float]) -> list[float]:
    """Return a translated three-coordinate point."""

    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"Expected three coordinate values, got: {value!r}")

    return [float(value[idx]) + float(offset_mm[idx]) for idx in range(3)]
