"""Scene-level YAML editing helpers for the optional GUI workbench."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "Scene settings editing requires PyYAML. Install the package with the "
        "YAML/IO dependencies enabled."
    ) from exc

from synthworkshop.gui.yaml_editor import dump_scene_dict, parse_scene_text

LABEL_MODES: tuple[str, ...] = ("priority", "first", "last")
SCALAR_BLEND_MODES: tuple[str, ...] = ("overwrite", "max", "sum", "weighted_mean")
OVERLAP_POLICIES: tuple[str, ...] = ("allow", "warn", "error")


def scene_settings_from_text(text: str) -> dict[str, Any]:
    """Return editable scene-level settings from YAML text."""

    data = parse_scene_text(text)

    scene = data.get("scene", {})
    if not isinstance(scene, dict):
        scene = {}

    grid = data.get("grid", {})
    if not isinstance(grid, dict):
        grid = {}

    composition = data.get("composition", {})
    if not isinstance(composition, dict):
        composition = {}

    mask_rules = data.get("mask_rules", {})
    if not isinstance(mask_rules, dict):
        mask_rules = {}

    return {
        "schema_version": str(data.get("schema_version", "0.1")),
        "scene_id": str(scene.get("id", "")),
        "description": str(scene.get("description", "")),
        "shape": _coerce_shape(grid.get("shape", [32, 32, 32])),
        "spacing": _coerce_spacing(grid.get("spacing", [1.0, 1.0, 1.0])),
        "origin": _coerce_float_list(grid.get("origin", []), allow_empty=True),
        "axis_names": _coerce_string_list(grid.get("axis_names", []), allow_empty=True),
        "label_mode": str(composition.get("label_mode", "priority")),
        "scalar_blend": str(composition.get("scalar_blend", "overwrite")),
        "overlap_policy": str(composition.get("overlap_policy", "warn")),
        "target_roles": _coerce_string_list(
            mask_rules.get("target_roles", ["target"]),
            allow_empty=False,
        ),
        "analysis_roles": _coerce_string_list(
            mask_rules.get("analysis_roles", ["target", "analysis_support"]),
            allow_empty=False,
        ),
    }


def update_scene_settings(
    text: str,
    *,
    scene_id: str,
    description: str,
    shape: Sequence[int] | str,
    spacing: Sequence[float] | str,
    label_mode: str,
    scalar_blend: str,
    overlap_policy: str,
    target_roles: Sequence[str] | str,
    analysis_roles: Sequence[str] | str,
    schema_version: str = "0.1",
    origin: Sequence[float] | str | None = None,
    axis_names: Sequence[str] | str | None = None,
) -> str:
    """Update scene-level settings and return YAML text."""

    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unknown label_mode: {label_mode!r}.")
    if scalar_blend not in SCALAR_BLEND_MODES:
        raise ValueError(f"Unknown scalar_blend: {scalar_blend!r}.")
    if overlap_policy not in OVERLAP_POLICIES:
        raise ValueError(f"Unknown overlap_policy: {overlap_policy!r}.")

    data = parse_scene_text(text)

    data["schema_version"] = str(schema_version)

    scene = data.get("scene")
    if not isinstance(scene, dict):
        scene = {}
        data["scene"] = scene
    scene["id"] = str(scene_id)
    scene["description"] = str(description)

    grid = data.get("grid")
    if not isinstance(grid, dict):
        grid = {}
        data["grid"] = grid
    grid["shape"] = _coerce_shape(shape)
    grid["spacing"] = _coerce_spacing(spacing)

    parsed_origin = _coerce_float_list(origin, allow_empty=True)
    if parsed_origin:
        grid["origin"] = parsed_origin
    else:
        grid.pop("origin", None)

    parsed_axis_names = _coerce_string_list(axis_names, allow_empty=True)
    if parsed_axis_names:
        grid["axis_names"] = parsed_axis_names
    else:
        grid.pop("axis_names", None)

    composition = data.get("composition")
    if not isinstance(composition, dict):
        composition = {}
        data["composition"] = composition
    composition["label_mode"] = label_mode
    composition["scalar_blend"] = scalar_blend
    composition["overlap_policy"] = overlap_policy

    mask_rules = data.get("mask_rules")
    if not isinstance(mask_rules, dict):
        mask_rules = {}
        data["mask_rules"] = mask_rules
    mask_rules["target_roles"] = _coerce_string_list(target_roles, allow_empty=False)
    mask_rules["analysis_roles"] = _coerce_string_list(
        analysis_roles, allow_empty=False
    )

    return dump_scene_dict(data)


def format_numeric_list(values: Sequence[int | float]) -> str:
    """Format a numeric sequence for a GUI text field."""

    return "[" + ", ".join(str(value) for value in values) + "]"


def format_string_list(values: Sequence[str]) -> str:
    """Format a string sequence for a GUI text field."""

    return "[" + ", ".join(str(value) for value in values) + "]"


def _parse_maybe_yaml(value: Sequence[Any] | str | None) -> Any:
    """Parse strings as YAML, otherwise return the original value."""

    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return []
        parsed = yaml.safe_load(stripped)
        if isinstance(parsed, str):
            return [part.strip() for part in parsed.split(",") if part.strip()]
        return parsed
    return value


def _coerce_shape(value: Sequence[int] | str | None) -> list[int]:
    """Coerce a shape sequence to positive integers."""

    parsed = _parse_maybe_yaml(value)
    if not isinstance(parsed, list | tuple):
        raise ValueError("shape must be a list of positive integers.")
    if len(parsed) not in {2, 3}:
        raise ValueError("shape must contain two or three values.")

    shape = [int(item) for item in parsed]
    if any(item <= 0 for item in shape):
        raise ValueError("shape values must be positive.")
    return shape


def _coerce_spacing(value: Sequence[float] | str | None) -> list[float]:
    """Coerce a spacing sequence to positive floats."""

    parsed = _parse_maybe_yaml(value)
    if not isinstance(parsed, list | tuple):
        raise ValueError("spacing must be a list of positive numbers.")
    if len(parsed) not in {2, 3}:
        raise ValueError("spacing must contain two or three values.")

    spacing = [float(item) for item in parsed]
    if any(item <= 0 for item in spacing):
        raise ValueError("spacing values must be positive.")
    return spacing


def _coerce_float_list(
    value: Sequence[float] | str | None,
    *,
    allow_empty: bool,
) -> list[float]:
    """Coerce a value to a list of floats."""

    parsed = _parse_maybe_yaml(value)
    if parsed in (None, ""):
        parsed = []
    if not isinstance(parsed, list | tuple):
        raise ValueError("Expected a list of numbers.")

    out = [float(item) for item in parsed]
    if not out and not allow_empty:
        raise ValueError("Expected at least one value.")
    return out


def _coerce_string_list(
    value: Sequence[str] | str | None,
    *,
    allow_empty: bool,
) -> list[str]:
    """Coerce a value to a list of strings."""

    parsed = _parse_maybe_yaml(value)
    if parsed in (None, ""):
        parsed = []
    if not isinstance(parsed, list | tuple):
        raise ValueError("Expected a list of strings.")

    out = [str(item).strip() for item in parsed if str(item).strip()]
    if not out and not allow_empty:
        raise ValueError("Expected at least one role.")
    return out
