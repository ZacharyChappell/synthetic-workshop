"""Geometry-control helpers for the optional GUI workbench.

The helpers in this module keep the GUI schema-first: slider values are written
back into YAML-compatible object fields rather than into a hidden scene model.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from synthworkshop.gui.yaml_editor import (
    get_object,
    parse_scene_text,
    replace_object,
)

ControlKind = Literal["point3", "positive_float", "positive_vector3", "vector3"]


@dataclass(frozen=True)
class GeometryControl:
    """One editable geometry control for a scene object."""

    path: str
    label: str
    kind: ControlKind
    value: float | list[float]
    max_value: float | list[float] | None = None
    help: str = ""


def grid_extent_mm(text: str) -> list[float]:
    """Return the physical grid extent along each axis."""

    data = parse_scene_text(text)
    grid = data.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("Scene field 'grid' must be a mapping.")

    shape = grid.get("shape", [32, 32, 32])
    spacing = grid.get("spacing", [1.0, 1.0, 1.0])
    if len(shape) != 3 or len(spacing) != 3:
        raise ValueError("grid.shape and grid.spacing must contain three values.")

    return [float(shape[idx]) * float(spacing[idx]) for idx in range(3)]


def geometry_controls_for_object(
    obj: Mapping[str, Any],
    *,
    extent_mm: list[float],
) -> list[GeometryControl]:
    """Return GUI-editable geometry controls for an object mapping."""

    kind = str(obj.get("kind", ""))
    max_radius = max(0.5, min(extent_mm) / 2.0)

    controls: list[GeometryControl] = []

    if kind == "tube":
        curve = obj.get("curve", {})
        cross_section = obj.get("cross_section", {})

        if isinstance(curve, dict):
            if "start_mm" in curve:
                controls.append(
                    GeometryControl(
                        path="curve.start_mm",
                        label="Tube start",
                        kind="point3",
                        value=_point3(curve["start_mm"]),
                        max_value=extent_mm,
                        help="Physical start coordinate of the tube centreline.",
                    )
                )
            if "end_mm" in curve:
                controls.append(
                    GeometryControl(
                        path="curve.end_mm",
                        label="Tube end",
                        kind="point3",
                        value=_point3(curve["end_mm"]),
                        max_value=extent_mm,
                        help="Physical end coordinate of the tube centreline.",
                    )
                )
            if "amplitude_mm" in curve:
                controls.append(
                    GeometryControl(
                        path="curve.amplitude_mm",
                        label="Sinusoidal amplitude",
                        kind="vector3",
                        value=_point3(curve["amplitude_mm"]),
                        max_value=[max_radius, max_radius, max_radius],
                        help="Axis-wise sinusoidal displacement amplitude.",
                    )
                )
            if "periods" in curve:
                controls.append(
                    GeometryControl(
                        path="curve.periods",
                        label="Sinusoidal periods",
                        kind="positive_float",
                        value=float(curve["periods"]),
                        max_value=8.0,
                        help="Number of sinusoidal periods along the curve.",
                    )
                )

        if isinstance(cross_section, dict):
            if "radius_mm" in cross_section:
                controls.append(
                    GeometryControl(
                        path="cross_section.radius_mm",
                        label="Tube radius",
                        kind="positive_float",
                        value=float(cross_section["radius_mm"]),
                        max_value=max_radius,
                        help="Circular tube radius.",
                    )
                )
            for key, label in (
                ("semi_axis_u_mm", "Ellipse semi-axis u"),
                ("semi_axis_v_mm", "Ellipse semi-axis v"),
                ("width_mm", "Ribbon width"),
                ("thickness_mm", "Ribbon thickness"),
                ("base_radius_mm", "Base radius"),
            ):
                if key in cross_section:
                    controls.append(
                        GeometryControl(
                            path=f"cross_section.{key}",
                            label=label,
                            kind="positive_float",
                            value=float(cross_section[key]),
                            max_value=max_radius,
                        )
                    )

        return controls

    if "centre_mm" in obj:
        controls.append(
            GeometryControl(
                path="centre_mm",
                label="Object centre",
                kind="point3",
                value=_point3(obj["centre_mm"]),
                max_value=extent_mm,
                help="Physical centre coordinate.",
            )
        )

    if kind == "sphere" and "radius_mm" in obj:
        controls.append(
            GeometryControl(
                path="radius_mm",
                label="Sphere radius",
                kind="positive_float",
                value=float(obj["radius_mm"]),
                max_value=max_radius,
            )
        )

    if kind == "ellipsoid" and "radii_mm" in obj:
        controls.append(
            GeometryControl(
                path="radii_mm",
                label="Ellipsoid radii",
                kind="positive_vector3",
                value=_point3(obj["radii_mm"]),
                max_value=[max_radius, max_radius, max_radius],
            )
        )

    if kind == "slab":
        if "normal" in obj:
            controls.append(
                GeometryControl(
                    path="normal",
                    label="Slab normal",
                    kind="vector3",
                    value=_point3(obj["normal"]),
                    max_value=[1.0, 1.0, 1.0],
                    help="Normal vector. It does not need to be unit length.",
                )
            )
        if "thickness_mm" in obj:
            controls.append(
                GeometryControl(
                    path="thickness_mm",
                    label="Slab thickness",
                    kind="positive_float",
                    value=float(obj["thickness_mm"]),
                    max_value=max_radius,
                )
            )

    if kind == "cone":
        if "apex_mm" in obj:
            controls.append(
                GeometryControl(
                    path="apex_mm",
                    label="Cone apex",
                    kind="point3",
                    value=_point3(obj["apex_mm"]),
                    max_value=extent_mm,
                )
            )
        for key, label in (
            ("height_mm", "Cone height"),
            ("base_radius_mm", "Cone base radius"),
        ):
            if key in obj:
                controls.append(
                    GeometryControl(
                        path=key,
                        label=label,
                        kind="positive_float",
                        value=float(obj[key]),
                        max_value=max(extent_mm),
                    )
                )

    if kind == "frustum":
        if "start_mm" in obj:
            controls.append(
                GeometryControl(
                    path="start_mm",
                    label="Frustum start",
                    kind="point3",
                    value=_point3(obj["start_mm"]),
                    max_value=extent_mm,
                )
            )
        for key, label in (
            ("height_mm", "Frustum height"),
            ("radius_start_mm", "Start radius"),
            ("radius_end_mm", "End radius"),
        ):
            if key in obj:
                controls.append(
                    GeometryControl(
                        path=key,
                        label=label,
                        kind="positive_float",
                        value=float(obj[key]),
                        max_value=max(extent_mm),
                    )
                )

    return controls


def update_object_geometry(
    text: str,
    object_id: str,
    updates: Mapping[str, Any],
) -> str:
    """Apply geometry-control updates to one object and return YAML text."""

    obj = get_object(text, object_id)
    updated = deepcopy(obj)

    for path, value in updates.items():
        _set_path(updated, path, value)

    return replace_object(text, object_id, updated)


def _set_path(obj: dict[str, Any], path: str, value: Any) -> None:
    """Set a dot-path in a nested dictionary."""

    parts = path.split(".")
    cursor = obj
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _point3(value: Any) -> list[float]:
    """Coerce a value to a three-element float list."""

    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"Expected three coordinate values, got: {value!r}")
    return [float(item) for item in value]
