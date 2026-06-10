"""Scalar-profile editing helpers for the optional GUI workbench."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from synthworkshop.gui.yaml_editor import get_object, replace_object

ProfileControlKind = Literal["float", "fraction", "asymmetry"]


@dataclass(frozen=True)
class ProfileControl:
    """One editable scalar-profile control."""

    key: str
    label: str
    kind: ProfileControlKind
    value: float
    min_value: float
    max_value: float
    step: float
    help: str = ""


PROFILE_KINDS: tuple[str, ...] = (
    "constant",
    "linear_radial",
    "gaussian_radial",
    "edge_enhanced",
    "asymmetric_linear",
)


def profile_template(
    kind: str,
    *,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a profile template, preserving compatible existing values."""

    existing = {} if existing is None else dict(existing)

    if kind == "constant":
        template: dict[str, Any] = {
            "kind": "constant",
            "value": 0.5,
            "background_value": 0.0,
        }
    elif kind == "gaussian_radial":
        template = {
            "kind": "gaussian_radial",
            "centre_value": 1.0,
            "edge_value": 0.2,
            "sigma_fraction": 0.5,
            "background_value": 0.0,
        }
    elif kind == "edge_enhanced":
        template = {
            "kind": "edge_enhanced",
            "centre_value": 0.2,
            "edge_value": 1.0,
            "edge_width_fraction": 0.2,
            "background_value": 0.0,
        }
    elif kind == "asymmetric_linear":
        template = {
            "kind": "asymmetric_linear",
            "centre_value": 1.0,
            "edge_value": 0.2,
            "asymmetry": 0.2,
            "background_value": 0.0,
        }
    else:
        template = {
            "kind": "linear_radial",
            "centre_value": 1.0,
            "edge_value": 0.2,
            "background_value": 0.0,
        }

    for key in tuple(template):
        if key == "kind":
            continue
        if key in existing and isinstance(existing[key], int | float):
            template[key] = float(existing[key])

    return template


def profile_for_object(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Return an object's scalar profile mapping."""

    profile = obj.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("Selected object has no editable 'profile' mapping.")
    return deepcopy(profile)


def profile_controls_for_profile(profile: Mapping[str, Any]) -> list[ProfileControl]:
    """Return editable controls for a scalar-profile mapping."""

    controls: list[ProfileControl] = []

    for key, label in (
        ("value", "Constant value"),
        ("centre_value", "Centre value"),
        ("edge_value", "Edge value"),
        ("background_value", "Background value"),
    ):
        if key in profile:
            controls.append(
                ProfileControl(
                    key=key,
                    label=label,
                    kind="float",
                    value=float(profile[key]),
                    min_value=-5.0,
                    max_value=5.0,
                    step=0.05,
                )
            )

    if "asymmetry" in profile:
        controls.append(
            ProfileControl(
                key="asymmetry",
                label="Asymmetry",
                kind="asymmetry",
                value=float(profile["asymmetry"]),
                min_value=-2.0,
                max_value=2.0,
                step=0.05,
                help=(
                    "Signed left/right scalar offset. Positive and negative "
                    "values bias opposite sides of the profile."
                ),
            )
        )

    for key, label in (
        ("sigma_fraction", "Gaussian sigma fraction"),
        ("gaussian_sigma_fraction", "Gaussian sigma fraction"),
        ("edge_width_fraction", "Edge width fraction"),
    ):
        if key in profile:
            controls.append(
                ProfileControl(
                    key=key,
                    label=label,
                    kind="fraction",
                    value=float(profile[key]),
                    min_value=0.01,
                    max_value=2.0 if "sigma" in key else 1.0,
                    step=0.01,
                )
            )

    return controls


def apply_profile_updates(
    profile: Mapping[str, Any],
    updates: Mapping[str, float],
) -> dict[str, Any]:
    """Apply numeric profile updates."""

    updated = deepcopy(dict(profile))
    for key, value in updates.items():
        updated[key] = float(value)
    return updated


def replace_object_profile(
    text: str,
    object_id: str,
    profile: Mapping[str, Any],
) -> str:
    """Replace one object's scalar profile and return updated YAML text."""

    obj = get_object(text, object_id)
    obj["profile"] = deepcopy(dict(profile))
    return replace_object(text, object_id, obj)


def update_object_profile(
    text: str,
    object_id: str,
    *,
    kind: str,
    updates: Mapping[str, float],
) -> str:
    """Update one object's scalar profile and return updated YAML text."""

    obj = get_object(text, object_id)
    current_profile = profile_for_object(obj)
    new_profile = profile_template(kind, existing=current_profile)
    new_profile = apply_profile_updates(new_profile, updates)
    obj["profile"] = new_profile
    return replace_object(text, object_id, obj)
