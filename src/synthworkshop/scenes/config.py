"""YAML/JSON scene specification loading and rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthworkshop.cross_sections import (
    CircularCrossSection,
    EllipticCrossSection,
    RibbonCrossSection,
    RotatingEllipticCrossSection,
    SuperellipseCrossSection,
    VariableCircularCrossSection,
    VariableEllipticCrossSection,
)
from synthworkshop.effects import apply_effects
from synthworkshop.grid import GridSpec
from synthworkshop.perturbations import apply_perturbations
from synthworkshop.primitives.curves import LineCurve, SinusoidalCurve
from synthworkshop.primitives.implicit import (
    ConeObject,
    EllipsoidObject,
    FrustumObject,
    SlabObject,
    SphereObject,
)
from synthworkshop.primitives.tubes import TubeObject
from synthworkshop.profiles import (
    AsymmetricLinearProfile,
    ConstantProfile,
    EdgeEnhancedProfile,
    GaussianRadialProfile,
    HollowCoreProfile,
    LinearRadialProfile,
    LongitudinalGradientProfile,
    MultiPeakRadialProfile,
    OneSidedLesionProfile,
    PeriodicLongitudinalProfile,
    RadialLongitudinalGradientProfile,
    SigmoidBoundaryProfile,
)
from synthworkshop.scenes.model import CompositionRules, MaskRules, RenderedScene
from synthworkshop.scenes.render import render_objects


@dataclass(frozen=True)
class SceneObjectSpec:
    """Declarative specification for one scene object."""

    object_id: str
    kind: str
    role: str
    label: int
    priority: int = 0
    map_name: str = "scalar"
    curve: Mapping[str, Any] = field(default_factory=dict)
    cross_section: Mapping[str, Any] = field(default_factory=dict)
    profile: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = str(self.object_id)
        kind = str(self.kind)
        role = str(self.role)
        map_name = str(self.map_name)
        label = int(self.label)
        priority = int(self.priority)

        if not object_id:
            raise ValueError("object_id must be a non-empty string.")
        if not kind:
            raise ValueError("object kind must be a non-empty string.")
        if not role:
            raise ValueError("role must be a non-empty string.")
        if not map_name:
            raise ValueError("map_name must be a non-empty string.")
        if label <= 0:
            raise ValueError("label must be a positive integer.")

        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "map_name", map_name)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "curve", dict(self.curve))
        object.__setattr__(self, "cross_section", dict(self.cross_section))
        object.__setattr__(self, "profile", dict(self.profile))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SceneSpec:
    """Declarative specification for a rendered synthetic scene."""

    scene_id: str
    grid: GridSpec
    objects: tuple[SceneObjectSpec, ...]
    description: str | None = None
    schema_version: str = "0.1"
    composition: CompositionRules = field(default_factory=CompositionRules)
    mask_rules: MaskRules = field(default_factory=MaskRules)
    render: Mapping[str, Any] = field(default_factory=dict)
    perturbations: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    effects: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id)
        if not scene_id:
            raise ValueError("scene_id must be a non-empty string.")
        if not self.objects:
            raise ValueError("SceneSpec requires at least one object.")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "render", dict(self.render))
        object.__setattr__(
            self,
            "perturbations",
            tuple(_perturbations_from_config(self.perturbations)),
        )
        object.__setattr__(
            self,
            "effects",
            tuple(_effects_from_config(self.effects)),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


def _kind(value: Any) -> str:
    """Normalise a registry kind string."""

    return str(value).strip().lower().replace("-", "_")


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Validate a mapping value."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping/object.")
    return value


def _optional_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return an optional mapping from a payload."""

    value = payload.get(key, {})
    if value is None:
        return {}
    return _as_mapping(value, name=key)


def _perturbations_from_config(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Validate perturbation specifications from config."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("perturbations must be a sequence of mappings.")

    perturbations: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"perturbations[{index}] must be a mapping/object.")
        perturbations.append(dict(item))
    return tuple(perturbations)


def _effects_from_config(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Validate known-effect specifications from config."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("effects must be a sequence of mappings.")

    effects: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"effects[{index}] must be a mapping/object.")
        effects.append(dict(item))
    return tuple(effects)


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return a required mapping from a payload."""

    if key not in payload:
        raise ValueError(f"Missing required section: {key!r}.")
    return _as_mapping(payload[key], name=key)


def _required_sequence(
    payload: Mapping[str, Any],
    *keys: str,
    name: str,
) -> Sequence[float]:
    """Return the first present sequence value from a mapping."""

    for key in keys:
        if key in payload:
            value = payload[key]
            if not isinstance(value, Sequence) or isinstance(value, str):
                raise ValueError(f"{key} must be a numeric sequence.")
            return value
    joined = " or ".join(repr(key) for key in keys)
    raise ValueError(f"Missing required {name}: {joined}.")


def _required_value(
    payload: Mapping[str, Any],
    *keys: str,
    name: str,
) -> Any:
    """Return the first present scalar or sequence value from a mapping."""

    for key in keys:
        if key in payload:
            return payload[key]
    joined = " or ".join(repr(key) for key in keys)
    raise ValueError(f"Missing required {name}: {joined}.")


def _grid_from_config(config: Mapping[str, Any]) -> GridSpec:
    """Build a GridSpec from a config mapping."""

    return GridSpec(
        shape=config["shape"],
        spacing=config["spacing"],
        origin=config.get("origin"),
        axis_names=config.get("axis_names"),
    )


def _composition_from_config(config: Mapping[str, Any]) -> CompositionRules:
    """Build CompositionRules from a config mapping."""

    return CompositionRules(
        label_mode=config.get("label_mode", "priority"),
        scalar_blend=config.get("scalar_blend", "overwrite"),
        overlap_policy=config.get("overlap_policy", "warn"),
    )


def _mask_rules_from_config(config: Mapping[str, Any]) -> MaskRules:
    """Build MaskRules from a config mapping."""

    if not config:
        return MaskRules()
    return MaskRules(
        target_roles=config.get("target_roles", ("target",)),
        analysis_roles=config.get(
            "analysis_roles",
            ("target", "analysis_support"),
        ),
    )


def _curve_from_config(config: Mapping[str, Any]):
    """Build an analytic curve from a config mapping."""

    kind = _kind(config.get("kind", ""))
    if kind == "line":
        return LineCurve(
            start_mm=_required_sequence(config, "start_mm", "start", name="start"),
            end_mm=_required_sequence(config, "end_mm", "end", name="end"),
        )

    if kind in {"sinusoid", "sinusoidal", "wavy"}:
        return SinusoidalCurve(
            start_mm=_required_sequence(config, "start_mm", "start", name="start"),
            end_mm=_required_sequence(config, "end_mm", "end", name="end"),
            amplitude_mm=_required_sequence(
                config,
                "amplitude_mm",
                "amplitude",
                name="amplitude",
            ),
            periods=float(config.get("periods", 1.0)),
            phase_radians=float(config.get("phase_radians", 0.0)),
        )

    raise ValueError(f"Unknown curve kind: {kind!r}.")


def _sample_curve(config: Mapping[str, Any], *, object_id: str):
    """Build and sample a curve from config."""

    curve = _curve_from_config(config)
    n_samples = config.get("n_samples")
    return curve.sample(
        step_mm=float(config.get("step_mm", 1.0)),
        n_samples=None if n_samples is None else int(n_samples),
        object_id=object_id,
        segment_id=config.get("segment_id"),
    )


def _cross_section_from_config(
    config: Mapping[str, Any],
    *,
    length_mm: float | None = None,
):
    """Build a cross-section from config."""

    kind = _kind(config.get("kind", ""))
    if kind in {"circle", "circular"}:
        return CircularCrossSection(radius_mm=float(config["radius_mm"]))

    if kind in {"ellipse", "elliptic"}:
        return EllipticCrossSection(
            semi_axis_u_mm=float(config["semi_axis_u_mm"]),
            semi_axis_v_mm=float(config["semi_axis_v_mm"]),
        )

    if kind in {"superellipse", "super_ellipse", "squircle"}:
        return SuperellipseCrossSection(
            semi_axis_u_mm=float(config["semi_axis_u_mm"]),
            semi_axis_v_mm=float(config["semi_axis_v_mm"]),
            exponent=float(config.get("exponent", 4.0)),
        )

    if kind in {"ribbon", "flattened_ribbon", "ribbon_cross_section"}:
        if "width_mm" in config:
            width_mm = config["width_mm"]
        elif "ribbon_width_mm" in config:
            width_mm = config["ribbon_width_mm"]
        else:
            raise ValueError("Ribbon cross-sections require width_mm.")

        if "thickness_mm" in config:
            thickness_mm = config["thickness_mm"]
        elif "ribbon_thickness_mm" in config:
            thickness_mm = config["ribbon_thickness_mm"]
        else:
            raise ValueError("Ribbon cross-sections require thickness_mm.")

        return RibbonCrossSection(
            width_mm=float(width_mm),
            thickness_mm=float(thickness_mm),
            exponent=float(config.get("exponent", 6.0)),
        )

    if kind in {
        "variable_circle",
        "variable_circular",
        "variable_circle_linear",
        "tapered_circle",
        "linear_tapered_circle",
    }:
        inferred_length = config.get("length_mm", length_mm)
        if inferred_length is None:
            raise ValueError(
                "Variable circular cross-sections require length_mm or a sampled "
                "centreline from which length can be inferred."
            )

        if "radius_start_mm" in config:
            radius_start = config["radius_start_mm"]
        elif "start_radius_mm" in config:
            radius_start = config["start_radius_mm"]
        else:
            raise ValueError(
                "Variable circular cross-sections require radius_start_mm."
            )

        if "radius_end_mm" in config:
            radius_end = config["radius_end_mm"]
        elif "end_radius_mm" in config:
            radius_end = config["end_radius_mm"]
        else:
            raise ValueError("Variable circular cross-sections require radius_end_mm.")

        return VariableCircularCrossSection(
            radius_start_mm=float(radius_start),
            radius_end_mm=float(radius_end),
            length_mm=float(inferred_length),
        )

    if kind in {
        "variable_ellipse",
        "variable_elliptic",
        "variable_ellipse_linear",
        "tapered_ellipse",
        "linear_tapered_ellipse",
    }:
        inferred_length = config.get("length_mm", length_mm)
        if inferred_length is None:
            raise ValueError(
                "Variable elliptic cross-sections require length_mm or a sampled "
                "centreline from which length can be inferred."
            )

        return VariableEllipticCrossSection(
            semi_axis_u_start_mm=float(config["semi_axis_u_start_mm"]),
            semi_axis_u_end_mm=float(config["semi_axis_u_end_mm"]),
            semi_axis_v_start_mm=float(config["semi_axis_v_start_mm"]),
            semi_axis_v_end_mm=float(config["semi_axis_v_end_mm"]),
            length_mm=float(inferred_length),
        )

    if kind in {
        "rotating_ellipse",
        "rotating_elliptic",
        "rotating_ellipse_linear",
    }:
        inferred_length = config.get("length_mm", length_mm)
        if inferred_length is None:
            raise ValueError(
                "Rotating elliptic cross-sections require length_mm or a sampled "
                "centreline from which length can be inferred."
            )

        return RotatingEllipticCrossSection(
            semi_axis_u_mm=float(config["semi_axis_u_mm"]),
            semi_axis_v_mm=float(config["semi_axis_v_mm"]),
            length_mm=float(inferred_length),
            angle_start_radians=float(config.get("angle_start_radians", 0.0)),
            angle_end_radians=float(
                config.get("angle_end_radians", 1.5707963267948966)
            ),
        )

    raise ValueError(f"Unknown cross_section kind: {kind!r}.")


def _profile_from_config(
    config: Mapping[str, Any],
    *,
    length_mm: float | None = None,
):
    """Build a scalar profile from config."""

    kind = _kind(config.get("kind", ""))
    common = {
        "background_value": float(config.get("background_value", 0.0)),
    }

    if kind == "constant":
        return ConstantProfile(
            value=float(config.get("value", 1.0)),
            **common,
        )

    if kind in {"linear_radial", "linear_radial_decay"}:
        return LinearRadialProfile(
            centre_value=float(config.get("centre_value", 1.0)),
            edge_value=float(config.get("edge_value", 0.2)),
            **common,
        )

    if kind == "gaussian_radial":
        return GaussianRadialProfile(
            centre_value=float(config.get("centre_value", 1.0)),
            edge_value=float(config.get("edge_value", 0.2)),
            sigma_fraction=float(config.get("sigma_fraction", 0.45)),
            **common,
        )

    if kind == "edge_enhanced":
        return EdgeEnhancedProfile(
            centre_value=float(config.get("centre_value", 0.2)),
            edge_value=float(config.get("edge_value", 1.0)),
            edge_width_fraction=float(config.get("edge_width_fraction", 0.15)),
            **common,
        )

    if kind == "asymmetric_linear":
        return AsymmetricLinearProfile(
            centre_value=float(config.get("centre_value", 1.0)),
            edge_value=float(config.get("edge_value", 0.2)),
            asymmetry=float(config.get("asymmetry", 0.1)),
            **common,
        )

    if kind in {"hollow_core", "hollow_core_radial"}:
        return HollowCoreProfile(
            core_value=float(config.get("core_value", 0.1)),
            shell_value=float(config.get("shell_value", 1.0)),
            edge_value=float(config.get("edge_value", 0.2)),
            core_radius_fraction=float(config.get("core_radius_fraction", 0.25)),
            shell_radius_fraction=float(config.get("shell_radius_fraction", 0.65)),
            **common,
        )

    if kind in {"sigmoid_boundary", "soft_boundary"}:
        return SigmoidBoundaryProfile(
            centre_value=float(config.get("centre_value", 1.0)),
            edge_value=float(config.get("edge_value", 0.2)),
            boundary_fraction=float(config.get("boundary_fraction", 0.75)),
            width_fraction=float(config.get("width_fraction", 0.08)),
            **common,
        )

    if kind in {"longitudinal_gradient", "linear_longitudinal"}:
        inferred_length = config.get("length_mm", length_mm)
        if inferred_length is None:
            raise ValueError(
                "Longitudinal gradient profiles require length_mm or a sampled "
                "object length from which length can be inferred."
            )
        return LongitudinalGradientProfile(
            start_value=float(config.get("start_value", 0.2)),
            end_value=float(config.get("end_value", 1.0)),
            length_mm=float(inferred_length),
            **common,
        )

    if kind in {"multi_peak_radial", "multi_peak", "multipeak_radial"}:
        centres = config.get(
            "peak_centres_fraction",
            config.get("peak_centers_fraction", (0.35, 0.75)),
        )
        return MultiPeakRadialProfile(
            base_value=float(config.get("base_value", 0.0)),
            peak_centres_fraction=tuple(float(value) for value in centres),
            peak_amplitudes=tuple(
                float(value) for value in config.get("peak_amplitudes", (1.0, 0.5))
            ),
            peak_widths_fraction=tuple(
                float(value)
                for value in config.get("peak_widths_fraction", (0.08, 0.10))
            ),
            **common,
        )

    if kind in {"one_sided_lesion", "one_sided_lesion_like", "lesion_one_sided"}:
        return OneSidedLesionProfile(
            baseline_value=float(config.get("baseline_value", 0.2)),
            lesion_delta=float(config.get("lesion_delta", 0.8)),
            lesion_side=str(config.get("lesion_side", "positive")),
            lesion_centre_mm=float(
                config.get("lesion_centre_mm", config.get("lesion_center_mm", 1.0))
            ),
            lesion_width_mm=float(config.get("lesion_width_mm", 1.0)),
            **common,
        )

    if kind in {
        "radial_longitudinal_gradient",
        "radial_plus_longitudinal",
        "radial_longitudinal",
    }:
        inferred_length = config.get("length_mm", length_mm)
        if inferred_length is None:
            raise ValueError(
                "Radial-longitudinal gradient profiles require length_mm or a "
                "sampled object length from which length can be inferred."
            )
        return RadialLongitudinalGradientProfile(
            centre_start_value=float(config.get("centre_start_value", 1.0)),
            centre_end_value=float(config.get("centre_end_value", 0.5)),
            edge_start_value=float(config.get("edge_start_value", 0.2)),
            edge_end_value=float(config.get("edge_end_value", 0.8)),
            length_mm=float(inferred_length),
            **common,
        )

    if kind in {
        "periodic_longitudinal",
        "longitudinal_periodic",
        "sinusoidal_longitudinal",
    }:
        inferred_length = config.get("length_mm", length_mm)
        if inferred_length is None:
            raise ValueError(
                "Periodic longitudinal profiles require length_mm or a sampled "
                "object length from which length can be inferred."
            )
        return PeriodicLongitudinalProfile(
            baseline_value=float(config.get("baseline_value", 0.5)),
            amplitude=float(config.get("amplitude", 0.2)),
            length_mm=float(inferred_length),
            periods=float(config.get("periods", 1.0)),
            phase_radians=float(config.get("phase_radians", 0.0)),
            **common,
        )

    raise ValueError(f"Unknown profile kind: {kind!r}.")


def _object_parameters_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract non-tube object parameters from a config mapping."""

    kind = _kind(config["kind"])

    if kind in {"sphere", "sphere_object"}:
        return {
            "centre_mm": _required_sequence(
                config,
                "centre_mm",
                "center_mm",
                "centre",
                "center",
                name="sphere centre",
            ),
            "radius_mm": _required_value(
                config,
                "radius_mm",
                "radius",
                name="sphere radius",
            ),
        }

    if kind in {"ellipsoid", "ellipsoid_object"}:
        return {
            "centre_mm": _required_sequence(
                config,
                "centre_mm",
                "center_mm",
                "centre",
                "center",
                name="ellipsoid centre",
            ),
            "radii_mm": _required_sequence(
                config,
                "radii_mm",
                "radii",
                "axes_mm",
                name="ellipsoid radii",
            ),
        }

    if kind in {"slab", "sheet", "slab_object", "sheet_object"}:
        parameters = {
            "centre_mm": _required_sequence(
                config,
                "centre_mm",
                "center_mm",
                "centre",
                "center",
                name="slab centre",
            ),
            "normal_axis": _required_value(
                config,
                "normal_axis",
                "axis",
                name="slab normal axis",
            ),
            "thickness_mm": _required_value(
                config,
                "thickness_mm",
                "thickness",
                name="slab thickness",
            ),
        }
        if "half_extent_mm" in config:
            parameters["half_extent_mm"] = config["half_extent_mm"]
        elif "half_extents_mm" in config:
            parameters["half_extent_mm"] = config["half_extents_mm"]
        elif "extent_mm" in config:
            extent = config["extent_mm"]
            parameters["half_extent_mm"] = [float(value) / 2.0 for value in extent]
        return parameters

    if kind in {"cone", "cone_object"}:
        return {
            "apex_mm": _required_sequence(
                config,
                "apex_mm",
                "apex",
                name="cone apex",
            ),
            "axis": _required_value(
                config,
                "axis",
                name="cone axis",
            ),
            "axis_direction": config.get("axis_direction", 1),
            "height_mm": _required_value(
                config,
                "height_mm",
                "height",
                name="cone height",
            ),
            "base_radius_mm": _required_value(
                config,
                "base_radius_mm",
                "radius_mm",
                "base_radius",
                "radius",
                name="cone base radius",
            ),
        }

    if kind in {"frustum", "frustum_object", "truncated_cone"}:
        return {
            "start_mm": _required_sequence(
                config,
                "start_mm",
                "base_centre_mm",
                "base_center_mm",
                "centre_start_mm",
                "center_start_mm",
                name="frustum start",
            ),
            "axis": _required_value(
                config,
                "axis",
                name="frustum axis",
            ),
            "axis_direction": config.get("axis_direction", 1),
            "height_mm": _required_value(
                config,
                "height_mm",
                "height",
                name="frustum height",
            ),
            "radius_start_mm": _required_value(
                config,
                "radius_start_mm",
                "start_radius_mm",
                name="frustum start radius",
            ),
            "radius_end_mm": _required_value(
                config,
                "radius_end_mm",
                "end_radius_mm",
                name="frustum end radius",
            ),
        }

    return {}


def _object_spec_from_config(config: Mapping[str, Any]) -> SceneObjectSpec:
    """Build a SceneObjectSpec from a config mapping."""

    kind = str(config["kind"])
    normalised_kind = _kind(kind)

    if normalised_kind == "tube":
        curve = _required_mapping(config, "curve")
        cross_section = _required_mapping(config, "cross_section")
    elif normalised_kind in {
        "sphere",
        "sphere_object",
        "ellipsoid",
        "ellipsoid_object",
        "slab",
        "sheet",
        "slab_object",
        "sheet_object",
        "cone",
        "cone_object",
        "frustum",
        "frustum_object",
        "truncated_cone",
    }:
        curve = {}
        cross_section = {}
    else:
        raise ValueError(f"Unknown object kind: {normalised_kind!r}.")

    return SceneObjectSpec(
        object_id=str(config["id"] if "id" in config else config["object_id"]),
        kind=kind,
        role=str(config.get("role", "target")),
        label=int(config.get("label", 1)),
        priority=int(config.get("priority", 0)),
        map_name=str(config.get("map_name", "scalar")),
        curve=curve,
        cross_section=cross_section,
        profile=_required_mapping(config, "profile"),
        parameters=_object_parameters_from_config(config),
        name=config.get("name"),
        description=config.get("description"),
        metadata=_optional_mapping(config, "metadata"),
    )


def _tube_from_spec(spec: SceneObjectSpec) -> TubeObject:
    """Build a TubeObject from a SceneObjectSpec."""

    centreline = _sample_curve(spec.curve, object_id=spec.object_id)
    cross_section = _cross_section_from_config(
        spec.cross_section,
        length_mm=centreline.length_mm,
    )
    profile = _profile_from_config(spec.profile, length_mm=centreline.length_mm)

    return TubeObject(
        object_id=spec.object_id,
        centreline=centreline,
        cross_section=cross_section,
        profile=profile,
        map_name=spec.map_name,
        role=spec.role,
        label=spec.label,
        priority=spec.priority,
        name=spec.name,
        description=spec.description,
        metadata=dict(spec.metadata),
    )


def _sphere_from_spec(spec: SceneObjectSpec) -> SphereObject:
    """Build a SphereObject from a SceneObjectSpec."""

    profile = _profile_from_config(spec.profile)
    return SphereObject(
        object_id=spec.object_id,
        centre_mm=spec.parameters["centre_mm"],
        radius_mm=float(spec.parameters["radius_mm"]),
        profile=profile,
        map_name=spec.map_name,
        role=spec.role,
        label=spec.label,
        priority=spec.priority,
        name=spec.name,
        description=spec.description,
        metadata=dict(spec.metadata),
    )


def _ellipsoid_from_spec(spec: SceneObjectSpec) -> EllipsoidObject:
    """Build an EllipsoidObject from a SceneObjectSpec."""

    profile = _profile_from_config(spec.profile)
    return EllipsoidObject(
        object_id=spec.object_id,
        centre_mm=spec.parameters["centre_mm"],
        radii_mm=spec.parameters["radii_mm"],
        profile=profile,
        map_name=spec.map_name,
        role=spec.role,
        label=spec.label,
        priority=spec.priority,
        name=spec.name,
        description=spec.description,
        metadata=dict(spec.metadata),
    )


def _slab_from_spec(spec: SceneObjectSpec) -> SlabObject:
    """Build a SlabObject from a SceneObjectSpec."""

    profile = _profile_from_config(spec.profile)
    return SlabObject(
        object_id=spec.object_id,
        centre_mm=spec.parameters["centre_mm"],
        normal_axis=spec.parameters["normal_axis"],
        thickness_mm=float(spec.parameters["thickness_mm"]),
        half_extent_mm=spec.parameters.get("half_extent_mm"),
        profile=profile,
        map_name=spec.map_name,
        role=spec.role,
        label=spec.label,
        priority=spec.priority,
        name=spec.name,
        description=spec.description,
        metadata=dict(spec.metadata),
    )


def _cone_from_spec(spec: SceneObjectSpec) -> ConeObject:
    """Build a ConeObject from a SceneObjectSpec."""

    profile = _profile_from_config(
        spec.profile,
        length_mm=float(spec.parameters["height_mm"]),
    )
    return ConeObject(
        object_id=spec.object_id,
        apex_mm=spec.parameters["apex_mm"],
        axis=spec.parameters["axis"],
        axis_direction=spec.parameters.get("axis_direction", 1),
        height_mm=float(spec.parameters["height_mm"]),
        base_radius_mm=float(spec.parameters["base_radius_mm"]),
        profile=profile,
        map_name=spec.map_name,
        role=spec.role,
        label=spec.label,
        priority=spec.priority,
        name=spec.name,
        description=spec.description,
        metadata=dict(spec.metadata),
    )


def _frustum_from_spec(spec: SceneObjectSpec) -> FrustumObject:
    """Build a FrustumObject from a SceneObjectSpec."""

    profile = _profile_from_config(
        spec.profile,
        length_mm=float(spec.parameters["height_mm"]),
    )
    return FrustumObject(
        object_id=spec.object_id,
        start_mm=spec.parameters["start_mm"],
        axis=spec.parameters["axis"],
        axis_direction=spec.parameters.get("axis_direction", 1),
        height_mm=float(spec.parameters["height_mm"]),
        radius_start_mm=float(spec.parameters["radius_start_mm"]),
        radius_end_mm=float(spec.parameters["radius_end_mm"]),
        profile=profile,
        map_name=spec.map_name,
        role=spec.role,
        label=spec.label,
        priority=spec.priority,
        name=spec.name,
        description=spec.description,
        metadata=dict(spec.metadata),
    )


def _object_from_spec(spec: SceneObjectSpec):
    """Build a renderable object from a SceneObjectSpec."""

    kind = _kind(spec.kind)
    if kind == "tube":
        return _tube_from_spec(spec)
    if kind in {"sphere", "sphere_object"}:
        return _sphere_from_spec(spec)
    if kind in {"ellipsoid", "ellipsoid_object"}:
        return _ellipsoid_from_spec(spec)
    if kind in {"slab", "sheet", "slab_object", "sheet_object"}:
        return _slab_from_spec(spec)
    if kind in {"cone", "cone_object"}:
        return _cone_from_spec(spec)
    if kind in {"frustum", "frustum_object", "truncated_cone"}:
        return _frustum_from_spec(spec)

    raise ValueError(f"Unknown object kind: {kind!r}.")


def _render_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate render keyword arguments passed to object renderers."""

    allowed = {"chunk_size"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown render option(s): {unknown}.")
    return dict(config)


def scene_spec_from_dict(payload: Mapping[str, Any]) -> SceneSpec:
    """Build a SceneSpec from a parsed YAML/JSON mapping."""

    payload = _as_mapping(payload, name="scene payload")
    scene_meta = _optional_mapping(payload, "scene")
    grid = _grid_from_config(_required_mapping(payload, "grid"))
    object_configs = payload.get("objects")
    if not isinstance(object_configs, Sequence) or isinstance(object_configs, str):
        raise ValueError("objects must be a sequence/list of object mappings.")

    objects = tuple(
        _object_spec_from_config(_as_mapping(config, name="object"))
        for config in object_configs
    )

    return SceneSpec(
        scene_id=str(
            scene_meta.get("id")
            or payload.get("scene_id")
            or payload.get("id")
            or "scene"
        ),
        description=scene_meta.get("description") or payload.get("description"),
        schema_version=str(payload.get("schema_version", "0.1")),
        grid=grid,
        objects=objects,
        composition=_composition_from_config(_optional_mapping(payload, "composition")),
        mask_rules=_mask_rules_from_config(_optional_mapping(payload, "mask_rules")),
        render=_render_kwargs(_optional_mapping(payload, "render")),
        perturbations=_perturbations_from_config(payload.get("perturbations", ())),
        effects=_effects_from_config(payload.get("effects", ())),
        metadata={
            "source_scene_section": dict(scene_meta),
            **dict(_optional_mapping(payload, "metadata")),
        },
    )


def scene_spec_to_dict(spec: SceneSpec) -> dict[str, Any]:
    """Return a compact serialisable summary of a SceneSpec."""

    return {
        "schema_version": spec.schema_version,
        "scene": {
            "id": spec.scene_id,
            "description": spec.description,
        },
        "grid": spec.grid.summary(),
        "n_objects": len(spec.objects),
        "objects": [
            {
                "id": obj.object_id,
                "kind": obj.kind,
                "role": obj.role,
                "label": obj.label,
                "priority": obj.priority,
                "map_name": obj.map_name,
                "parameters": dict(obj.parameters),
            }
            for obj in spec.objects
        ],
        "composition": {
            "label_mode": spec.composition.label_mode.value,
            "scalar_blend": spec.composition.scalar_blend.value,
            "overlap_policy": spec.composition.overlap_policy.value,
        },
        "mask_rules": {
            "target_roles": [role.value for role in spec.mask_rules.target_roles],
            "analysis_roles": [role.value for role in spec.mask_rules.analysis_roles],
        },
        "render": dict(spec.render),
        "perturbations": [dict(item) for item in spec.perturbations],
        "effects": [dict(item) for item in spec.effects],
        "metadata": dict(spec.metadata),
    }


def render_scene_from_spec(spec: SceneSpec) -> RenderedScene:
    """Render a SceneSpec into a RenderedScene."""

    objects = [_object_from_spec(object_spec) for object_spec in spec.objects]
    scene = render_objects(
        spec.grid,
        objects,
        composition=spec.composition,
        mask_rules=spec.mask_rules,
        render_kwargs=spec.render,
        metadata={
            "scene_id": spec.scene_id,
            "description": spec.description,
            "schema_version": spec.schema_version,
            "scene_spec": scene_spec_to_dict(spec),
        },
        provenance={
            "source": "SceneSpec",
            "scene_id": spec.scene_id,
        },
    )
    scene = apply_effects(scene, spec.effects)
    return apply_perturbations(scene, spec.perturbations)


def render_scene_from_dict(payload: Mapping[str, Any]) -> RenderedScene:
    """Build and render a scene from a parsed scene mapping."""

    return render_scene_from_spec(scene_spec_from_dict(payload))


def _load_yaml(path: Path) -> Mapping[str, Any]:
    """Load a YAML mapping from disk."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "YAML scene loading requires PyYAML. Install with "
            '`python -m pip install -e ".[io]"` or `python -m pip install PyYAML`.'
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        raise ValueError(f"Scene file is empty: {path}")
    return _as_mapping(payload, name="YAML scene payload")


def _load_json(path: Path) -> Mapping[str, Any]:
    """Load a JSON mapping from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return _as_mapping(payload, name="JSON scene payload")


def load_scene_spec(path: str | Path) -> SceneSpec:
    """Load a SceneSpec from YAML or JSON."""

    scene_path = Path(path)
    suffix = scene_path.suffix.lower()

    if suffix in {".yml", ".yaml"}:
        payload = _load_yaml(scene_path)
    elif suffix == ".json":
        payload = _load_json(scene_path)
    else:
        raise ValueError("Scene specs must be .yml, .yaml, or .json files.")

    return scene_spec_from_dict(payload)


def render_scene_from_path(path: str | Path) -> RenderedScene:
    """Load and render a YAML/JSON scene spec."""

    return render_scene_from_spec(load_scene_spec(path))
