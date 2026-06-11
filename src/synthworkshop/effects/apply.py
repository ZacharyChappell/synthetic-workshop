"""Apply configured known effects to rendered scenes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from synthworkshop.effects.localised import (
    add_axis_interval_value_shift,
    add_branch_value_shift,
)
from synthworkshop.effects.morphology import (
    add_hollow_core_change,
    add_rim_enhancement,
    contract_object_width,
    expand_object_width,
)
from synthworkshop.effects.scalar import (
    add_centre_value_shift,
    add_edge_value_shift,
    add_multi_object_value_shift,
    add_object_value_shift,
    inject_no_effect,
)
from synthworkshop.scenes import RenderedScene

EffectFunction = Callable[..., RenderedScene]


EFFECT_REGISTRY: dict[str, EffectFunction] = {
    "no_effect": inject_no_effect,
    "no_effect_null": inject_no_effect,
    "inject_no_effect": inject_no_effect,
    "object_value_shift": add_object_value_shift,
    "add_object_value_shift": add_object_value_shift,
    "whole_object_shift": add_object_value_shift,
    "centre_value_shift": add_centre_value_shift,
    "center_value_shift": add_centre_value_shift,
    "add_centre_value_shift": add_centre_value_shift,
    "add_center_value_shift": add_centre_value_shift,
    "edge_value_shift": add_edge_value_shift,
    "add_edge_value_shift": add_edge_value_shift,
    "multi_object_value_shift": add_multi_object_value_shift,
    "add_multi_object_value_shift": add_multi_object_value_shift,
    "width_expansion": expand_object_width,
    "expand_object_width": expand_object_width,
    "width_contraction": contract_object_width,
    "contract_object_width": contract_object_width,
    "rim_enhancement": add_rim_enhancement,
    "add_rim_enhancement": add_rim_enhancement,
    "hollow_core_change": add_hollow_core_change,
    "add_hollow_core_change": add_hollow_core_change,
    "axis_interval_value_shift": add_axis_interval_value_shift,
    "longitudinal_local_value_shift": add_axis_interval_value_shift,
    "add_axis_interval_value_shift": add_axis_interval_value_shift,
    "branch_value_shift": add_branch_value_shift,
    "branch_specific_value_shift": add_branch_value_shift,
    "add_branch_value_shift": add_branch_value_shift,
}


def available_effects() -> tuple[str, ...]:
    """Return known effect kinds."""

    return tuple(sorted(EFFECT_REGISTRY))


def apply_effect(
    scene: RenderedScene,
    effect: Mapping[str, Any],
) -> RenderedScene:
    """Apply one known-effect specification to a scene."""

    if not isinstance(effect, Mapping):
        raise TypeError("effect must be a mapping.")

    spec = dict(effect)
    kind = spec.pop("kind", None)
    if kind is None:
        raise ValueError("effect specification must include a 'kind' field.")
    if not isinstance(kind, str) or not kind:
        raise ValueError("effect 'kind' must be a non-empty string.")

    try:
        effect_function = EFFECT_REGISTRY[kind]
    except KeyError as error:
        known = ", ".join(available_effects())
        raise ValueError(
            f"Unknown effect kind {kind!r}. Known kinds: {known}."
        ) from error

    return effect_function(scene, **spec)


def apply_effects(
    scene: RenderedScene,
    effects: Iterable[Mapping[str, Any]] | None,
) -> RenderedScene:
    """Apply known-effect specifications in order."""

    if effects is None:
        return scene

    current = scene
    for effect in effects:
        current = apply_effect(current, effect)
    return current
