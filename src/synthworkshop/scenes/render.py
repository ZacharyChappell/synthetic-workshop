"""Scene-rendering helpers for composable renderable objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from synthworkshop.grid import GridSpec
from synthworkshop.scenes.compose import compose_rendered_scenes
from synthworkshop.scenes.model import CompositionRules, MaskRules, RenderedScene


@runtime_checkable
class RenderableObject(Protocol):
    """Protocol for objects that can render themselves on a GridSpec."""

    def render(self, grid: GridSpec, **kwargs: Any) -> RenderedScene:
        """Render the object on a grid."""


def render_objects(
    grid: GridSpec,
    objects: Sequence[RenderableObject],
    *,
    composition: CompositionRules | None = None,
    mask_rules: MaskRules | None = None,
    scalar_weights: Mapping[str, float] | None = None,
    render_kwargs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> RenderedScene:
    """Render multiple objects and compose them into one scene."""

    if not objects:
        raise ValueError("At least one renderable object is required.")

    kwargs = dict(render_kwargs or {})
    rendered = [object_.render(grid, **kwargs) for object_ in objects]

    return compose_rendered_scenes(
        rendered,
        composition=composition,
        mask_rules=mask_rules,
        scalar_weights=scalar_weights,
        metadata=metadata,
        provenance=provenance,
    )
