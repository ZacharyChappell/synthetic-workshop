"""Scene containers, composition, and rendering helpers."""

from synthworkshop.scenes.compose import compose_rendered_scenes
from synthworkshop.scenes.model import (
    CompositionRules,
    LabelMode,
    MaskRules,
    ObjectRole,
    OverlapPolicy,
    OverlapReport,
    RenderedScene,
    ScalarBlend,
    SceneObjectMetadata,
)
from synthworkshop.scenes.render import RenderableObject, render_objects

__all__ = [
    "CompositionRules",
    "LabelMode",
    "MaskRules",
    "ObjectRole",
    "OverlapPolicy",
    "OverlapReport",
    "RenderableObject",
    "RenderedScene",
    "ScalarBlend",
    "SceneObjectMetadata",
    "compose_rendered_scenes",
    "render_objects",
]
