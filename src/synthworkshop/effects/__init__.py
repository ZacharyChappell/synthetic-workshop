"""Known-effect injection for rendered synthetic scenes."""

from synthworkshop.effects.base import EffectRecord
from synthworkshop.effects.scalar import (
    add_centre_value_shift,
    add_edge_value_shift,
    add_multi_object_value_shift,
    add_object_value_shift,
    inject_no_effect,
)

__all__ = [
    "EffectRecord",
    "add_centre_value_shift",
    "add_edge_value_shift",
    "add_multi_object_value_shift",
    "add_object_value_shift",
    "inject_no_effect",
]
