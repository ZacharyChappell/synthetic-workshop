"""Known-effect injection for rendered synthetic scenes."""

from synthworkshop.effects.apply import (
    EFFECT_REGISTRY,
    apply_effect,
    apply_effects,
    available_effects,
)
from synthworkshop.effects.base import EffectRecord
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

__all__ = [
    "EFFECT_REGISTRY",
    "EffectRecord",
    "add_axis_interval_value_shift",
    "add_branch_value_shift",
    "add_centre_value_shift",
    "add_edge_value_shift",
    "add_hollow_core_change",
    "add_multi_object_value_shift",
    "add_object_value_shift",
    "add_rim_enhancement",
    "apply_effect",
    "apply_effects",
    "available_effects",
    "contract_object_width",
    "expand_object_width",
    "inject_no_effect",
]
