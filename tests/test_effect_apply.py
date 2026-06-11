"""Tests for known-effect dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.effects import (
    apply_effect,
    apply_effects,
    available_effects,
)
from synthworkshop.grid import GridSpec
from synthworkshop.scenes import (
    CompositionRules,
    RenderedScene,
    SceneObjectMetadata,
)


def _scene() -> RenderedScene:
    grid = GridSpec(shape=(5, 5, 5), spacing=(1.0, 1.0, 1.0))
    scalar = np.ones(grid.shape, dtype=float)

    target_mask = np.zeros(grid.shape, dtype=bool)
    target_mask[1:4, 1:4, 1:4] = True

    label_map = np.where(target_mask, 1, 0).astype(np.int32)

    return RenderedScene(
        grid=grid,
        scalar_maps={"scalar": scalar},
        label_map=label_map,
        object_masks={"target": target_mask},
        object_metadata={
            "target": SceneObjectMetadata(
                object_id="target",
                role="target",
                label=1,
                priority=10,
            )
        },
        skeleton_masks={},
        composition=CompositionRules(overlap_policy="allow"),
        metadata={"scene_id": "effect_apply_test"},
    )


def test_available_effects_exposes_core_kinds() -> None:
    kinds = available_effects()

    assert "no_effect_null" in kinds
    assert "object_value_shift" in kinds
    assert "centre_value_shift" in kinds
    assert "edge_value_shift" in kinds


def test_apply_effect_dispatches_one_specification() -> None:
    scene = _scene()

    effected = apply_effect(
        scene,
        {
            "kind": "object_value_shift",
            "object_id": "target",
            "map_name": "scalar",
            "delta": 0.5,
        },
    )

    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.5)
    assert "001_object_value_shift" in effected.truth.metadata["effects"]


def test_apply_effects_applies_specs_in_order() -> None:
    scene = _scene()

    effected = apply_effects(
        scene,
        [
            {
                "kind": "no_effect_null",
            },
            {
                "kind": "centre_value_shift",
                "object_id": "target",
                "map_name": "scalar",
                "delta": 0.5,
                "erosion_iterations": 1,
            },
            {
                "kind": "edge_value_shift",
                "object_id": "target",
                "map_name": "scalar",
                "delta": -0.25,
                "erosion_iterations": 1,
            },
        ],
    )

    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.5)
    assert effected.scalar_maps["scalar"][1, 1, 1] == pytest.approx(0.75)
    assert list(effected.truth.metadata["effects"]) == [
        "001_no_effect_null",
        "002_centre_value_shift",
        "003_edge_value_shift",
    ]


def test_apply_effects_accepts_none_or_empty_specs() -> None:
    scene = _scene()

    assert apply_effects(scene, None) is scene
    assert apply_effects(scene, []) is scene


def test_apply_effect_rejects_missing_kind() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="kind"):
        apply_effect(scene, {"delta": 0.5})


def test_apply_effect_rejects_unknown_kind() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="Unknown effect kind"):
        apply_effect(scene, {"kind": "does_not_exist"})


def test_apply_effect_rejects_non_mapping_specification() -> None:
    scene = _scene()

    with pytest.raises(TypeError, match="mapping"):
        apply_effect(scene, ["not", "a", "mapping"])  # type: ignore[arg-type]


def test_apply_effect_dispatches_width_and_profile_support_effects() -> None:
    scene = _scene()

    widened = apply_effect(
        scene,
        {
            "kind": "width_expansion",
            "object_id": "target",
            "iterations": 1,
        },
    )
    assert "001_width_expansion" in widened.truth.metadata["effects"]

    rimmed = apply_effect(
        scene,
        {
            "kind": "rim_enhancement",
            "object_id": "target",
            "map_name": "scalar",
            "delta": 0.5,
        },
    )
    assert "001_rim_enhancement" in rimmed.truth.metadata["effects"]


def test_apply_effect_dispatches_localised_effects() -> None:
    scene = _scene()

    localised = apply_effect(
        scene,
        {
            "kind": "axis_interval_value_shift",
            "object_id": "target",
            "map_name": "scalar",
            "delta": 0.5,
            "axis": 0,
            "start_mm": 2.0,
            "end_mm": 2.0,
        },
    )

    assert "001_axis_interval_value_shift" in localised.truth.metadata["effects"]
