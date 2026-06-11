"""Tests for morphology and profile-support effects."""

from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.effects import (
    add_hollow_core_change,
    add_rim_enhancement,
    contract_object_width,
    expand_object_width,
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

    label_map = np.zeros(grid.shape, dtype=np.int32)
    label_map[target_mask] = 1

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
        metadata={"scene_id": "morphology_effect_test"},
    )


def test_expand_object_width_dilates_mask_and_label_map() -> None:
    scene = _scene()

    effected = expand_object_width(scene, object_id="target", iterations=1)

    assert effected.object_masks["target"].sum() == 125
    assert effected.label_map.sum() == 125
    assert effected.target_masks["target"].sum() == 125

    record = effected.truth.metadata["effects"]["001_width_expansion"]
    assert record["support_voxels"] == 98
    assert record["expected_direction"] == "increase"


def test_contract_object_width_erodes_mask_and_label_map() -> None:
    scene = _scene()

    effected = contract_object_width(scene, object_id="target", iterations=1)

    assert effected.object_masks["target"].sum() == 1
    assert effected.label_map.sum() == 1
    assert effected.object_masks["target"][2, 2, 2]

    record = effected.truth.metadata["effects"]["001_width_contraction"]
    assert record["support_voxels"] == 26
    assert record["expected_direction"] == "decrease"


def test_rim_enhancement_changes_edge_not_core() -> None:
    scene = _scene()

    effected = add_rim_enhancement(
        scene,
        object_id="target",
        map_name="scalar",
        delta=0.5,
        erosion_iterations=1,
    )

    assert effected.scalar_maps["scalar"][1, 1, 1] == pytest.approx(1.5)
    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.0)

    record = effected.truth.metadata["effects"]["001_rim_enhancement"]
    assert record["support_voxels"] == 26
    assert record["expected_direction"] == "increase"


def test_hollow_core_change_changes_core_not_edge() -> None:
    scene = _scene()

    effected = add_hollow_core_change(
        scene,
        object_id="target",
        map_name="scalar",
        delta=-0.5,
        erosion_iterations=1,
    )

    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(0.5)
    assert effected.scalar_maps["scalar"][1, 1, 1] == pytest.approx(1.0)

    record = effected.truth.metadata["effects"]["001_hollow_core_change"]
    assert record["support_voxels"] == 1
    assert record["expected_direction"] == "decrease"


def test_width_effect_rejects_unknown_object() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="Unknown object_id"):
        expand_object_width(scene, object_id="missing")


def test_width_effect_rejects_invalid_iterations() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="iterations"):
        contract_object_width(scene, object_id="target", iterations=0)
