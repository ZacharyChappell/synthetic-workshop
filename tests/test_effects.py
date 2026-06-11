"""Tests for known-effect injection."""

from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.effects import (
    add_centre_value_shift,
    add_edge_value_shift,
    add_multi_object_value_shift,
    add_object_value_shift,
    inject_no_effect,
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

    second_mask = np.zeros(grid.shape, dtype=bool)
    second_mask[0, 0, 0] = True

    label_map = np.zeros(grid.shape, dtype=np.int32)
    label_map[target_mask] = 1
    label_map[second_mask] = 2

    return RenderedScene(
        grid=grid,
        scalar_maps={"scalar": scalar},
        label_map=label_map,
        object_masks={
            "target": target_mask,
            "second": second_mask,
        },
        object_metadata={
            "target": SceneObjectMetadata(
                object_id="target",
                role="target",
                label=1,
                priority=10,
            ),
            "second": SceneObjectMetadata(
                object_id="second",
                role="target",
                label=2,
                priority=5,
            ),
        },
        skeleton_masks={},
        composition=CompositionRules(overlap_policy="allow"),
        metadata={"scene_id": "effect_test"},
    )


def test_inject_no_effect_records_explicit_null_without_changing_arrays() -> None:
    scene = _scene()

    effected = inject_no_effect(scene)

    np.testing.assert_allclose(
        effected.scalar_maps["scalar"], scene.scalar_maps["scalar"]
    )
    record = effected.truth.metadata["effects"]["001_no_effect_null"]
    assert record["clean_null"] is True
    assert record["truth_changed"] is False
    assert effected.metadata["effects"][0]["expected_direction"] == "none"


def test_add_object_value_shift_changes_only_object_support() -> None:
    scene = _scene()

    effected = add_object_value_shift(
        scene,
        object_id="target",
        map_name="scalar",
        delta=0.5,
    )

    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.5)
    assert effected.scalar_maps["scalar"][0, 4, 4] == pytest.approx(1.0)
    assert (
        effected.truth.metadata["effects"]["001_object_value_shift"]["support_voxels"]
        == 27
    )


def test_add_centre_value_shift_changes_eroded_centre() -> None:
    scene = _scene()

    effected = add_centre_value_shift(
        scene,
        object_id="target",
        map_name="scalar",
        delta=0.5,
        erosion_iterations=1,
    )

    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.5)
    assert effected.scalar_maps["scalar"][1, 1, 1] == pytest.approx(1.0)
    assert (
        effected.truth.metadata["effects"]["001_centre_value_shift"]["support_voxels"]
        == 1
    )


def test_add_edge_value_shift_changes_boundary_not_centre() -> None:
    scene = _scene()

    effected = add_edge_value_shift(
        scene,
        object_id="target",
        map_name="scalar",
        delta=-0.25,
        erosion_iterations=1,
    )

    assert effected.scalar_maps["scalar"][1, 1, 1] == pytest.approx(0.75)
    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.0)

    record = effected.truth.metadata["effects"]["001_edge_value_shift"]
    assert record["support_voxels"] == 26
    assert record["expected_direction"] == "decrease"


def test_multiple_effects_are_recorded_in_order() -> None:
    scene = _scene()

    effected = inject_no_effect(scene)
    effected = add_object_value_shift(
        effected,
        object_id="target",
        map_name="scalar",
        delta=0.5,
    )

    assert list(effected.truth.metadata["effects"]) == [
        "001_no_effect_null",
        "002_object_value_shift",
    ]
    assert len(effected.metadata["effects"]) == 2


def test_add_multi_object_value_shift_applies_to_selected_objects() -> None:
    scene = _scene()

    effected = add_multi_object_value_shift(
        scene,
        object_ids=["target", "second"],
        map_name="scalar",
        delta=0.5,
    )

    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.5)
    assert effected.scalar_maps["scalar"][0, 0, 0] == pytest.approx(1.5)
    assert len(effected.metadata["effects"]) == 2


def test_scalar_effect_rejects_unknown_object() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="Unknown object_id"):
        add_object_value_shift(
            scene,
            object_id="missing",
            map_name="scalar",
            delta=0.5,
        )


def test_scalar_effect_rejects_unknown_map() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="Unknown scalar map"):
        add_object_value_shift(
            scene,
            object_id="target",
            map_name="missing",
            delta=0.5,
        )


def test_scalar_effect_rejects_invalid_erosion_iterations() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="erosion_iterations"):
        add_centre_value_shift(
            scene,
            object_id="target",
            map_name="scalar",
            delta=0.5,
            erosion_iterations=0,
        )
