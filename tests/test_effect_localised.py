"""Tests for localised known effects."""

from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.effects import (
    add_axis_interval_value_shift,
    add_branch_value_shift,
)
from synthworkshop.grid import GridSpec
from synthworkshop.scenes import (
    CompositionRules,
    RenderedScene,
    SceneObjectMetadata,
)


def _axis_scene() -> RenderedScene:
    grid = GridSpec(shape=(5, 5, 5), spacing=(1.0, 1.0, 1.0))
    scalar = np.ones(grid.shape, dtype=float)

    mask = np.zeros(grid.shape, dtype=bool)
    mask[1:4, 1:4, 1:4] = True

    label_map = np.zeros(grid.shape, dtype=np.int32)
    label_map[mask] = 1

    return RenderedScene(
        grid=grid,
        scalar_maps={"scalar": scalar},
        label_map=label_map,
        object_masks={"target": mask},
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
        metadata={"scene_id": "axis_local_effect_test"},
    )


def _branch_scene() -> RenderedScene:
    grid = GridSpec(shape=(5, 5, 5), spacing=(1.0, 1.0, 1.0))
    scalar = np.ones(grid.shape, dtype=float)

    ab = np.zeros(grid.shape, dtype=bool)
    ab[1, 1:4, 2] = True

    bc = np.zeros(grid.shape, dtype=bool)
    bc[3, 1:4, 2] = True

    label_map = np.zeros(grid.shape, dtype=np.int32)
    label_map[ab] = 1
    label_map[bc] = 2

    return RenderedScene(
        grid=grid,
        scalar_maps={"scalar": scalar},
        label_map=label_map,
        object_masks={
            "graph__edge__ab": ab,
            "graph__edge__bc": bc,
        },
        object_metadata={
            "graph__edge__ab": SceneObjectMetadata(
                object_id="graph__edge__ab",
                role="target",
                label=1,
                priority=10,
                metadata={"graph_object_id": "graph", "edge_id": "ab"},
            ),
            "graph__edge__bc": SceneObjectMetadata(
                object_id="graph__edge__bc",
                role="target",
                label=2,
                priority=9,
                metadata={"graph_object_id": "graph", "edge_id": "bc"},
            ),
        },
        skeleton_masks={},
        composition=CompositionRules(overlap_policy="allow"),
        metadata={"scene_id": "branch_effect_test"},
    )


def test_axis_interval_value_shift_affects_only_interval_support() -> None:
    scene = _axis_scene()

    effected = add_axis_interval_value_shift(
        scene,
        object_id="target",
        map_name="scalar",
        delta=0.5,
        axis=0,
        start_mm=2.0,
        end_mm=2.0,
    )

    assert effected.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.5)
    assert effected.scalar_maps["scalar"][1, 2, 2] == pytest.approx(1.0)
    assert effected.scalar_maps["scalar"][4, 2, 2] == pytest.approx(1.0)

    record = effected.truth.metadata["effects"]["001_axis_interval_value_shift"]
    assert record["support_voxels"] == 9
    assert record["parameters"]["start_mm"] == pytest.approx(2.0)
    assert record["parameters"]["end_mm"] == pytest.approx(2.0)


def test_axis_interval_value_shift_records_empty_support_as_null() -> None:
    scene = _axis_scene()

    effected = add_axis_interval_value_shift(
        scene,
        object_id="target",
        map_name="scalar",
        delta=0.5,
        axis=0,
        start_mm=4.0,
        end_mm=4.0,
    )

    record = effected.truth.metadata["effects"]["001_axis_interval_value_shift"]
    assert record["support_voxels"] == 0
    assert record["clean_null"] is True
    assert record["truth_changed"] is False


def test_axis_interval_value_shift_rejects_invalid_interval() -> None:
    scene = _axis_scene()

    with pytest.raises(ValueError, match="end_mm"):
        add_axis_interval_value_shift(
            scene,
            object_id="target",
            map_name="scalar",
            delta=0.5,
            axis=0,
            start_mm=3.0,
            end_mm=2.0,
        )


def test_axis_interval_value_shift_rejects_invalid_axis() -> None:
    scene = _axis_scene()

    with pytest.raises(ValueError, match="axis"):
        add_axis_interval_value_shift(
            scene,
            object_id="target",
            map_name="scalar",
            delta=0.5,
            axis=3,
            start_mm=1.0,
            end_mm=2.0,
        )


def test_branch_value_shift_targets_graph_edge_object() -> None:
    scene = _branch_scene()

    effected = add_branch_value_shift(
        scene,
        graph_object_id="graph",
        edge_id="ab",
        map_name="scalar",
        delta=0.5,
    )

    assert effected.scalar_maps["scalar"][1, 2, 2] == pytest.approx(1.5)
    assert effected.scalar_maps["scalar"][3, 2, 2] == pytest.approx(1.0)

    record = effected.truth.metadata["effects"]["001_branch_value_shift"]
    assert record["affected_objects"] == ["graph__edge__ab"]
    assert record["support_voxels"] == 3
    assert record["parameters"]["edge_id"] == "ab"


def test_branch_value_shift_accepts_explicit_branch_object_id() -> None:
    scene = _branch_scene()

    effected = add_branch_value_shift(
        scene,
        branch_object_id="graph__edge__bc",
        map_name="scalar",
        delta=-0.25,
    )

    assert effected.scalar_maps["scalar"][3, 2, 2] == pytest.approx(0.75)
    assert effected.scalar_maps["scalar"][1, 2, 2] == pytest.approx(1.0)


def test_branch_value_shift_requires_branch_identifier() -> None:
    scene = _branch_scene()

    with pytest.raises(ValueError, match="branch_object_id"):
        add_branch_value_shift(
            scene,
            map_name="scalar",
            delta=0.5,
        )
