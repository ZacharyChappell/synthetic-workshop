from __future__ import annotations

import numpy as np
import pytest

from synthworkshop import GridSpec
from synthworkshop.cross_sections import CircularCrossSection
from synthworkshop.profiles import LinearRadialProfile
from synthworkshop.topology import (
    GraphEdge,
    GraphNode,
    GraphSpec,
    GraphTubeObject,
)


def _t_graph() -> GraphSpec:
    return GraphSpec(
        graph_id="t_graph",
        nodes=(
            GraphNode("root", (4.0, 12.0, 12.0)),
            GraphNode("junction", (12.0, 12.0, 12.0)),
            GraphNode("left", (20.0, 8.0, 12.0)),
            GraphNode("right", (20.0, 16.0, 12.0)),
        ),
        edges=(
            GraphEdge("trunk", "root", "junction"),
            GraphEdge("left_branch", "junction", "left"),
            GraphEdge("right_branch", "junction", "right"),
        ),
    )


def test_graph_tube_object_renders_edge_masks_and_scalar_map() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(centre_value=1.0, edge_value=0.2),
        map_name="fa_like",
        role="target",
        label=7,
        priority=10,
    )

    scene = graph_tube.render(grid)

    expected = {
        "target_graph__edge__trunk",
        "target_graph__edge__left_branch",
        "target_graph__edge__right_branch",
    }
    assert set(scene.object_masks) == expected
    assert "fa_like" in scene.scalar_maps
    assert scene.scalar_maps["fa_like"].shape == grid.shape
    assert scene.target_masks["target"].sum() > 0
    assert scene.analysis_masks["analysis"].sum() == scene.target_masks["target"].sum()
    assert np.any(scene.label_map == 7)


def test_graph_tube_records_graph_truth_tables() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(),
        map_name="scalar",
        role="target",
        label=1,
    )

    scene = graph_tube.render(grid)

    for table_name in (
        "graph_nodes",
        "graph_edges",
        "graph_components",
        "graph_centrelines",
        "graph_edge_summaries",
    ):
        assert table_name in scene.truth.tables

    nodes = scene.truth.tables["graph_nodes"]
    assert nodes.loc[nodes["node_id"] == "junction", "is_junction"].item()
    assert scene.truth.geometric["target_graph"]["kind"] == "graph_tube"
    assert scene.truth.objects["target_graph"]["n_edges"] == 3


def test_graph_tube_preserves_edge_centrelines_and_frames() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.5),
        profile=LinearRadialProfile(),
        n_samples_per_edge=4,
    )

    scene = graph_tube.render(grid)

    expected_edge_keys = {
        "target_graph__edge__trunk",
        "target_graph__edge__left_branch",
        "target_graph__edge__right_branch",
    }

    assert expected_edge_keys.issubset(scene.centrelines)
    assert "target_graph" in scene.centrelines
    assert expected_edge_keys.issubset(scene.frames)
    assert "target_graph" in scene.frames

    graph_centrelines = scene.truth.tables["graph_centrelines"]
    assert set(graph_centrelines["segment_id"]) == {
        "trunk",
        "left_branch",
        "right_branch",
    }
    assert set(graph_centrelines["object_id"]) == {"target_graph"}


def test_graph_tube_overlap_policy_error_raises_at_junction() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(),
    )

    with pytest.raises(ValueError, match="overlap_policy='error'"):
        graph_tube.render(grid, overlap_policy="error")


def test_graph_tube_records_overlap_report_by_default() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(),
    )

    scene = graph_tube.render(grid)

    assert scene.overlap_report.has_overlap
    assert scene.overlap_report.n_overlap_voxels > 0
    assert scene.truth.tables["overlaps"].shape[0] > 0


def test_graph_tube_rejects_2d_graph() -> None:
    graph = GraphSpec(
        graph_id="two_d",
        nodes=(
            GraphNode("a", (0.0, 0.0)),
            GraphNode("b", (1.0, 0.0)),
        ),
        edges=(GraphEdge("ab", "a", "b"),),
    )

    with pytest.raises(ValueError, match="3D graph"):
        GraphTubeObject(
            object_id="target_graph",
            graph=graph,
            cross_section=CircularCrossSection(radius_mm=1.0),
            profile=LinearRadialProfile(),
        )


def test_graph_tube_rejects_2d_grid() -> None:
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.0),
        profile=LinearRadialProfile(),
    )
    grid = GridSpec(shape=(16, 16), spacing=(1.0, 1.0))

    with pytest.raises(ValueError, match="3D grid"):
        graph_tube.render(grid)


def test_graph_tube_edge_tube_objects_have_stable_ids() -> None:
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.0),
        profile=LinearRadialProfile(),
    )
    edge_tubes = graph_tube.edge_tube_objects()

    assert {tube.object_id for tube in edge_tubes} == {
        "target_graph__edge__trunk",
        "target_graph__edge__left_branch",
        "target_graph__edge__right_branch",
    }


def test_top_level_exports_graph_tube_object() -> None:
    import synthworkshop

    assert synthworkshop.GraphTubeObject is GraphTubeObject


def test_graph_tube_supports_branch_specific_cross_sections() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.0),
        edge_cross_sections={
            "left_branch": CircularCrossSection(radius_mm=3.0),
        },
        profile=LinearRadialProfile(),
        map_name="scalar",
        role="target",
        label=1,
    )

    scene = graph_tube.render(grid)

    trunk_voxels = scene.object_masks["target_graph__edge__trunk"].sum()
    left_voxels = scene.object_masks["target_graph__edge__left_branch"].sum()

    assert left_voxels > trunk_voxels
    parameters = scene.truth.tables["graph_edge_parameters"]
    left_row = parameters.loc[parameters["edge_id"] == "left_branch"].iloc[0]
    trunk_row = parameters.loc[parameters["edge_id"] == "trunk"].iloc[0]
    assert left_row["cross_section_kind"] == "circle"
    assert trunk_row["cross_section_kind"] == "circle"


def test_graph_tube_supports_branch_specific_labels_priorities_and_roles() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.0),
        profile=LinearRadialProfile(),
        role="target",
        label=1,
        priority=1,
        edge_labels={
            "left_branch": 2,
            "right_branch": 3,
        },
        edge_priorities={
            "left_branch": 5,
            "right_branch": 7,
        },
        edge_roles={
            "right_branch": "analysis_support",
        },
    )

    scene = graph_tube.render(grid)

    left_id = "target_graph__edge__left_branch"
    right_id = "target_graph__edge__right_branch"

    assert scene.object_metadata[left_id].label == 2
    assert scene.object_metadata[left_id].priority == 5
    assert scene.object_metadata[right_id].label == 3
    assert scene.object_metadata[right_id].priority == 7
    assert scene.object_metadata[right_id].role.value == "analysis_support"

    assert scene.target_masks["target"].sum() < scene.analysis_masks["analysis"].sum()
    parameters = scene.truth.tables["graph_edge_parameters"]
    assert set(parameters["label"]) == {1, 2, 3}
    assert set(parameters["role"]) == {"target", "analysis_support"}


def test_graph_tube_supports_branch_specific_profiles_and_map_names() -> None:
    from synthworkshop.profiles import ConstantProfile

    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.0),
        profile=ConstantProfile(value=0.2),
        map_name="fa_like",
        role="target",
        label=1,
        edge_profiles={
            "right_branch": ConstantProfile(value=0.9),
        },
        edge_map_names={
            "right_branch": "md_like",
        },
    )

    scene = graph_tube.render(grid)

    assert set(scene.scalar_maps) == {"fa_like", "md_like"}
    assert np.isclose(scene.scalar_maps["fa_like"][8, 12, 12], 0.2)
    assert np.isclose(scene.scalar_maps["md_like"][16, 14, 12], 0.9)
    assert np.isclose(scene.scalar_maps["fa_like"][16, 14, 12], 0.0)

    parameters = scene.truth.tables["graph_edge_parameters"]
    right_row = parameters.loc[parameters["edge_id"] == "right_branch"].iloc[0]
    assert right_row["map_name"] == "md_like"
    assert right_row["profile_kind"] == "constant"


def test_graph_tube_rejects_unknown_edge_override_keys() -> None:
    with pytest.raises(ValueError, match="edge_cross_sections"):
        GraphTubeObject(
            object_id="target_graph",
            graph=_t_graph(),
            cross_section=CircularCrossSection(radius_mm=1.0),
            edge_cross_sections={
                "not_an_edge": CircularCrossSection(radius_mm=2.0),
            },
            profile=LinearRadialProfile(),
        )

    with pytest.raises(ValueError, match="edge_labels"):
        GraphTubeObject(
            object_id="target_graph",
            graph=_t_graph(),
            cross_section=CircularCrossSection(radius_mm=1.0),
            profile=LinearRadialProfile(),
            edge_labels={"not_an_edge": 2},
        )


def test_graph_tube_edge_parameter_table_contains_defaults_and_overrides() -> None:
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.0),
        profile=LinearRadialProfile(),
        map_name="fa_like",
        role="target",
        label=1,
        priority=1,
        edge_labels={"left_branch": 2},
        edge_map_names={"right_branch": "md_like"},
    )

    table = graph_tube.edge_parameter_table()

    assert table.shape[0] == 3
    assert {
        "graph_id",
        "graph_object_id",
        "edge_id",
        "edge_object_id",
        "role",
        "label",
        "priority",
        "map_name",
        "cross_section_kind",
        "profile_kind",
    }.issubset(table.columns)

    left_row = table.loc[table["edge_id"] == "left_branch"].iloc[0]
    right_row = table.loc[table["edge_id"] == "right_branch"].iloc[0]
    trunk_row = table.loc[table["edge_id"] == "trunk"].iloc[0]

    assert left_row["label"] == 2
    assert right_row["map_name"] == "md_like"
    assert trunk_row["label"] == 1
    assert trunk_row["map_name"] == "fa_like"


def test_graph_tube_adds_parent_graph_skeleton_mask() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.5),
        profile=LinearRadialProfile(),
    )

    scene = graph_tube.render(grid)

    assert "target_graph" in scene.skeleton_masks
    edge_union = np.zeros(grid.shape, dtype=bool)
    for edge_id in ("trunk", "left_branch", "right_branch"):
        edge_union |= scene.skeleton_masks[f"target_graph__edge__{edge_id}"]

    assert np.array_equal(scene.skeleton_masks["target_graph"], edge_union)
    assert scene.truth.geometric["target_graph"]["graph_skeleton_key"] == "target_graph"
    assert scene.truth.objects["target_graph"]["graph_skeleton_voxels"] == int(
        edge_union.sum()
    )


def test_graph_tube_adds_graph_level_centreline_and_frame_tables() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.5),
        profile=LinearRadialProfile(),
        n_samples_per_edge=4,
    )

    scene = graph_tube.render(grid)

    assert "target_graph" in scene.centrelines
    assert "target_graph" in scene.frames
    assert "graph_frames" in scene.truth.tables

    graph_centrelines = scene.centrelines["target_graph"]
    graph_frames = scene.frames["target_graph"]

    assert set(graph_centrelines["edge_id"]) == {
        "trunk",
        "left_branch",
        "right_branch",
    }
    assert set(graph_frames["edge_id"]) == {
        "trunk",
        "left_branch",
        "right_branch",
    }
    assert graph_centrelines.shape[0] == 12
    assert graph_frames.shape[0] == 12


def test_graph_tube_adds_endpoint_junction_and_incident_edge_tables() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.5),
        profile=LinearRadialProfile(),
    )

    scene = graph_tube.render(grid)

    for table_name in (
        "graph_node_topology",
        "graph_endpoints",
        "graph_junctions",
        "graph_incident_edges",
    ):
        assert table_name in scene.truth.tables

    endpoints = scene.truth.tables["graph_endpoints"]
    junctions = scene.truth.tables["graph_junctions"]
    incidences = scene.truth.tables["graph_incident_edges"]

    assert set(endpoints["node_id"]) == {"root", "left", "right"}
    assert set(junctions["node_id"]) == {"junction"}
    assert junctions.loc[0, "degree"] == 3
    assert set(incidences.loc[incidences["node_id"] == "junction", "edge_id"]) == {
        "trunk",
        "left_branch",
        "right_branch",
    }


def test_graph_tube_metadata_records_endpoint_and_junction_ids() -> None:
    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.5),
        profile=LinearRadialProfile(),
    )

    scene = graph_tube.render(grid)

    assert scene.metadata["junction_node_ids"] == ("junction",)
    assert set(scene.metadata["endpoint_node_ids"]) == {"root", "left", "right"}
    assert scene.truth.geometric["target_graph"]["junction_node_ids"] == ("junction",)
    assert set(scene.truth.geometric["target_graph"]["endpoint_node_ids"]) == {
        "root",
        "left",
        "right",
    }


def test_graph_node_topology_table_has_incident_edge_object_ids() -> None:
    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=_t_graph(),
        cross_section=CircularCrossSection(radius_mm=1.5),
        profile=LinearRadialProfile(),
    )

    table = graph_tube.graph_node_topology_table()
    junction = table.loc[table["node_id"] == "junction"].iloc[0]

    assert junction["node_kind"] == "junction"
    assert junction["degree"] == 3
    assert set(junction["incident_edges"].split("|")) == {
        "trunk",
        "left_branch",
        "right_branch",
    }
    assert set(junction["incident_edge_object_ids"].split("|")) == {
        "target_graph__edge__trunk",
        "target_graph__edge__left_branch",
        "target_graph__edge__right_branch",
    }


def test_graph_tube_supports_branch_specific_variable_radius_cross_section() -> None:
    from synthworkshop.cross_sections import VariableCircularCrossSection

    grid = GridSpec(shape=(28, 28, 28), spacing=(1.0, 1.0, 1.0))
    graph = _t_graph()
    left_length = graph.edge_length_mm("left_branch")

    graph_tube = GraphTubeObject(
        object_id="target_graph",
        graph=graph,
        cross_section=CircularCrossSection(radius_mm=1.0),
        edge_cross_sections={
            "left_branch": VariableCircularCrossSection(
                radius_start_mm=1.0,
                radius_end_mm=3.0,
                length_mm=left_length,
            ),
        },
        profile=LinearRadialProfile(),
        map_name="scalar",
        role="target",
        label=1,
    )

    scene = graph_tube.render(grid)

    left_id = "target_graph__edge__left_branch"
    trunk_id = "target_graph__edge__trunk"

    assert scene.object_masks[left_id].sum() > scene.object_masks[trunk_id].sum()

    parameters = scene.truth.tables["graph_edge_parameters"]
    left_row = parameters.loc[parameters["edge_id"] == "left_branch"].iloc[0]
    assert left_row["cross_section_kind"] == "variable_circle_linear"

    geometric = scene.truth.geometric["target_graph"]
    edge_parameters = {row["edge_id"]: row for row in geometric["edge_parameters"]}
    assert edge_parameters["left_branch"]["cross_section_kind"] == (
        "variable_circle_linear"
    )
