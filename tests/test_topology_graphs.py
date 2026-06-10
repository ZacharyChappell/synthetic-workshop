from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.topology import GraphEdge, GraphNode, GraphSpec, graph_spec_from_dict


def _y_graph() -> GraphSpec:
    return GraphSpec(
        graph_id="y_graph",
        nodes=(
            GraphNode("root", (0.0, 0.0, 0.0)),
            GraphNode("junction", (1.0, 0.0, 0.0)),
            GraphNode("left", (2.0, -1.0, 0.0)),
            GraphNode("right", (2.0, 1.0, 0.0)),
            GraphNode("top", (2.0, 0.0, 1.0)),
        ),
        edges=(
            GraphEdge("trunk", "root", "junction"),
            GraphEdge("left_branch", "junction", "left"),
            GraphEdge("right_branch", "junction", "right"),
            GraphEdge("top_branch", "junction", "top"),
        ),
    )


def test_graph_node_validates_coordinate() -> None:
    node = GraphNode("a", (1.0, 2.0, 3.0), label="A")

    assert node.node_id == "a"
    assert node.ndim == 3
    assert node.to_record()["k_mm"] == 3.0

    with pytest.raises(ValueError, match="2D or 3D"):
        GraphNode("bad", (1.0,))


def test_graph_edge_rejects_self_loop() -> None:
    with pytest.raises(ValueError, match="must differ"):
        GraphEdge("loop", "a", "a")


def test_graph_spec_reports_degrees_endpoints_and_junctions() -> None:
    graph = _y_graph()

    assert graph.n_nodes == 5
    assert graph.n_edges == 4
    assert graph.degree("junction") == 4
    assert set(graph.endpoints) == {"root", "left", "right", "top"}
    assert graph.junctions == ("junction",)
    assert graph.isolated_nodes == ()


def test_graph_adjacency_preserves_node_order() -> None:
    graph = _y_graph()
    adjacency = graph.adjacency()

    assert adjacency["root"] == ("junction",)
    assert adjacency["junction"] == ("root", "left", "right", "top")


def test_graph_connected_components_and_connected_flag() -> None:
    graph = _y_graph()

    assert graph.is_connected
    assert graph.connected_components() == (
        ("root", "junction", "left", "right", "top"),
    )


def test_disconnected_graph_reports_multiple_components() -> None:
    graph = GraphSpec(
        nodes=(
            GraphNode("a", (0.0, 0.0, 0.0)),
            GraphNode("b", (1.0, 0.0, 0.0)),
            GraphNode("c", (5.0, 0.0, 0.0)),
        ),
        edges=(GraphEdge("ab", "a", "b"),),
    )

    assert not graph.is_connected
    assert graph.isolated_nodes == ("c",)
    assert graph.connected_components() == (("a", "b"), ("c",))


def test_graph_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate node_id"):
        GraphSpec(
            nodes=(
                GraphNode("a", (0.0, 0.0, 0.0)),
                GraphNode("a", (1.0, 0.0, 0.0)),
            ),
            edges=(),
        )


def test_graph_rejects_duplicate_edge_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate edge_id"):
        GraphSpec(
            nodes=(
                GraphNode("a", (0.0, 0.0, 0.0)),
                GraphNode("b", (1.0, 0.0, 0.0)),
                GraphNode("c", (2.0, 0.0, 0.0)),
            ),
            edges=(
                GraphEdge("e", "a", "b"),
                GraphEdge("e", "b", "c"),
            ),
        )


def test_graph_rejects_unknown_edge_nodes() -> None:
    with pytest.raises(ValueError, match="unknown end_node"):
        GraphSpec(
            nodes=(GraphNode("a", (0.0, 0.0, 0.0)),),
            edges=(GraphEdge("bad", "a", "missing"),),
        )


def test_graph_rejects_duplicate_undirected_pairs() -> None:
    with pytest.raises(ValueError, match="Duplicate undirected edge pair"):
        GraphSpec(
            nodes=(
                GraphNode("a", (0.0, 0.0, 0.0)),
                GraphNode("b", (1.0, 0.0, 0.0)),
            ),
            edges=(
                GraphEdge("ab", "a", "b"),
                GraphEdge("ba", "b", "a"),
            ),
        )


def test_graph_rejects_mixed_dimensional_nodes() -> None:
    with pytest.raises(ValueError, match="same dimensionality"):
        GraphSpec(
            nodes=(
                GraphNode("a", (0.0, 0.0, 0.0)),
                GraphNode("b", (1.0, 0.0)),
            ),
            edges=(),
        )


def test_edge_lengths_and_coordinate_array() -> None:
    graph = GraphSpec(
        nodes=(
            GraphNode("a", (0.0, 0.0, 0.0)),
            GraphNode("b", (3.0, 4.0, 0.0)),
        ),
        edges=(GraphEdge("ab", "a", "b"),),
    )

    assert np.isclose(graph.edge_length_mm("ab"), 5.0)
    assert np.allclose(graph.node_coordinate_array(node_ids=("b",)), [[3.0, 4.0, 0.0]])


def test_node_edge_and_component_tables() -> None:
    graph = _y_graph()
    nodes = graph.node_table()
    edges = graph.edge_table()
    components = graph.component_table()

    assert {"node_id", "degree", "is_endpoint", "is_junction"}.issubset(nodes.columns)
    assert {"edge_id", "start_node", "end_node", "length_mm"}.issubset(edges.columns)
    assert {"component_index", "node_id", "component_size"}.issubset(components.columns)
    assert nodes.loc[nodes["node_id"] == "junction", "is_junction"].item()


def test_graph_summary_contains_topology_counts() -> None:
    graph = _y_graph()
    summary = graph.summary()

    assert summary["graph_id"] == "y_graph"
    assert summary["n_nodes"] == 5
    assert summary["n_edges"] == 4
    assert summary["junctions"] == ("junction",)
    assert summary["max_degree"] == 4


def test_graph_spec_from_dict_accepts_id_aliases() -> None:
    graph = graph_spec_from_dict(
        {
            "id": "dict_graph",
            "nodes": [
                {"id": "a", "coordinate_mm": [0.0, 0.0, 0.0]},
                {"id": "b", "coordinate_mm": [1.0, 0.0, 0.0]},
            ],
            "edges": [
                {"id": "ab", "start": "a", "end": "b"},
            ],
        }
    )

    assert graph.graph_id == "dict_graph"
    assert graph.node_ids == ("a", "b")
    assert graph.edge_ids == ("ab",)
    assert graph.is_connected


def test_graph_spec_from_dict_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="mapping"):
        graph_spec_from_dict([])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="nodes"):
        graph_spec_from_dict({"nodes": "not-a-list"})


def test_top_level_exports_graph_types() -> None:
    import synthworkshop

    assert synthworkshop.GraphNode is GraphNode
    assert synthworkshop.GraphEdge is GraphEdge
    assert synthworkshop.GraphSpec is GraphSpec
    assert synthworkshop.graph_spec_from_dict is graph_spec_from_dict
