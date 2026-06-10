from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.primitives import LineCurve, PolylineCurve
from synthworkshop.topology import (
    GraphCentrelineSet,
    GraphEdge,
    GraphNode,
    GraphSpec,
    graph_edge_curve,
    sample_graph_edges,
)


def _graph_with_polyline_edge() -> GraphSpec:
    return GraphSpec(
        graph_id="graph",
        nodes=(
            GraphNode("a", (0.0, 0.0, 0.0)),
            GraphNode("b", (4.0, 0.0, 0.0)),
            GraphNode("c", (4.0, 4.0, 0.0)),
        ),
        edges=(
            GraphEdge("ab", "a", "b"),
            GraphEdge(
                "bc",
                "b",
                "c",
                kind="polyline",
                metadata={"vertices_mm": [[5.0, 1.0, 0.0], [5.0, 3.0, 0.0]]},
            ),
        ),
    )


def test_graph_edge_curve_returns_line_for_line_edge() -> None:
    graph = _graph_with_polyline_edge()
    curve = graph_edge_curve(graph, "ab")

    assert isinstance(curve, LineCurve)
    assert np.isclose(curve.length_mm, 4.0)


def test_graph_edge_curve_returns_polyline_for_polyline_edge() -> None:
    graph = _graph_with_polyline_edge()
    curve = graph_edge_curve(graph, "bc")

    assert isinstance(curve, PolylineCurve)
    assert curve.n_vertices == 4
    assert curve.length_mm > 4.0


def test_graph_edge_curve_rejects_unknown_edge_kind() -> None:
    graph = GraphSpec(
        nodes=(
            GraphNode("a", (0.0, 0.0, 0.0)),
            GraphNode("b", (1.0, 0.0, 0.0)),
        ),
        edges=(GraphEdge("ab", "a", "b", kind="bezier"),),
    )

    with pytest.raises(ValueError, match="Unsupported graph edge kind"):
        graph_edge_curve(graph, "ab")


def test_sample_graph_edges_returns_centreline_set() -> None:
    graph = _graph_with_polyline_edge()
    sampled = sample_graph_edges(graph, step_mm=1.0)

    assert isinstance(sampled, GraphCentrelineSet)
    assert sampled.n_edges == 2
    assert set(sampled.centrelines) == {"ab", "bc"}
    assert sampled.total_points > 0
    assert sampled.total_length_mm > graph.edge_length_mm("ab")


def test_graph_centreline_set_to_dataframe_contains_edge_metadata() -> None:
    graph = _graph_with_polyline_edge()
    sampled = sample_graph_edges(graph, n_samples_per_edge=3, object_id="object")

    table = sampled.to_dataframe()

    assert {"graph_id", "edge_id", "start_node", "end_node"}.issubset(table.columns)
    assert set(table["edge_id"]) == {"ab", "bc"}
    assert table.shape[0] == 6
    assert set(table["object_id"]) == {"object"}
    assert set(table["segment_id"]) == {"ab", "bc"}


def test_graph_centreline_edge_summary_table() -> None:
    graph = _graph_with_polyline_edge()
    sampled = sample_graph_edges(graph, n_samples_per_edge=4)
    table = sampled.edge_summary_table()

    assert table.shape[0] == 2
    assert {"edge_id", "n_points", "length_mm"}.issubset(table.columns)
    assert set(table["n_points"]) == {4}


def test_graph_centreline_set_rejects_missing_edge() -> None:
    graph = _graph_with_polyline_edge()
    centreline = graph_edge_curve(graph, "ab").sample(n_samples=3)

    with pytest.raises(ValueError, match="Missing centreline"):
        GraphCentrelineSet(graph=graph, centrelines={"ab": centreline})


def test_graph_centreline_set_rejects_unknown_edge_key() -> None:
    graph = GraphSpec(
        nodes=(
            GraphNode("a", (0.0, 0.0, 0.0)),
            GraphNode("b", (1.0, 0.0, 0.0)),
        ),
        edges=(GraphEdge("ab", "a", "b"),),
    )
    centreline = graph_edge_curve(graph, "ab").sample(n_samples=3)

    with pytest.raises(ValueError, match="Unknown centreline"):
        GraphCentrelineSet(
            graph=graph,
            centrelines={
                "ab": centreline,
                "extra": centreline,
            },
        )


def test_graph_centreline_summary_contains_graph_summary() -> None:
    graph = _graph_with_polyline_edge()
    sampled = sample_graph_edges(graph, n_samples_per_edge=3)
    summary = sampled.summary()

    assert summary["graph_id"] == "graph"
    assert summary["n_edges"] == 2
    assert summary["graph"]["n_nodes"] == 3


def test_top_level_exports_graph_centreline_helpers() -> None:
    import synthworkshop

    assert synthworkshop.GraphCentrelineSet is GraphCentrelineSet
    assert synthworkshop.graph_edge_curve is graph_edge_curve
    assert synthworkshop.sample_graph_edges is sample_graph_edges
