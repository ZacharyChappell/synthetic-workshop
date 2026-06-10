"""Sampling graph edges as analytic centrelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from synthworkshop.primitives.curves import Centreline, LineCurve, PolylineCurve
from synthworkshop.topology.graphs import GraphEdge, GraphSpec


@dataclass(frozen=True)
class GraphCentrelineSet:
    """Sampled centrelines for all edges of a GraphSpec."""

    graph: GraphSpec
    centrelines: Mapping[str, Centreline]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        centrelines = dict(self.centrelines)
        expected = set(self.graph.edge_ids)
        observed = set(centrelines)

        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            raise ValueError(f"Missing centreline(s) for edge_id value(s): {missing}.")
        if extra:
            raise ValueError(f"Unknown centreline edge_id value(s): {extra}.")

        for edge_id, centreline in centrelines.items():
            if centreline.ndim != self.graph.ndim:
                raise ValueError(
                    f"Centreline for edge {edge_id!r} has ndim={centreline.ndim}, "
                    f"but graph ndim={self.graph.ndim}."
                )

        object.__setattr__(self, "centrelines", centrelines)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_edges(self) -> int:
        """Number of sampled graph edges."""

        return len(self.centrelines)

    @property
    def total_points(self) -> int:
        """Total sampled centreline points across all edges."""

        return int(sum(centreline.n_points for centreline in self.centrelines.values()))

    @property
    def total_length_mm(self) -> float:
        """Total sampled edge length across all centrelines."""

        return float(
            sum(centreline.length_mm for centreline in self.centrelines.values())
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return all graph-edge centrelines as one table."""

        frames: list[pd.DataFrame] = []
        edge_map = self.graph.edge_map

        for edge_id in self.graph.edge_ids:
            centreline = self.centrelines[edge_id]
            edge = edge_map[edge_id]
            table = centreline.to_dataframe().copy()
            table.insert(0, "graph_id", self.graph.graph_id)
            table.insert(1, "edge_id", edge_id)
            table.insert(2, "start_node", edge.start_node)
            table.insert(3, "end_node", edge.end_node)
            frames.append(table)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "graph_id",
                    "edge_id",
                    "start_node",
                    "end_node",
                    "point_index",
                    "parameter",
                    "arclength_mm",
                ]
            )

        return pd.concat(frames, axis=0, ignore_index=True)

    def edge_summary_table(self) -> pd.DataFrame:
        """Return one summary row per sampled edge centreline."""

        rows: list[dict[str, object]] = []
        edge_map = self.graph.edge_map

        for edge_id in self.graph.edge_ids:
            edge = edge_map[edge_id]
            centreline = self.centrelines[edge_id]
            rows.append(
                {
                    "graph_id": self.graph.graph_id,
                    "edge_id": edge_id,
                    "start_node": edge.start_node,
                    "end_node": edge.end_node,
                    "edge_kind": edge.kind,
                    "n_points": centreline.n_points,
                    "length_mm": centreline.length_mm,
                }
            )

        return pd.DataFrame(rows)

    def summary(self) -> dict[str, object]:
        """Return a compact summary."""

        return {
            "graph_id": self.graph.graph_id,
            "n_edges": self.n_edges,
            "total_points": self.total_points,
            "total_length_mm": self.total_length_mm,
            "graph": self.graph.summary(),
            "metadata": dict(self.metadata or {}),
        }


def _edge_vertices_from_metadata(edge: GraphEdge) -> list[list[float]] | None:
    """Return optional intermediate polyline vertices from edge metadata."""

    vertices = edge.metadata.get("vertices_mm")
    if vertices is None:
        return None
    if not isinstance(vertices, list | tuple):
        raise ValueError(
            f"Edge {edge.edge_id!r} metadata['vertices_mm'] must be a sequence."
        )
    return [list(vertex) for vertex in vertices]


def graph_edge_curve(graph: GraphSpec, edge_id: str):
    """Build a LineCurve or PolylineCurve for one graph edge."""

    edge_map = graph.edge_map
    if edge_id not in edge_map:
        raise KeyError(f"Unknown edge_id: {edge_id!r}.")

    edge = edge_map[edge_id]
    node_map = graph.node_map
    start = node_map[edge.start_node].coordinate_mm
    end = node_map[edge.end_node].coordinate_mm

    if edge.kind == "line":
        return LineCurve(start_mm=start, end_mm=end)

    if edge.kind == "polyline":
        intermediate = _edge_vertices_from_metadata(edge) or []
        return PolylineCurve(vertices_mm=[start, *intermediate, end])

    raise ValueError(
        f"Unsupported graph edge kind for centreline sampling: {edge.kind!r}."
    )


def sample_graph_edges(
    graph: GraphSpec,
    *,
    step_mm: float = 1.0,
    n_samples_per_edge: int | None = None,
    object_id: str | None = None,
) -> GraphCentrelineSet:
    """Sample every edge in a graph as an independent centreline."""

    centrelines: dict[str, Centreline] = {}
    for edge in graph.edges:
        curve = graph_edge_curve(graph, edge.edge_id)
        centrelines[edge.edge_id] = curve.sample(
            step_mm=step_mm,
            n_samples=n_samples_per_edge,
            object_id=object_id or graph.graph_id,
            segment_id=edge.edge_id,
        )

    return GraphCentrelineSet(
        graph=graph,
        centrelines=centrelines,
        metadata={
            "sampling": {
                "step_mm": step_mm,
                "n_samples_per_edge": n_samples_per_edge,
            }
        },
    )
