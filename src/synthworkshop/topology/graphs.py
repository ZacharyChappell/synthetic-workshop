"""Graph topology containers for branching and crossing scene definitions."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from synthworkshop.coordinates import FloatArray


def _validate_id(value: object, *, name: str) -> str:
    """Validate a non-empty string identifier."""

    text = str(value)
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _validate_coordinate(value: ArrayLike, *, name: str) -> FloatArray:
    """Validate a 2D or 3D physical coordinate."""

    coord = np.asarray(value, dtype=float)
    if coord.ndim != 1 or coord.shape[0] not in {2, 3}:
        raise ValueError(f"{name} must be a 2D or 3D coordinate.")
    if not np.all(np.isfinite(coord)):
        raise ValueError(f"{name} contains non-finite values.")
    return coord


@dataclass(frozen=True)
class GraphNode:
    """One node in a geometric topology graph."""

    node_id: str
    coordinate_mm: ArrayLike
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _validate_id(self.node_id, name="node_id"))
        object.__setattr__(
            self,
            "coordinate_mm",
            _validate_coordinate(self.coordinate_mm, name="coordinate_mm"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return int(np.asarray(self.coordinate_mm).shape[0])

    def to_record(self) -> dict[str, object]:
        """Return a table-ready node record."""

        coord = np.asarray(self.coordinate_mm, dtype=float)
        record: dict[str, object] = {
            "node_id": self.node_id,
            "label": self.label,
        }
        axis_names = ("i", "j") if self.ndim == 2 else ("i", "j", "k")
        for axis, axis_name in enumerate(axis_names):
            record[f"{axis_name}_mm"] = float(coord[axis])
        return record


@dataclass(frozen=True)
class GraphEdge:
    """One undirected edge in a geometric topology graph."""

    edge_id: str
    start_node: str
    end_node: str
    kind: str = "line"
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edge_id = _validate_id(self.edge_id, name="edge_id")
        start_node = _validate_id(self.start_node, name="start_node")
        end_node = _validate_id(self.end_node, name="end_node")
        kind = _validate_id(self.kind, name="kind")

        if start_node == end_node:
            raise ValueError("GraphEdge start_node and end_node must differ.")

        object.__setattr__(self, "edge_id", edge_id)
        object.__setattr__(self, "start_node", start_node)
        object.__setattr__(self, "end_node", end_node)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def node_pair(self) -> tuple[str, str]:
        """Return the directed node pair as stored."""

        return (self.start_node, self.end_node)

    @property
    def undirected_node_pair(self) -> tuple[str, str]:
        """Return a canonical undirected node pair."""

        return tuple(sorted((self.start_node, self.end_node)))

    def to_record(self) -> dict[str, object]:
        """Return a table-ready edge record."""

        return {
            "edge_id": self.edge_id,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "kind": self.kind,
            "label": self.label,
        }


@dataclass(frozen=True)
class GraphSpec:
    """Validated undirected geometric topology graph."""

    nodes: Sequence[GraphNode]
    edges: Sequence[GraphEdge]
    graph_id: str = "graph"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        graph_id = _validate_id(self.graph_id, name="graph_id")
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)

        if not nodes:
            raise ValueError("GraphSpec requires at least one node.")

        node_ids = [node.node_id for node in nodes]
        duplicate_nodes = sorted(
            node_id for node_id in set(node_ids) if node_ids.count(node_id) > 1
        )
        if duplicate_nodes:
            raise ValueError(f"Duplicate node_id value(s): {duplicate_nodes}.")

        edge_ids = [edge.edge_id for edge in edges]
        duplicate_edges = sorted(
            edge_id for edge_id in set(edge_ids) if edge_ids.count(edge_id) > 1
        )
        if duplicate_edges:
            raise ValueError(f"Duplicate edge_id value(s): {duplicate_edges}.")

        ndim = nodes[0].ndim
        if any(node.ndim != ndim for node in nodes):
            raise ValueError("All graph nodes must have the same dimensionality.")

        known_nodes = set(node_ids)
        for edge in edges:
            if edge.start_node not in known_nodes:
                raise ValueError(
                    f"Edge {edge.edge_id!r} references unknown start_node "
                    f"{edge.start_node!r}."
                )
            if edge.end_node not in known_nodes:
                raise ValueError(
                    f"Edge {edge.edge_id!r} references unknown end_node "
                    f"{edge.end_node!r}."
                )

        undirected_pairs = [edge.undirected_node_pair for edge in edges]
        duplicate_pairs = sorted(
            pair for pair in set(undirected_pairs) if undirected_pairs.count(pair) > 1
        )
        if duplicate_pairs:
            raise ValueError(f"Duplicate undirected edge pair(s): {duplicate_pairs}.")

        object.__setattr__(self, "graph_id", graph_id)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return self.nodes[0].ndim

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Node IDs in insertion order."""

        return tuple(node.node_id for node in self.nodes)

    @property
    def edge_ids(self) -> tuple[str, ...]:
        """Edge IDs in insertion order."""

        return tuple(edge.edge_id for edge in self.edges)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""

        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges."""

        return len(self.edges)

    @property
    def node_map(self) -> dict[str, GraphNode]:
        """Map node IDs to nodes."""

        return {node.node_id: node for node in self.nodes}

    @property
    def edge_map(self) -> dict[str, GraphEdge]:
        """Map edge IDs to edges."""

        return {edge.edge_id: edge for edge in self.edges}

    def node_coordinate_array(
        self,
        *,
        node_ids: Sequence[str] | None = None,
    ) -> FloatArray:
        """Return node coordinates in requested or insertion order."""

        mapping = self.node_map
        order = self.node_ids if node_ids is None else tuple(node_ids)
        coords = []
        for node_id in order:
            if node_id not in mapping:
                raise KeyError(f"Unknown node_id: {node_id!r}.")
            coords.append(mapping[node_id].coordinate_mm)
        return np.asarray(coords, dtype=float)

    def adjacency(self) -> dict[str, tuple[str, ...]]:
        """Return undirected adjacency in node insertion order."""

        neighbours: dict[str, list[str]] = {node_id: [] for node_id in self.node_ids}
        for edge in self.edges:
            neighbours[edge.start_node].append(edge.end_node)
            neighbours[edge.end_node].append(edge.start_node)
        return {node_id: tuple(values) for node_id, values in neighbours.items()}

    def degree(self, node_id: str) -> int:
        """Return undirected node degree."""

        node_id = _validate_id(node_id, name="node_id")
        adjacency = self.adjacency()
        if node_id not in adjacency:
            raise KeyError(f"Unknown node_id: {node_id!r}.")
        return len(adjacency[node_id])

    def degrees(self) -> dict[str, int]:
        """Return undirected degree for every node."""

        adjacency = self.adjacency()
        return {node_id: len(neighbours) for node_id, neighbours in adjacency.items()}

    @property
    def endpoints(self) -> tuple[str, ...]:
        """Nodes with degree 1."""

        return tuple(
            node_id for node_id, degree in self.degrees().items() if degree == 1
        )

    @property
    def junctions(self) -> tuple[str, ...]:
        """Nodes with degree at least 3."""

        return tuple(
            node_id for node_id, degree in self.degrees().items() if degree >= 3
        )

    @property
    def isolated_nodes(self) -> tuple[str, ...]:
        """Nodes with degree 0."""

        return tuple(
            node_id for node_id, degree in self.degrees().items() if degree == 0
        )

    def connected_components(self) -> tuple[tuple[str, ...], ...]:
        """Return undirected connected components."""

        adjacency = self.adjacency()
        remaining = set(self.node_ids)
        components: list[tuple[str, ...]] = []

        while remaining:
            start = next(node_id for node_id in self.node_ids if node_id in remaining)
            queue: deque[str] = deque([start])
            visited: set[str] = set()

            while queue:
                node_id = queue.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)
                remaining.discard(node_id)
                for neighbour in adjacency[node_id]:
                    if neighbour not in visited:
                        queue.append(neighbour)

            ordered = tuple(node_id for node_id in self.node_ids if node_id in visited)
            components.append(ordered)

        return tuple(components)

    @property
    def is_connected(self) -> bool:
        """Whether all nodes lie in one connected component."""

        return len(self.connected_components()) == 1

    def edge_length_mm(self, edge_id: str) -> float:
        """Return Euclidean edge length."""

        edge_id = _validate_id(edge_id, name="edge_id")
        edges = self.edge_map
        if edge_id not in edges:
            raise KeyError(f"Unknown edge_id: {edge_id!r}.")
        edge = edges[edge_id]
        nodes = self.node_map
        start = np.asarray(nodes[edge.start_node].coordinate_mm, dtype=float)
        end = np.asarray(nodes[edge.end_node].coordinate_mm, dtype=float)
        return float(np.linalg.norm(end - start))

    def edge_lengths_mm(self) -> dict[str, float]:
        """Return Euclidean length for every edge."""

        return {edge.edge_id: self.edge_length_mm(edge.edge_id) for edge in self.edges}

    def node_table(self) -> pd.DataFrame:
        """Return graph nodes as a table."""

        table = pd.DataFrame.from_records([node.to_record() for node in self.nodes])
        degree = self.degrees()
        table["degree"] = table["node_id"].map(degree).astype(int)
        table["is_endpoint"] = table["degree"] == 1
        table["is_junction"] = table["degree"] >= 3
        table["is_isolated"] = table["degree"] == 0
        return table

    def edge_table(self) -> pd.DataFrame:
        """Return graph edges as a table."""

        if not self.edges:
            return pd.DataFrame(
                columns=[
                    "edge_id",
                    "start_node",
                    "end_node",
                    "kind",
                    "label",
                    "length_mm",
                ]
            )
        table = pd.DataFrame.from_records([edge.to_record() for edge in self.edges])
        lengths = self.edge_lengths_mm()
        table["length_mm"] = table["edge_id"].map(lengths).astype(float)
        return table

    def component_table(self) -> pd.DataFrame:
        """Return connected components as a table."""

        rows: list[dict[str, object]] = []
        for component_index, component in enumerate(self.connected_components()):
            for node_id in component:
                rows.append(
                    {
                        "component_index": component_index,
                        "node_id": node_id,
                        "component_size": len(component),
                    }
                )
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, object]:
        """Return a compact graph summary."""

        degrees = self.degrees()
        return {
            "graph_id": self.graph_id,
            "ndim": self.ndim,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "is_connected": self.is_connected,
            "n_components": len(self.connected_components()),
            "endpoints": self.endpoints,
            "junctions": self.junctions,
            "isolated_nodes": self.isolated_nodes,
            "max_degree": max(degrees.values()) if degrees else 0,
            "metadata": dict(self.metadata),
        }


def graph_spec_from_dict(payload: Mapping[str, Any]) -> GraphSpec:
    """Build a GraphSpec from a dictionary-like payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("Graph payload must be a mapping/object.")

    node_payloads = payload.get("nodes")
    edge_payloads = payload.get("edges", [])

    if not isinstance(node_payloads, Sequence) or isinstance(node_payloads, str):
        raise ValueError("Graph payload 'nodes' must be a sequence/list.")
    if not isinstance(edge_payloads, Sequence) or isinstance(edge_payloads, str):
        raise ValueError("Graph payload 'edges' must be a sequence/list.")

    nodes = []
    for item in node_payloads:
        if not isinstance(item, Mapping):
            raise ValueError("Each graph node entry must be a mapping/object.")
        nodes.append(
            GraphNode(
                node_id=str(item["id"] if "id" in item else item["node_id"]),
                coordinate_mm=item["coordinate_mm"],
                label=item.get("label"),
                metadata=item.get("metadata", {}),
            )
        )

    edges = []
    for item in edge_payloads:
        if not isinstance(item, Mapping):
            raise ValueError("Each graph edge entry must be a mapping/object.")
        edges.append(
            GraphEdge(
                edge_id=str(item["id"] if "id" in item else item["edge_id"]),
                start_node=str(
                    item["start"] if "start" in item else item["start_node"]
                ),
                end_node=str(item["end"] if "end" in item else item["end_node"]),
                kind=str(item.get("kind", "line")),
                label=item.get("label"),
                metadata=item.get("metadata", {}),
            )
        )

    return GraphSpec(
        graph_id=str(payload.get("id", payload.get("graph_id", "graph"))),
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata=payload.get("metadata", {}),
    )
