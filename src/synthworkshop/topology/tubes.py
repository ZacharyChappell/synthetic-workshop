"""Edge-wise graph tube rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from synthworkshop.cross_sections import CrossSection
from synthworkshop.grid import GridSpec
from synthworkshop.ground_truth import SceneTruth
from synthworkshop.primitives import TubeObject
from synthworkshop.profiles import ScalarProfile
from synthworkshop.scenes import (
    CompositionRules,
    MaskRules,
    ObjectRole,
    OverlapPolicy,
    RenderedScene,
    compose_rendered_scenes,
)
from synthworkshop.topology.centrelines import GraphCentrelineSet, sample_graph_edges
from synthworkshop.topology.graphs import GraphEdge, GraphSpec


def _validate_object_id(value: object) -> str:
    """Validate a non-empty object identifier."""

    text = str(value)
    if not text:
        raise ValueError("object_id must be a non-empty string.")
    return text


def _validate_positive_int(value: int, *, name: str) -> int:
    """Validate a positive integer."""

    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive.")
    return out


def _validate_non_empty_string(value: object, *, name: str) -> str:
    """Validate a non-empty string."""

    text = str(value)
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _edge_object_id(graph_object_id: str, edge_id: str) -> str:
    """Return the rendered object ID for one graph edge."""

    return f"{graph_object_id}__edge__{edge_id}"


def _validate_override_keys(
    graph: GraphSpec,
    mapping: Mapping[str, object],
    *,
    name: str,
) -> None:
    """Validate that all override keys refer to graph edges."""

    unknown = sorted(set(mapping) - set(graph.edge_ids))
    if unknown:
        raise ValueError(f"{name} contains unknown edge_id value(s): {unknown}.")


def _coerce_edge_roles(
    graph: GraphSpec,
    edge_roles: Mapping[str, ObjectRole | str],
) -> dict[str, ObjectRole]:
    """Validate and coerce edge role overrides."""

    _validate_override_keys(graph, edge_roles, name="edge_roles")
    return {str(edge_id): ObjectRole(role) for edge_id, role in edge_roles.items()}


def _coerce_edge_labels(
    graph: GraphSpec,
    edge_labels: Mapping[str, int],
) -> dict[str, int]:
    """Validate and coerce edge label overrides."""

    _validate_override_keys(graph, edge_labels, name="edge_labels")
    return {
        str(edge_id): _validate_positive_int(label, name="edge label")
        for edge_id, label in edge_labels.items()
    }


def _coerce_edge_priorities(
    graph: GraphSpec,
    edge_priorities: Mapping[str, int],
) -> dict[str, int]:
    """Validate and coerce edge priority overrides."""

    _validate_override_keys(graph, edge_priorities, name="edge_priorities")
    return {
        str(edge_id): int(priority) for edge_id, priority in edge_priorities.items()
    }


def _coerce_edge_map_names(
    graph: GraphSpec,
    edge_map_names: Mapping[str, str],
) -> dict[str, str]:
    """Validate and coerce edge scalar-map-name overrides."""

    _validate_override_keys(graph, edge_map_names, name="edge_map_names")
    return {
        str(edge_id): _validate_non_empty_string(map_name, name="edge map name")
        for edge_id, map_name in edge_map_names.items()
    }


def _node_kind_from_degree(degree: int) -> str:
    """Classify a graph node by undirected degree."""

    if degree == 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree == 2:
        return "internal"
    return "junction"


@dataclass(frozen=True)
class GraphTubeObject:
    """Render each graph edge as a separate tube contribution.

    This renderer intentionally treats graph edges as separate rendered objects,
    then composes them. This preserves branch/edge masks and makes graph-junction
    overlap explicit rather than silently hiding it inside a bespoke union
    operation.

    M2c-b added optional per-edge overrides for cross-section, profile, role,
    label, priority, and scalar map name.

    M2c-c adds graph-level truth aggregation: parent graph skeletons, graph-level
    centreline/frame tables, endpoint tables, junction tables, and incident-edge
    metadata. It still does not implement smooth junction-union geometry.
    """

    object_id: str
    graph: GraphSpec
    cross_section: CrossSection
    profile: ScalarProfile
    map_name: str = "scalar"
    role: ObjectRole | str = ObjectRole.TARGET
    label: int = 1
    priority: int = 0
    step_mm: float = 1.0
    n_samples_per_edge: int | None = None
    edge_cross_sections: Mapping[str, CrossSection] = field(default_factory=dict)
    edge_profiles: Mapping[str, ScalarProfile] = field(default_factory=dict)
    edge_labels: Mapping[str, int] = field(default_factory=dict)
    edge_priorities: Mapping[str, int] = field(default_factory=dict)
    edge_roles: Mapping[str, ObjectRole | str] = field(default_factory=dict)
    edge_map_names: Mapping[str, str] = field(default_factory=dict)
    name: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object_id = _validate_object_id(self.object_id)
        map_name = _validate_non_empty_string(self.map_name, name="map_name")
        if self.graph.ndim != 3:
            raise ValueError("GraphTubeObject currently requires a 3D graph.")

        label = _validate_positive_int(self.label, name="label")
        priority = int(self.priority)
        role = ObjectRole(self.role)

        step_mm = float(self.step_mm)
        if not np.isfinite(step_mm) or step_mm <= 0:
            raise ValueError("step_mm must be finite and positive.")

        n_samples = self.n_samples_per_edge
        if n_samples is not None and int(n_samples) < 2:
            raise ValueError("n_samples_per_edge must be at least 2 when provided.")

        edge_cross_sections = dict(self.edge_cross_sections)
        edge_profiles = dict(self.edge_profiles)
        _validate_override_keys(
            self.graph,
            edge_cross_sections,
            name="edge_cross_sections",
        )
        _validate_override_keys(self.graph, edge_profiles, name="edge_profiles")

        edge_labels = _coerce_edge_labels(self.graph, self.edge_labels)
        edge_priorities = _coerce_edge_priorities(self.graph, self.edge_priorities)
        edge_roles = _coerce_edge_roles(self.graph, self.edge_roles)
        edge_map_names = _coerce_edge_map_names(self.graph, self.edge_map_names)

        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "map_name", map_name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "step_mm", step_mm)
        object.__setattr__(
            self,
            "n_samples_per_edge",
            None if n_samples is None else int(n_samples),
        )
        object.__setattr__(self, "edge_cross_sections", edge_cross_sections)
        object.__setattr__(self, "edge_profiles", edge_profiles)
        object.__setattr__(self, "edge_labels", edge_labels)
        object.__setattr__(self, "edge_priorities", edge_priorities)
        object.__setattr__(self, "edge_roles", edge_roles)
        object.__setattr__(self, "edge_map_names", edge_map_names)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def edge_object_id(self, edge_id: str) -> str:
        """Return the rendered object ID used for one graph edge."""

        return _edge_object_id(self.object_id, edge_id)

    def edge_cross_section(self, edge_id: str) -> CrossSection:
        """Return the cross-section for one edge."""

        return self.edge_cross_sections.get(edge_id, self.cross_section)

    def edge_profile(self, edge_id: str) -> ScalarProfile:
        """Return the scalar profile for one edge."""

        return self.edge_profiles.get(edge_id, self.profile)

    def edge_label(self, edge_id: str) -> int:
        """Return the integer label for one edge."""

        return self.edge_labels.get(edge_id, self.label)

    def edge_priority(self, edge_id: str) -> int:
        """Return the label priority for one edge."""

        return self.edge_priorities.get(edge_id, self.priority)

    def edge_role(self, edge_id: str) -> ObjectRole:
        """Return the object role for one edge."""

        return self.edge_roles.get(edge_id, self.role)

    def edge_map_name(self, edge_id: str) -> str:
        """Return the scalar map name for one edge."""

        return self.edge_map_names.get(edge_id, self.map_name)

    def sample_edges(self) -> GraphCentrelineSet:
        """Sample graph edges as independent centrelines."""

        return sample_graph_edges(
            self.graph,
            step_mm=self.step_mm,
            n_samples_per_edge=self.n_samples_per_edge,
            object_id=self.object_id,
        )

    def edge_tube_object(
        self,
        edge: GraphEdge,
        centreline_set: GraphCentrelineSet,
    ) -> TubeObject:
        """Create a TubeObject for one sampled graph edge."""

        edge_object_id = self.edge_object_id(edge.edge_id)
        cross_section = self.edge_cross_section(edge.edge_id)
        profile = self.edge_profile(edge.edge_id)
        role = self.edge_role(edge.edge_id)
        label = self.edge_label(edge.edge_id)
        priority = self.edge_priority(edge.edge_id)
        map_name = self.edge_map_name(edge.edge_id)

        return TubeObject(
            object_id=edge_object_id,
            centreline=centreline_set.centrelines[edge.edge_id],
            cross_section=cross_section,
            profile=profile,
            map_name=map_name,
            role=role,
            label=label,
            priority=priority,
            name=f"{self.name or self.object_id}: {edge.edge_id}",
            description=self.description,
            metadata={
                **dict(self.metadata or {}),
                "graph_object_id": self.object_id,
                "graph_id": self.graph.graph_id,
                "edge_id": edge.edge_id,
                "start_node": edge.start_node,
                "end_node": edge.end_node,
                "edge_kind": edge.kind,
                "edge_cross_section": cross_section.summary(),
                "edge_profile": profile.summary(),
                "edge_role": role.value,
                "edge_label": label,
                "edge_priority": priority,
                "edge_map_name": map_name,
            },
        )

    def edge_tube_objects(
        self,
        centreline_set: GraphCentrelineSet | None = None,
    ) -> tuple[TubeObject, ...]:
        """Create TubeObject instances for all graph edges."""

        sampled = self.sample_edges() if centreline_set is None else centreline_set
        return tuple(self.edge_tube_object(edge, sampled) for edge in self.graph.edges)

    def edge_parameter_table(self) -> pd.DataFrame:
        """Return one row per graph edge describing rendering parameters."""

        rows: list[dict[str, object]] = []
        for edge in self.graph.edges:
            cross_section = self.edge_cross_section(edge.edge_id)
            profile = self.edge_profile(edge.edge_id)
            rows.append(
                {
                    "graph_id": self.graph.graph_id,
                    "graph_object_id": self.object_id,
                    "edge_id": edge.edge_id,
                    "edge_object_id": self.edge_object_id(edge.edge_id),
                    "start_node": edge.start_node,
                    "end_node": edge.end_node,
                    "edge_kind": edge.kind,
                    "role": self.edge_role(edge.edge_id).value,
                    "label": self.edge_label(edge.edge_id),
                    "priority": self.edge_priority(edge.edge_id),
                    "map_name": self.edge_map_name(edge.edge_id),
                    "cross_section_kind": cross_section.kind,
                    "profile_kind": profile.kind,
                }
            )
        return pd.DataFrame(rows)

    def graph_node_topology_table(self) -> pd.DataFrame:
        """Return graph nodes with endpoint/junction classifications."""

        nodes = self.graph.node_table().copy()
        edge_lookup = {node_id: [] for node_id in self.graph.node_ids}
        edge_object_lookup = {node_id: [] for node_id in self.graph.node_ids}

        for edge in self.graph.edges:
            edge_lookup[edge.start_node].append(edge.edge_id)
            edge_lookup[edge.end_node].append(edge.edge_id)
            edge_object_lookup[edge.start_node].append(
                self.edge_object_id(edge.edge_id)
            )
            edge_object_lookup[edge.end_node].append(self.edge_object_id(edge.edge_id))

        nodes["node_kind"] = nodes["degree"].map(_node_kind_from_degree)
        nodes["incident_edges"] = nodes["node_id"].map(
            lambda node_id: "|".join(edge_lookup[node_id])
        )
        nodes["incident_edge_object_ids"] = nodes["node_id"].map(
            lambda node_id: "|".join(edge_object_lookup[node_id])
        )
        return nodes

    def graph_endpoint_table(self) -> pd.DataFrame:
        """Return graph endpoint nodes."""

        table = self.graph_node_topology_table()
        return table.loc[table["node_kind"] == "endpoint"].reset_index(drop=True)

    def graph_junction_table(self) -> pd.DataFrame:
        """Return graph junction nodes."""

        table = self.graph_node_topology_table()
        return table.loc[table["node_kind"] == "junction"].reset_index(drop=True)

    def graph_incident_edge_table(self) -> pd.DataFrame:
        """Return one row per node-edge incidence."""

        rows: list[dict[str, object]] = []
        degrees = self.graph.degrees()
        for edge in self.graph.edges:
            for node_id, neighbour_id in (
                (edge.start_node, edge.end_node),
                (edge.end_node, edge.start_node),
            ):
                rows.append(
                    {
                        "graph_id": self.graph.graph_id,
                        "graph_object_id": self.object_id,
                        "node_id": node_id,
                        "node_kind": _node_kind_from_degree(degrees[node_id]),
                        "degree": degrees[node_id],
                        "edge_id": edge.edge_id,
                        "edge_object_id": self.edge_object_id(edge.edge_id),
                        "neighbour_node_id": neighbour_id,
                        "edge_role": self.edge_role(edge.edge_id).value,
                        "edge_label": self.edge_label(edge.edge_id),
                        "edge_priority": self.edge_priority(edge.edge_id),
                        "edge_map_name": self.edge_map_name(edge.edge_id),
                    }
                )
        return pd.DataFrame(rows)

    def _graph_truth_tables(
        self,
        centreline_set: GraphCentrelineSet,
        graph_frame_table: pd.DataFrame,
    ) -> dict[str, object]:
        """Return graph-specific truth tables."""

        return {
            "graph_nodes": self.graph.node_table(),
            "graph_node_topology": self.graph_node_topology_table(),
            "graph_endpoints": self.graph_endpoint_table(),
            "graph_junctions": self.graph_junction_table(),
            "graph_incident_edges": self.graph_incident_edge_table(),
            "graph_edges": self.graph.edge_table(),
            "graph_components": self.graph.component_table(),
            "graph_centrelines": centreline_set.to_dataframe(),
            "graph_frames": graph_frame_table,
            "graph_edge_summaries": centreline_set.edge_summary_table(),
            "graph_edge_parameters": self.edge_parameter_table(),
        }

    def _edge_object_ids(self) -> dict[str, str]:
        """Return edge-to-rendered-object IDs."""

        return {
            edge.edge_id: self.edge_object_id(edge.edge_id) for edge in self.graph.edges
        }

    def _edge_parameter_records(self) -> list[dict[str, object]]:
        """Return edge parameters as serialisable records."""

        return self.edge_parameter_table().to_dict(orient="records")

    def _graph_skeleton_mask(self, composed: RenderedScene) -> np.ndarray:
        """Return the union of all rendered edge skeleton masks."""

        skeleton = np.zeros(composed.grid.shape, dtype=bool)
        for edge in self.graph.edges:
            edge_object_id = self.edge_object_id(edge.edge_id)
            skeleton |= composed.skeleton_masks[edge_object_id]
        return skeleton

    def _graph_frame_table(self, composed: RenderedScene) -> pd.DataFrame:
        """Return all edge frame tables with graph/edge metadata."""

        frames: list[pd.DataFrame] = []
        for edge in self.graph.edges:
            edge_object_id = self.edge_object_id(edge.edge_id)
            table = composed.frames[edge_object_id].copy()
            table.insert(0, "graph_id", self.graph.graph_id)
            table.insert(1, "graph_object_id", self.object_id)
            table.insert(2, "edge_id", edge.edge_id)
            table.insert(3, "edge_object_id", edge_object_id)
            table.insert(4, "start_node", edge.start_node)
            table.insert(5, "end_node", edge.end_node)
            frames.append(table)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "graph_id",
                    "graph_object_id",
                    "edge_id",
                    "edge_object_id",
                    "start_node",
                    "end_node",
                ]
            )
        return pd.concat(frames, axis=0, ignore_index=True)

    def _enhance_truth(
        self,
        composed: RenderedScene,
        centreline_set: GraphCentrelineSet,
        graph_frame_table: pd.DataFrame,
        graph_skeleton_mask: np.ndarray,
    ) -> SceneTruth:
        """Add graph-level truth metadata/tables to a composed edge scene."""

        tables = {
            **dict(composed.truth.tables),
            **self._graph_truth_tables(centreline_set, graph_frame_table),
        }
        edge_object_ids = self._edge_object_ids()
        edge_parameters = self._edge_parameter_records()
        junction_ids = self.graph.junctions
        endpoint_ids = self.graph.endpoints

        geometric = {
            **dict(composed.truth.geometric),
            self.object_id: {
                "kind": "graph_tube",
                "rendering": "edge_wise_composed_tubes",
                "graph_id": self.graph.graph_id,
                "graph_summary": self.graph.summary(),
                "edge_object_ids": edge_object_ids,
                "edge_parameters": edge_parameters,
                "junction_node_ids": junction_ids,
                "endpoint_node_ids": endpoint_ids,
                "graph_skeleton_key": self.object_id,
                "graph_skeleton_voxels": int(np.sum(graph_skeleton_mask)),
                "default_cross_section": self.cross_section.summary(),
                "default_profile": self.profile.summary(),
            },
        }
        objects = {
            **dict(composed.truth.objects),
            self.object_id: {
                "kind": "graph_tube_parent",
                "default_role": self.role.value,
                "default_label": self.label,
                "default_priority": self.priority,
                "edge_object_ids": edge_object_ids,
                "edge_parameters": edge_parameters,
                "n_edges": self.graph.n_edges,
                "n_nodes": self.graph.n_nodes,
                "junction_node_ids": junction_ids,
                "endpoint_node_ids": endpoint_ids,
                "graph_skeleton_voxels": int(np.sum(graph_skeleton_mask)),
            },
        }

        scalar_fields = dict(composed.truth.scalar_fields)
        map_names = sorted(
            {self.edge_map_name(edge.edge_id) for edge in self.graph.edges}
        )
        for map_name in map_names:
            scalar_fields[f"{self.object_id}:{map_name}"] = {
                "object_id": self.object_id,
                "map_name": map_name,
                "edge_object_ids": {
                    edge.edge_id: self.edge_object_id(edge.edge_id)
                    for edge in self.graph.edges
                    if self.edge_map_name(edge.edge_id) == map_name
                },
                "edge_parameters": [
                    record
                    for record in edge_parameters
                    if record["map_name"] == map_name
                ],
            }

        return SceneTruth(
            geometric=geometric,
            objects=objects,
            scalar_fields=scalar_fields,
            perturbations=dict(composed.truth.perturbations),
            tables=tables,
            metadata={
                **dict(composed.truth.metadata),
                "graph_tube_scope": (
                    "Edge-wise graph tube truth: graph topology, edge masks, "
                    "sampled edge centrelines, graph-level centreline/frame tables, "
                    "parent graph skeleton mask, branch-specific rendering "
                    "parameters, endpoint/junction tables, incident-edge metadata, "
                    "and explicit edge-overlap reporting. Smooth junction-union "
                    "geometry is not yet implemented."
                ),
            },
        )

    def render(
        self,
        grid: GridSpec,
        *,
        chunk_size: int = 65536,
        overlap_policy: OverlapPolicy | str = OverlapPolicy.ALLOW,
        label_mode: str = "priority",
        scalar_blend: str = "overwrite",
    ) -> RenderedScene:
        """Render graph edges as tubes and compose them into one scene."""

        if grid.ndim != 3:
            raise ValueError("GraphTubeObject rendering currently requires a 3D grid.")

        centreline_set = self.sample_edges()
        edge_scenes = [
            edge_tube.render(
                grid,
                chunk_size=chunk_size,
                overlap_policy=OverlapPolicy.ALLOW,
            )
            for edge_tube in self.edge_tube_objects(centreline_set)
        ]

        composed = compose_rendered_scenes(
            edge_scenes,
            composition=CompositionRules(
                label_mode=label_mode,
                scalar_blend=scalar_blend,
                overlap_policy=overlap_policy,
            ),
            mask_rules=MaskRules(),
            metadata={
                "renderer": "GraphTubeObject.render",
                "graph_object_id": self.object_id,
                "graph_id": self.graph.graph_id,
            },
            provenance={
                "package": "synthworkshop",
                "stage": "M2c-c",
            },
        )

        graph_skeleton_mask = self._graph_skeleton_mask(composed)
        graph_frame_table = self._graph_frame_table(composed)
        truth = self._enhance_truth(
            composed,
            centreline_set,
            graph_frame_table,
            graph_skeleton_mask,
        )

        skeleton_masks = {
            **dict(composed.skeleton_masks),
            self.object_id: graph_skeleton_mask,
        }
        centrelines = {
            **dict(composed.centrelines),
            self.object_id: centreline_set.to_dataframe(),
        }
        frames = {
            **dict(composed.frames),
            self.object_id: graph_frame_table,
        }

        return RenderedScene(
            grid=composed.grid,
            scalar_maps=composed.scalar_maps,
            label_map=composed.label_map,
            object_masks=composed.object_masks,
            object_metadata=composed.object_metadata,
            truth=truth,
            composition=composed.composition,
            mask_rules=composed.mask_rules,
            target_masks=composed.target_masks,
            analysis_masks=composed.analysis_masks,
            skeleton_masks=skeleton_masks,
            centrelines=centrelines,
            frames=frames,
            distance_maps=composed.distance_maps,
            signed_offset_maps=composed.signed_offset_maps,
            overlap_report=composed.overlap_report,
            metadata={
                **dict(composed.metadata),
                "graph_object_id": self.object_id,
                "graph_id": self.graph.graph_id,
                "edge_object_ids": self._edge_object_ids(),
                "edge_parameters": self._edge_parameter_records(),
                "junction_node_ids": self.graph.junctions,
                "endpoint_node_ids": self.graph.endpoints,
                "graph_skeleton_key": self.object_id,
            },
            provenance=composed.provenance,
        )
