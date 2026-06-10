"""Method-agnostic scene-truth containers."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SceneTruth:
    """Mathematical and rendering truth owned by the workshop.

    This class deliberately avoids downstream method-specific estimands. For
    example, a target centreline, object mask, distance map, and scalar field are
    scene truth. A downstream method's chosen edge detector or feature estimator
    is not universal scene truth and should live in a separate evaluation layer.
    """

    geometric: Mapping[str, Any] = field(default_factory=dict)
    objects: Mapping[str, Any] = field(default_factory=dict)
    scalar_fields: Mapping[str, Any] = field(default_factory=dict)
    perturbations: Mapping[str, Any] = field(default_factory=dict)
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def table_names(self) -> tuple[str, ...]:
        """Return available truth-table names."""

        return tuple(self.tables.keys())

    def summary(self) -> dict[str, object]:
        """Return a compact truth summary."""

        return {
            "n_geometric_entries": len(self.geometric),
            "n_object_entries": len(self.objects),
            "n_scalar_field_entries": len(self.scalar_fields),
            "n_perturbation_entries": len(self.perturbations),
            "truth_tables": self.table_names(),
            "metadata_keys": tuple(self.metadata.keys()),
        }
