"""Shared helpers for scene perturbations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from synthworkshop.ground_truth import SceneTruth
from synthworkshop.scenes import RenderedScene


@dataclass(frozen=True)
class PerturbationRecord:
    """Metadata for one controlled perturbation."""

    name: str
    target: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    truth_changed: bool = False
    observed_changed: bool = True
    affected_arrays: tuple[str, ...] = ()
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""

        payload: dict[str, Any] = {
            "name": self.name,
            "target": self.target,
            "parameters": _json_safe(dict(self.parameters)),
            "seed": self.seed,
            "truth_changed": self.truth_changed,
            "observed_changed": self.observed_changed,
            "affected_arrays": list(self.affected_arrays),
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


def selected_names(
    mapping: Mapping[str, Any],
    names: Sequence[str] | None,
    *,
    label: str,
) -> tuple[str, ...]:
    """Return selected mapping names with validation."""

    if names is None:
        return tuple(mapping.keys())

    selected = tuple(str(name) for name in names)
    missing = sorted(set(selected) - set(mapping))
    if missing:
        raise ValueError(f"Unknown {label} name(s): {missing}.")
    return selected


def validate_non_negative_finite(value: float, *, name: str) -> float:
    """Return a finite non-negative float."""

    out = float(value)
    if not np.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return out


def validate_positive_int(value: int, *, name: str) -> int:
    """Return a positive integer."""

    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive.")
    return out


def validate_shift(
    shift_voxels: Sequence[int],
    *,
    ndim: int,
    name: str = "shift_voxels",
) -> tuple[int, ...]:
    """Return an integer voxel shift."""

    shift = tuple(int(value) for value in shift_voxels)
    if len(shift) != ndim:
        raise ValueError(f"{name} must contain {ndim} values.")
    return shift


def attach_perturbation_record(
    scene: RenderedScene,
    record: PerturbationRecord,
) -> RenderedScene:
    """Attach perturbation metadata to a rendered scene."""

    truth = scene.truth if scene.truth is not None else SceneTruth()
    existing = dict(truth.perturbations)
    record_key = f"{len(existing) + 1:03d}_{record.name}"

    record_payload = record.to_dict()
    existing[record_key] = record_payload

    truth_metadata = dict(truth.metadata)
    truth_metadata["n_perturbations"] = len(existing)

    new_truth = SceneTruth(
        geometric=truth.geometric,
        objects=truth.objects,
        scalar_fields=truth.scalar_fields,
        perturbations=existing,
        tables=truth.tables,
        metadata=truth_metadata,
    )

    metadata = dict(scene.metadata)
    perturbations = list(metadata.get("perturbations", []))
    perturbations.append(record_payload)
    metadata["perturbations"] = perturbations

    provenance = dict(scene.provenance)
    provenance["last_perturbation"] = record.name

    return replace(
        scene,
        truth=new_truth,
        metadata=metadata,
        provenance=provenance,
    )


def _json_safe(value: Any) -> Any:
    """Convert common NumPy values into JSON-friendly objects."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
