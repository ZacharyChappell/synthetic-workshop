"""Shared helpers for known-effect injection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from synthworkshop.ground_truth import SceneTruth
from synthworkshop.scenes import RenderedScene


@dataclass(frozen=True)
class EffectRecord:
    """Metadata for one known injected effect."""

    name: str
    target: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    affected_maps: tuple[str, ...] = ()
    affected_objects: tuple[str, ...] = ()
    support_voxels: int = 0
    magnitude: float | None = None
    expected_direction: str | None = None
    clean_null: bool = False
    truth_changed: bool = True
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""

        payload: dict[str, Any] = {
            "name": self.name,
            "target": self.target,
            "parameters": _json_safe(dict(self.parameters)),
            "affected_maps": list(self.affected_maps),
            "affected_objects": list(self.affected_objects),
            "support_voxels": int(self.support_voxels),
            "magnitude": self.magnitude,
            "expected_direction": self.expected_direction,
            "clean_null": self.clean_null,
            "truth_changed": self.truth_changed,
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


def attach_effect_record(scene: RenderedScene, record: EffectRecord) -> RenderedScene:
    """Attach known-effect metadata to a rendered scene."""

    truth = scene.truth if scene.truth is not None else SceneTruth()

    truth_metadata = dict(truth.metadata)
    existing = dict(truth_metadata.get("effects", {}))
    record_key = f"{len(existing) + 1:03d}_{record.name}"

    record_payload = record.to_dict()
    existing[record_key] = record_payload
    truth_metadata["effects"] = existing
    truth_metadata["n_effects"] = len(existing)

    new_truth = SceneTruth(
        geometric=truth.geometric,
        objects=truth.objects,
        scalar_fields=truth.scalar_fields,
        perturbations=truth.perturbations,
        tables=truth.tables,
        metadata=truth_metadata,
    )

    metadata = dict(scene.metadata)
    effects = list(metadata.get("effects", []))
    effects.append(record_payload)
    metadata["effects"] = effects

    provenance = dict(scene.provenance)
    provenance["last_effect"] = record.name

    return replace(
        scene,
        truth=new_truth,
        metadata=metadata,
        provenance=provenance,
    )


def require_object_mask(scene: RenderedScene, object_id: str) -> np.ndarray:
    """Return an object mask with validation."""

    if object_id not in scene.object_masks:
        raise ValueError(f"Unknown object_id: {object_id!r}.")
    return np.asarray(scene.object_masks[object_id], dtype=bool)


def require_scalar_map(scene: RenderedScene, map_name: str) -> np.ndarray:
    """Return a scalar map with validation."""

    if map_name not in scene.scalar_maps:
        raise ValueError(f"Unknown scalar map: {map_name!r}.")
    return np.asarray(scene.scalar_maps[map_name], dtype=float)


def validate_finite_float(value: float, *, name: str) -> float:
    """Return a finite float."""

    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def validate_positive_int(value: int, *, name: str) -> int:
    """Return a positive integer."""

    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive.")
    return out


def selected_object_ids(
    scene: RenderedScene,
    object_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return selected object IDs with validation."""

    if object_ids is None:
        selected = tuple(scene.object_masks)
    else:
        selected = tuple(str(object_id) for object_id in object_ids)

    missing = sorted(set(selected) - set(scene.object_masks))
    if missing:
        raise ValueError(f"Unknown object_id value(s): {missing}.")
    return selected


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
