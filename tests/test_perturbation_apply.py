"""Tests for perturbation dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.grid import GridSpec
from synthworkshop.perturbations import (
    apply_perturbation,
    apply_perturbations,
    available_perturbations,
)
from synthworkshop.scenes import (
    CompositionRules,
    RenderedScene,
    SceneObjectMetadata,
)


def _scene() -> RenderedScene:
    grid = GridSpec(shape=(5, 5, 5), spacing=(1.0, 1.0, 1.0))

    scalar = np.ones(grid.shape, dtype=float)

    target_mask = np.zeros(grid.shape, dtype=bool)
    target_mask[1:4, 1:4, 1:4] = True

    label_map = np.where(target_mask, 1, 0).astype(np.int32)

    skeleton = np.zeros(grid.shape, dtype=bool)
    skeleton[2, 1:4, 2] = True

    return RenderedScene(
        grid=grid,
        scalar_maps={"scalar": scalar},
        label_map=label_map,
        object_masks={"target": target_mask},
        object_metadata={
            "target": SceneObjectMetadata(
                object_id="target",
                role="target",
                label=1,
                priority=10,
            )
        },
        skeleton_masks={"target": skeleton},
        composition=CompositionRules(overlap_policy="allow"),
        metadata={"scene_id": "perturbation_apply_test"},
    )


def test_available_perturbations_exposes_core_kinds() -> None:
    kinds = available_perturbations()

    assert "gaussian_noise" in kinds
    assert "mask_holes" in kinds
    assert "linear_bias_field" in kinds
    assert "broken_skeleton" in kinds


def test_apply_perturbation_dispatches_one_specification() -> None:
    scene = _scene()

    perturbed = apply_perturbation(
        scene,
        {
            "kind": "intensity_scaling",
            "factor": 2.0,
            "map_names": ["scalar"],
        },
    )

    assert perturbed.scalar_maps["scalar"][2, 2, 2] == pytest.approx(2.0)
    assert "001_intensity_scaling" in perturbed.truth.perturbations


def test_apply_perturbations_applies_specs_in_order() -> None:
    scene = _scene()

    perturbed = apply_perturbations(
        scene,
        [
            {
                "kind": "intensity_scaling",
                "factor": 2.0,
                "map_names": ["scalar"],
            },
            {
                "kind": "linear_bias_field",
                "axis": 0,
                "start_scale": 1.0,
                "end_scale": 3.0,
                "map_names": ["scalar"],
            },
        ],
    )

    assert perturbed.scalar_maps["scalar"][0, 2, 2] == pytest.approx(2.0)
    assert perturbed.scalar_maps["scalar"][-1, 2, 2] == pytest.approx(6.0)
    assert list(perturbed.truth.perturbations) == [
        "001_intensity_scaling",
        "002_linear_bias_field",
    ]


def test_apply_perturbations_accepts_none_or_empty_specs() -> None:
    scene = _scene()

    assert apply_perturbations(scene, None) is scene
    assert apply_perturbations(scene, []) is scene


def test_apply_perturbations_can_dispatch_mask_operation() -> None:
    scene = _scene()

    perturbed = apply_perturbations(
        scene,
        [
            {
                "kind": "mask_holes",
                "mask_group": "object_masks",
                "mask_names": ["target"],
                "n_holes": 1,
                "radius_voxels": 1,
                "centres_voxels": [(2, 2, 2)],
            }
        ],
    )

    assert perturbed.object_masks["target"].sum() == 0
    assert perturbed.target_masks["target"].sum() == 0
    assert "001_mask_holes" in perturbed.truth.perturbations


def test_apply_perturbation_rejects_missing_kind() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="kind"):
        apply_perturbation(scene, {"factor": 2.0})


def test_apply_perturbation_rejects_unknown_kind() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="Unknown perturbation kind"):
        apply_perturbation(scene, {"kind": "does_not_exist"})


def test_apply_perturbation_rejects_non_mapping_specification() -> None:
    scene = _scene()

    with pytest.raises(TypeError, match="mapping"):
        apply_perturbation(scene, ["not", "a", "mapping"])  # type: ignore[arg-type]
