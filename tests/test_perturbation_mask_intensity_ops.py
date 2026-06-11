"""Tests for extended perturbation operations."""

from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.grid import GridSpec
from synthworkshop.perturbations import (
    add_linear_bias_field,
    add_mask_contamination,
    add_mask_holes,
    break_skeleton_masks,
    scale_scalar_maps,
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

    source_mask = np.zeros(grid.shape, dtype=bool)
    source_mask[0, 0, 0] = True
    source_mask[0, 0, 1] = True

    label_map = np.zeros(grid.shape, dtype=np.int32)
    label_map[target_mask] = 1
    label_map[source_mask] = 2

    skeleton = np.zeros(grid.shape, dtype=bool)
    skeleton[2, 1:4, 2] = True

    return RenderedScene(
        grid=grid,
        scalar_maps={"scalar": scalar},
        label_map=label_map,
        object_masks={
            "target": target_mask,
            "source": source_mask,
        },
        object_metadata={
            "target": SceneObjectMetadata(
                object_id="target",
                role="target",
                label=1,
                priority=10,
            ),
            "source": SceneObjectMetadata(
                object_id="source",
                role="distractor",
                label=2,
                priority=5,
            ),
        },
        skeleton_masks={"target": skeleton},
        composition=CompositionRules(overlap_policy="allow"),
        metadata={"scene_id": "extended_perturbation_test"},
    )


def test_scale_scalar_maps_multiplies_selected_map() -> None:
    scene = _scene()

    scaled = scale_scalar_maps(scene, factor=2.5, map_names=["scalar"])

    assert scaled.scalar_maps["scalar"][2, 2, 2] == pytest.approx(2.5)
    assert scaled.truth.perturbations["001_intensity_scaling"]["parameters"][
        "factor"
    ] == pytest.approx(2.5)


def test_scale_scalar_maps_can_be_masked() -> None:
    scene = _scene()
    mask = np.zeros(scene.grid.shape, dtype=bool)
    mask[2, 2, 2] = True

    scaled = scale_scalar_maps(scene, factor=3.0, mask=mask)

    assert scaled.scalar_maps["scalar"][2, 2, 2] == pytest.approx(3.0)
    assert scaled.scalar_maps["scalar"][0, 0, 0] == pytest.approx(1.0)


def test_linear_bias_field_varies_along_axis() -> None:
    scene = _scene()

    biased = add_linear_bias_field(
        scene,
        axis=0,
        start_scale=1.0,
        end_scale=2.0,
    )

    assert biased.scalar_maps["scalar"][0, 2, 2] == pytest.approx(1.0)
    assert biased.scalar_maps["scalar"][-1, 2, 2] == pytest.approx(2.0)
    assert (
        biased.truth.perturbations["001_linear_bias_field"]["parameters"]["axis"] == 0
    )


def test_linear_bias_field_rejects_invalid_axis() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="axis"):
        add_linear_bias_field(scene, axis=3, start_scale=1.0, end_scale=2.0)


def test_add_mask_holes_removes_known_local_block() -> None:
    scene = _scene()

    holed = add_mask_holes(
        scene,
        mask_group="object_masks",
        mask_names=["target"],
        n_holes=1,
        radius_voxels=1,
        centres_voxels=[(2, 2, 2)],
    )

    assert holed.object_masks["target"].sum() == 0
    assert holed.target_masks["target"].sum() == 0
    assert "001_mask_holes" in holed.truth.perturbations


def test_add_mask_contamination_from_source_mask() -> None:
    scene = _scene()

    contaminated = add_mask_contamination(
        scene,
        mask_group="object_masks",
        mask_names=["target"],
        source_mask_group="object_masks",
        source_mask_name="source",
    )

    assert contaminated.object_masks["target"][0, 0, 0]
    assert contaminated.object_masks["target"][0, 0, 1]
    assert contaminated.object_masks["target"].sum() == 29
    assert "001_mask_contamination" in contaminated.truth.perturbations


def test_add_mask_contamination_can_sample_fraction() -> None:
    scene = _scene()

    contaminated = add_mask_contamination(
        scene,
        mask_group="object_masks",
        mask_names=["target"],
        source_mask_group="object_masks",
        source_mask_name="source",
        fraction=0.5,
        seed=10,
    )

    assert contaminated.object_masks["target"].sum() == 28
    assert contaminated.truth.perturbations["001_mask_contamination"]["seed"] == 10


def test_break_skeleton_masks_removes_known_block() -> None:
    scene = _scene()

    broken = break_skeleton_masks(
        scene,
        mask_names=["target"],
        n_breaks=1,
        radius_voxels=0,
        centres_voxels=[(2, 2, 2)],
    )

    assert broken.skeleton_masks["target"].sum() == 2
    assert not broken.skeleton_masks["target"][2, 2, 2]
    assert (
        broken.truth.perturbations["001_broken_skeleton"]["parameters"][
            "frames_updated"
        ]
        is False
    )


def test_break_skeleton_masks_rejects_bad_centre_shape() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="centres_voxels"):
        break_skeleton_masks(
            scene,
            centres_voxels=[(1, 2)],
        )
