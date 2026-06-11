"""Tests for scene perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.grid import GridSpec
from synthworkshop.perturbations import (
    add_gaussian_noise,
    dilate_masks,
    erode_masks,
    mean_blur_scalar_maps,
    shift_scalar_maps,
    shift_skeleton_masks,
)
from synthworkshop.scenes import (
    CompositionRules,
    RenderedScene,
    SceneObjectMetadata,
)


def _scene(*, cube_mask: bool = False) -> RenderedScene:
    grid = GridSpec(shape=(5, 5, 5), spacing=(1.0, 1.0, 1.0))

    scalar = np.zeros(grid.shape, dtype=float)
    scalar[2, 2, 2] = 1.0

    mask = np.zeros(grid.shape, dtype=bool)
    if cube_mask:
        mask[1:4, 1:4, 1:4] = True
    else:
        mask[2, 2, 2] = True

    label_map = np.where(mask, 1, 0).astype(np.int32)

    skeleton = np.zeros(grid.shape, dtype=bool)
    skeleton[2, 2, 2] = True

    return RenderedScene(
        grid=grid,
        scalar_maps={"scalar": scalar},
        label_map=label_map,
        object_masks={"target": mask},
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
        metadata={"scene_id": "perturbation_test"},
    )


def test_gaussian_noise_is_reproducible() -> None:
    scene = _scene()

    first = add_gaussian_noise(scene, sigma=0.1, seed=123)
    second = add_gaussian_noise(scene, sigma=0.1, seed=123)

    np.testing.assert_allclose(
        first.scalar_maps["scalar"], second.scalar_maps["scalar"]
    )
    assert first.truth.perturbations["001_gaussian_noise"]["seed"] == 123
    assert first.metadata["perturbations"][0]["name"] == "gaussian_noise"


def test_gaussian_noise_rejects_negative_sigma() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="sigma"):
        add_gaussian_noise(scene, sigma=-0.1)


def test_mean_blur_spreads_delta_scalar_map() -> None:
    scene = _scene()

    blurred = mean_blur_scalar_maps(scene, radius_voxels=1)

    assert blurred.scalar_maps["scalar"][2, 2, 2] == pytest.approx(1.0 / 27.0)
    assert blurred.scalar_maps["scalar"].sum() == pytest.approx(1.0)
    assert (
        blurred.truth.perturbations["001_mean_blur"]["parameters"]["radius_voxels"] == 1
    )


def test_scalar_integer_shift_moves_values_without_wraparound() -> None:
    scene = _scene()

    shifted = shift_scalar_maps(scene, shift_voxels=(1, 0, 0))

    assert shifted.scalar_maps["scalar"][2, 2, 2] == 0.0
    assert shifted.scalar_maps["scalar"][3, 2, 2] == 1.0
    assert shifted.truth.perturbations["001_integer_scalar_shift"]["parameters"][
        "shift_voxels"
    ] == [1, 0, 0]


def test_object_mask_erosion_rederives_role_masks() -> None:
    scene = _scene(cube_mask=True)

    eroded = erode_masks(scene, mask_group="object_masks", iterations=1)

    assert eroded.object_masks["target"].sum() == 1
    assert eroded.target_masks["target"].sum() == 1
    assert eroded.analysis_masks["analysis"].sum() == 1
    assert "001_mask_erosion" in eroded.truth.perturbations


def test_object_mask_dilation_expands_single_voxel() -> None:
    scene = _scene()

    dilated = dilate_masks(scene, mask_group="object_masks", iterations=1)

    assert dilated.object_masks["target"].sum() == 27
    assert dilated.target_masks["target"].sum() == 27
    assert "001_mask_dilation" in dilated.truth.perturbations


def test_skeleton_shift_moves_mask_only() -> None:
    scene = _scene()

    shifted = shift_skeleton_masks(scene, shift_voxels=(0, 1, 0))

    assert shifted.skeleton_masks["target"][2, 2, 2] == 0
    assert shifted.skeleton_masks["target"][2, 3, 2] == 1
    assert (
        shifted.truth.perturbations["001_integer_skeleton_shift"]["parameters"][
            "centrelines_updated"
        ]
        is False
    )


def test_mask_perturbation_rejects_unknown_group() -> None:
    scene = _scene()

    with pytest.raises(ValueError, match="mask_group"):
        erode_masks(scene, mask_group="bad")
