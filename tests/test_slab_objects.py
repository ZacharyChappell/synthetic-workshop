from __future__ import annotations

import numpy as np
import pytest

from synthworkshop import GridSpec
from synthworkshop.primitives import SlabObject
from synthworkshop.profiles import ConstantProfile, LinearRadialProfile
from synthworkshop.scenes import ObjectRole, RenderedScene


def test_slab_object_renders_finite_sheet_support() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    slab = SlabObject(
        object_id="sheet",
        centre_mm=(10.0, 10.0, 10.0),
        normal_axis="j",
        thickness_mm=4.0,
        half_extent_mm=(5.0, 3.0),
        profile=ConstantProfile(value=0.5),
        map_name="wm_pve_like",
        role="environment",
        label=4,
        priority=1,
    )

    scene = slab.render(grid)

    assert isinstance(scene, RenderedScene)
    assert scene.object_masks["sheet"][10, 10, 10]
    assert scene.object_masks["sheet"][15, 10, 13]
    assert not scene.object_masks["sheet"][16, 10, 10]
    assert not scene.object_masks["sheet"][10, 13, 10]
    assert scene.label_map[10, 10, 10] == 4
    assert np.isclose(scene.scalar_maps["wm_pve_like"][10, 10, 10], 0.5)
    assert scene.object_metadata["sheet"].role is ObjectRole.ENVIRONMENT
    assert scene.truth.geometric["sheet"]["kind"] == "slab"


def test_slab_object_renders_infinite_in_plane_slab_across_grid() -> None:
    grid = GridSpec(shape=(11, 11, 11), spacing=(1.0, 1.0, 1.0))
    slab = SlabObject(
        object_id="sheet",
        centre_mm=(5.0, 5.0, 5.0),
        normal_axis=2,
        thickness_mm=2.0,
        half_extent_mm=None,
        profile=ConstantProfile(value=1.0),
    )

    scene = slab.render(grid)

    assert scene.object_masks["sheet"][:, :, 5].all()
    assert scene.object_masks["sheet"][:, :, 4].all()
    assert scene.object_masks["sheet"][:, :, 6].all()
    assert not scene.object_masks["sheet"][:, :, 3].any()
    assert scene.truth.geometric["sheet"]["half_extent_mm"] is None


def test_slab_object_records_normal_distance_and_offsets() -> None:
    grid = GridSpec(shape=(11, 11, 11), spacing=(1.0, 1.0, 1.0))
    slab = SlabObject(
        object_id="sheet",
        centre_mm=(5.0, 5.0, 5.0),
        normal_axis="k",
        thickness_mm=4.0,
        half_extent_mm=(3.0, 3.0),
        profile=LinearRadialProfile(),
    )

    scene = slab.render(grid)

    assert np.isclose(scene.distance_maps["sheet"][5, 5, 5], 0.0)
    assert np.isclose(scene.distance_maps["sheet"][5, 5, 7], 2.0)
    assert np.isclose(scene.signed_offset_maps["sheet:normal_mm"][5, 5, 7], 2.0)
    assert np.isclose(scene.signed_offset_maps["sheet:rho"][5, 5, 7], 1.0)


def test_slab_object_axis_aliases() -> None:
    assert (
        SlabObject(
            object_id="a",
            centre_mm=(0.0, 0.0, 0.0),
            normal_axis="x",
            thickness_mm=1.0,
            profile=ConstantProfile(),
        ).normal_axis
        == 0
    )
    assert (
        SlabObject(
            object_id="b",
            centre_mm=(0.0, 0.0, 0.0),
            normal_axis="j",
            thickness_mm=1.0,
            profile=ConstantProfile(),
        ).normal_axis
        == 1
    )


def test_slab_object_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="normal_axis"):
        SlabObject(
            object_id="bad",
            centre_mm=(0.0, 0.0, 0.0),
            normal_axis="bad",
            thickness_mm=1.0,
            profile=ConstantProfile(),
        )

    with pytest.raises(ValueError, match="thickness_mm"):
        SlabObject(
            object_id="bad",
            centre_mm=(0.0, 0.0, 0.0),
            normal_axis=0,
            thickness_mm=0.0,
            profile=ConstantProfile(),
        )

    with pytest.raises(ValueError, match="half_extent_mm"):
        SlabObject(
            object_id="bad",
            centre_mm=(0.0, 0.0, 0.0),
            normal_axis=0,
            thickness_mm=1.0,
            half_extent_mm=(1.0,),
            profile=ConstantProfile(),
        )


def test_slab_object_rejects_2d_grid() -> None:
    grid = GridSpec(shape=(8, 8), spacing=(1.0, 1.0))
    slab = SlabObject(
        object_id="sheet",
        centre_mm=(1.0, 1.0, 1.0),
        normal_axis=0,
        thickness_mm=1.0,
        profile=ConstantProfile(),
    )

    with pytest.raises(ValueError, match="3D grid"):
        slab.render(grid)


def test_top_level_exports_slab_object() -> None:
    import synthworkshop

    assert synthworkshop.SlabObject is SlabObject
