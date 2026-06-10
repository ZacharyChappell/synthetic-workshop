from __future__ import annotations

import numpy as np
import pytest

from synthworkshop import GridSpec
from synthworkshop.primitives import ConeObject, FrustumObject
from synthworkshop.profiles import ConstantProfile, LinearRadialProfile
from synthworkshop.scenes import ObjectRole, RenderedScene


def test_cone_object_renders_linear_radius_support() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    cone = ConeObject(
        object_id="cone",
        apex_mm=(5.0, 10.0, 10.0),
        axis="i",
        height_mm=10.0,
        base_radius_mm=4.0,
        profile=ConstantProfile(value=2.0),
        map_name="qsm_like",
        role="inclusion",
        label=5,
        priority=10,
    )

    scene = cone.render(grid)

    assert isinstance(scene, RenderedScene)
    assert scene.object_masks["cone"][5, 10, 10]
    assert scene.object_masks["cone"][10, 12, 10]
    assert not scene.object_masks["cone"][10, 13, 10]
    assert scene.object_masks["cone"][15, 14, 10]
    assert not scene.object_masks["cone"][15, 15, 10]
    assert scene.label_map[5, 10, 10] == 5
    assert np.isclose(scene.scalar_maps["qsm_like"][5, 10, 10], 2.0)
    assert scene.object_metadata["cone"].role is ObjectRole.INCLUSION
    assert scene.truth.geometric["cone"]["kind"] == "cone"


def test_cone_object_records_axial_radial_and_rho_maps() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    cone = ConeObject(
        object_id="cone",
        apex_mm=(5.0, 10.0, 10.0),
        axis=0,
        height_mm=10.0,
        base_radius_mm=4.0,
        profile=LinearRadialProfile(),
    )

    scene = cone.render(grid)

    assert np.isclose(scene.signed_offset_maps["cone:axial_mm"][10, 10, 10], 5.0)
    assert np.isclose(scene.signed_offset_maps["cone:radial_mm"][10, 12, 10], 2.0)
    assert np.isclose(scene.signed_offset_maps["cone:rho"][10, 12, 10], 1.0)
    assert np.isclose(scene.distance_maps["cone"][10, 12, 10], 2.0)


def test_cone_object_supports_negative_axis_direction() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    cone = ConeObject(
        object_id="cone",
        apex_mm=(15.0, 10.0, 10.0),
        axis="i",
        axis_direction=-1,
        height_mm=10.0,
        base_radius_mm=4.0,
        profile=ConstantProfile(value=1.0),
    )

    scene = cone.render(grid)

    assert scene.object_masks["cone"][15, 10, 10]
    assert scene.object_masks["cone"][5, 14, 10]
    assert not scene.object_masks["cone"][16, 10, 10]
    assert scene.truth.geometric["cone"]["axis_direction"] == -1


def test_frustum_object_renders_linear_radius_support() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    frustum = FrustumObject(
        object_id="frustum",
        start_mm=(5.0, 10.0, 10.0),
        axis="i",
        height_mm=10.0,
        radius_start_mm=2.0,
        radius_end_mm=4.0,
        profile=ConstantProfile(value=1.5),
        map_name="wm_pve_like",
        role="environment",
        label=6,
    )

    scene = frustum.render(grid)

    assert scene.object_masks["frustum"][5, 12, 10]
    assert not scene.object_masks["frustum"][5, 13, 10]
    assert scene.object_masks["frustum"][10, 13, 10]
    assert scene.object_masks["frustum"][15, 14, 10]
    assert not scene.object_masks["frustum"][15, 15, 10]
    assert scene.label_map[5, 10, 10] == 6
    assert np.isclose(scene.scalar_maps["wm_pve_like"][5, 10, 10], 1.5)
    assert scene.object_metadata["frustum"].role is ObjectRole.ENVIRONMENT
    assert scene.truth.geometric["frustum"]["kind"] == "frustum"


def test_frustum_object_records_axial_radial_and_rho_maps() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    frustum = FrustumObject(
        object_id="frustum",
        start_mm=(5.0, 10.0, 10.0),
        axis=0,
        height_mm=10.0,
        radius_start_mm=2.0,
        radius_end_mm=4.0,
        profile=LinearRadialProfile(),
    )

    scene = frustum.render(grid)

    assert np.isclose(scene.signed_offset_maps["frustum:axial_mm"][10, 10, 10], 5.0)
    assert np.isclose(scene.signed_offset_maps["frustum:radial_mm"][10, 13, 10], 3.0)
    assert np.isclose(scene.signed_offset_maps["frustum:rho"][10, 13, 10], 1.0)


def test_frustum_object_supports_negative_axis_direction() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    frustum = FrustumObject(
        object_id="frustum",
        start_mm=(15.0, 10.0, 10.0),
        axis="i",
        axis_direction="-1",
        height_mm=10.0,
        radius_start_mm=2.0,
        radius_end_mm=4.0,
        profile=ConstantProfile(value=1.0),
    )

    scene = frustum.render(grid)

    assert scene.object_masks["frustum"][15, 12, 10]
    assert scene.object_masks["frustum"][5, 14, 10]
    assert not scene.object_masks["frustum"][16, 10, 10]
    assert scene.truth.geometric["frustum"]["axis_direction"] == -1


def test_cone_and_frustum_volume_formulae() -> None:
    cone = ConeObject(
        object_id="cone",
        apex_mm=(0.0, 0.0, 0.0),
        axis=0,
        height_mm=6.0,
        base_radius_mm=3.0,
        profile=ConstantProfile(),
    )
    frustum = FrustumObject(
        object_id="frustum",
        start_mm=(0.0, 0.0, 0.0),
        axis=0,
        height_mm=6.0,
        radius_start_mm=2.0,
        radius_end_mm=4.0,
        profile=ConstantProfile(),
    )

    assert np.isclose(cone.volume_mm3, (1.0 / 3.0) * np.pi * 9.0 * 6.0)
    assert np.isclose(frustum.volume_mm3, (np.pi * 6.0 / 3.0) * (4.0 + 8.0 + 16.0))


def test_cone_and_frustum_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="axis_direction"):
        ConeObject(
            object_id="bad",
            apex_mm=(0.0, 0.0, 0.0),
            axis=0,
            axis_direction=0,
            height_mm=1.0,
            base_radius_mm=1.0,
            profile=ConstantProfile(),
        )

    with pytest.raises(ValueError, match="radius_start_mm"):
        FrustumObject(
            object_id="bad",
            start_mm=(0.0, 0.0, 0.0),
            axis=0,
            height_mm=1.0,
            radius_start_mm=0.0,
            radius_end_mm=1.0,
            profile=ConstantProfile(),
        )


def test_cone_and_frustum_reject_2d_grid() -> None:
    grid = GridSpec(shape=(8, 8), spacing=(1.0, 1.0))
    cone = ConeObject(
        object_id="cone",
        apex_mm=(0.0, 0.0, 0.0),
        axis=0,
        height_mm=1.0,
        base_radius_mm=1.0,
        profile=ConstantProfile(),
    )

    with pytest.raises(ValueError, match="3D grid"):
        cone.render(grid)


def test_top_level_exports_cone_and_frustum_objects() -> None:
    import synthworkshop

    assert synthworkshop.ConeObject is ConeObject
    assert synthworkshop.FrustumObject is FrustumObject
