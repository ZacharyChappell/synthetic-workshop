from __future__ import annotations

import numpy as np
import pytest

from synthworkshop import GridSpec
from synthworkshop.primitives import EllipsoidObject, SphereObject
from synthworkshop.profiles import ConstantProfile, LinearRadialProfile
from synthworkshop.scenes import ObjectRole, RenderedScene, render_objects


def test_sphere_object_renders_scalar_label_and_mask() -> None:
    grid = GridSpec(shape=(17, 17, 17), spacing=(1.0, 1.0, 1.0))
    sphere = SphereObject(
        object_id="lesion",
        centre_mm=(8.0, 8.0, 8.0),
        radius_mm=3.0,
        profile=ConstantProfile(value=2.0),
        map_name="qsm_like",
        role="inclusion",
        label=5,
        priority=10,
    )

    scene = sphere.render(grid)

    assert isinstance(scene, RenderedScene)
    assert scene.scalar_maps["qsm_like"].shape == grid.shape
    assert scene.object_masks["lesion"][8, 8, 8]
    assert np.isclose(scene.scalar_maps["qsm_like"][8, 8, 8], 2.0)
    assert scene.label_map[8, 8, 8] == 5
    assert scene.object_metadata["lesion"].role is ObjectRole.INCLUSION
    assert scene.truth.geometric["lesion"]["kind"] == "sphere"


def test_sphere_object_records_distance_and_offsets() -> None:
    grid = GridSpec(shape=(17, 17, 17), spacing=(1.0, 1.0, 1.0))
    sphere = SphereObject(
        object_id="lesion",
        centre_mm=(8.0, 8.0, 8.0),
        radius_mm=3.0,
        profile=LinearRadialProfile(),
    )

    scene = sphere.render(grid)

    assert np.isclose(scene.distance_maps["lesion"][8, 8, 8], 0.0)
    assert np.isclose(scene.distance_maps["lesion"][11, 8, 8], 3.0)
    assert np.isclose(scene.signed_offset_maps["lesion:i_mm"][11, 8, 8], 3.0)
    assert np.isclose(scene.signed_offset_maps["lesion:rho"][11, 8, 8], 1.0)


def test_ellipsoid_object_renders_anisotropic_support() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    ellipsoid = EllipsoidObject(
        object_id="ellipsoid",
        centre_mm=(10.0, 10.0, 10.0),
        radii_mm=(4.0, 2.0, 1.0),
        profile=ConstantProfile(value=1.5),
        map_name="scalar",
        role="environment",
        label=3,
    )

    scene = ellipsoid.render(grid)

    assert scene.object_masks["ellipsoid"][14, 10, 10]
    assert not scene.object_masks["ellipsoid"][15, 10, 10]
    assert scene.object_masks["ellipsoid"][10, 12, 10]
    assert not scene.object_masks["ellipsoid"][10, 13, 10]
    assert scene.object_masks["ellipsoid"][10, 10, 11]
    assert not scene.object_masks["ellipsoid"][10, 10, 12]
    assert np.isclose(scene.scalar_maps["scalar"][10, 10, 10], 1.5)
    assert scene.truth.geometric["ellipsoid"]["kind"] == "ellipsoid"


def test_ellipsoid_object_records_normalised_radius() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    ellipsoid = EllipsoidObject(
        object_id="ellipsoid",
        centre_mm=(10.0, 10.0, 10.0),
        radii_mm=(4.0, 2.0, 1.0),
        profile=LinearRadialProfile(),
    )

    scene = ellipsoid.render(grid)

    assert np.isclose(scene.signed_offset_maps["ellipsoid:rho"][14, 10, 10], 1.0)
    assert np.isclose(scene.signed_offset_maps["ellipsoid:rho"][10, 12, 10], 1.0)
    assert np.isclose(scene.signed_offset_maps["ellipsoid:rho"][10, 10, 11], 1.0)


def test_implicit_objects_can_be_composed_with_render_objects() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    lesion = SphereObject(
        object_id="lesion",
        centre_mm=(8.0, 10.0, 10.0),
        radius_mm=2.0,
        profile=ConstantProfile(value=2.0),
        map_name="qsm_like",
        role="inclusion",
        label=2,
        priority=10,
    )
    environment = EllipsoidObject(
        object_id="environment",
        centre_mm=(12.0, 10.0, 10.0),
        radii_mm=(3.0, 2.0, 2.0),
        profile=ConstantProfile(value=0.5),
        map_name="qsm_like",
        role="environment",
        label=3,
        priority=1,
    )

    scene = render_objects(grid, [environment, lesion])

    assert set(scene.object_masks) == {"environment", "lesion"}
    assert scene.scalar_maps["qsm_like"].shape == grid.shape
    assert scene.object_ids_by_role("inclusion") == ("lesion",)
    assert scene.object_ids_by_role("environment") == ("environment",)


def test_sphere_object_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="3D coordinate"):
        SphereObject(
            object_id="bad",
            centre_mm=(1.0, 2.0),
            radius_mm=1.0,
            profile=ConstantProfile(),
        )

    with pytest.raises(ValueError, match="radius_mm"):
        SphereObject(
            object_id="bad",
            centre_mm=(1.0, 2.0, 3.0),
            radius_mm=0.0,
            profile=ConstantProfile(),
        )


def test_ellipsoid_object_rejects_invalid_radii() -> None:
    with pytest.raises(ValueError, match="radii_mm"):
        EllipsoidObject(
            object_id="bad",
            centre_mm=(1.0, 2.0, 3.0),
            radii_mm=(1.0, 0.0, 1.0),
            profile=ConstantProfile(),
        )


def test_implicit_objects_reject_2d_grid() -> None:
    grid = GridSpec(shape=(8, 8), spacing=(1.0, 1.0))
    sphere = SphereObject(
        object_id="sphere",
        centre_mm=(1.0, 1.0, 1.0),
        radius_mm=1.0,
        profile=ConstantProfile(),
    )

    with pytest.raises(ValueError, match="3D grid"):
        sphere.render(grid)


def test_top_level_exports_implicit_objects() -> None:
    import synthworkshop

    assert synthworkshop.SphereObject is SphereObject
    assert synthworkshop.EllipsoidObject is EllipsoidObject
