from __future__ import annotations

import numpy as np
import pytest

from synthworkshop import GridSpec
from synthworkshop.cross_sections import CircularCrossSection, EllipticCrossSection
from synthworkshop.primitives import LineCurve, SinusoidalCurve, TubeObject
from synthworkshop.profiles import AsymmetricLinearProfile, LinearRadialProfile
from synthworkshop.scenes import ObjectRole, RenderedScene


def _straight_tube_scene() -> RenderedScene:
    grid = GridSpec(shape=(17, 17, 17), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 8.0, 8.0),
        end_mm=(12.0, 8.0, 8.0),
    ).sample(step_mm=1.0, object_id="target")
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(centre_value=1.0, edge_value=0.2),
        map_name="fa_like",
        role="target",
        label=3,
        priority=10,
    )
    return tube.render(grid)


def test_tube_object_renders_single_object_scene() -> None:
    scene = _straight_tube_scene()

    assert isinstance(scene, RenderedScene)
    assert scene.scalar_maps["fa_like"].shape == scene.grid.shape
    assert scene.object_masks["target"].shape == scene.grid.shape
    assert scene.label_map.dtype == np.int32
    assert np.all(scene.label_map[scene.object_masks["target"]] == 3)
    assert scene.object_metadata["target"].role is ObjectRole.TARGET


def test_tube_rendering_has_expected_centre_and_background_values() -> None:
    scene = _straight_tube_scene()
    scalar = scene.scalar_maps["fa_like"]

    assert np.isclose(scalar[8, 8, 8], 1.0)
    assert np.isclose(scalar[8, 10, 8], 0.2)
    assert np.isclose(scalar[8, 11, 8], 0.0)


def test_tube_rendering_records_masks_and_truth() -> None:
    scene = _straight_tube_scene()

    assert scene.target_masks["target"].sum() == scene.object_masks["target"].sum()
    assert scene.analysis_masks["analysis"].sum() == scene.object_masks["target"].sum()
    assert scene.skeleton_masks["target"].sum() == 9
    assert "centrelines" in scene.truth.tables
    assert "frames" in scene.truth.tables
    assert scene.truth.objects["target"]["label"] == 3
    assert scene.truth.scalar_fields["fa_like"]["object_id"] == "target"


def test_tube_rendering_records_distance_and_signed_offsets() -> None:
    scene = _straight_tube_scene()

    distance = scene.distance_maps["target"]
    signed_u = scene.signed_offset_maps["target:u_mm"]
    signed_v = scene.signed_offset_maps["target:v_mm"]
    nearest = scene.signed_offset_maps["target:nearest_centreline_index"]

    assert np.isclose(distance[8, 8, 8], 0.0)
    assert np.isclose(distance[8, 10, 8], 2.0)
    assert np.isclose(signed_u[8, 10, 8], 2.0)
    assert np.isclose(signed_v[8, 8, 10], 2.0)
    assert nearest.shape == scene.grid.shape


def test_elliptic_tube_has_larger_extent_along_primary_axis() -> None:
    grid = GridSpec(shape=(21, 21, 21), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(5.0, 10.0, 10.0),
        end_mm=(15.0, 10.0, 10.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=EllipticCrossSection(
            semi_axis_u_mm=4.0,
            semi_axis_v_mm=1.0,
        ),
        profile=LinearRadialProfile(),
    )
    scene = tube.render(grid)
    central_slice = scene.object_masks["target"][10]
    coords = np.argwhere(central_slice)

    j_extent = coords[:, 0].max() - coords[:, 0].min()
    k_extent = coords[:, 1].max() - coords[:, 1].min()

    assert j_extent > k_extent


def test_asymmetric_tube_profile_uses_signed_primary_axis() -> None:
    grid = GridSpec(shape=(17, 17, 17), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 8.0, 8.0),
        end_mm=(12.0, 8.0, 8.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=AsymmetricLinearProfile(
            centre_value=1.0,
            edge_value=0.2,
            asymmetry=0.1,
        ),
    )
    scene = tube.render(grid)
    scalar = scene.scalar_maps["scalar"]

    assert scalar[8, 6, 8] < scalar[8, 10, 8]


def test_curved_tube_renders_non_empty_mask() -> None:
    grid = GridSpec(shape=(24, 24, 24), spacing=(1.0, 1.0, 1.0))
    centreline = SinusoidalCurve(
        start_mm=(4.0, 12.0, 12.0),
        end_mm=(20.0, 12.0, 12.0),
        amplitude_mm=(0.0, 3.0, 0.0),
        periods=1.0,
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(),
    )
    scene = tube.render(grid)

    assert scene.object_masks["target"].sum() > 0
    assert scene.skeleton_masks["target"].sum() > 0
    assert scene.truth.geometric["target"]["centreline_length_mm"] > 16.0


def test_tube_object_rejects_2d_centreline() -> None:
    centreline = LineCurve(start_mm=(0.0, 0.0), end_mm=(1.0, 0.0)).sample(n_samples=3)

    with pytest.raises(ValueError, match="3D centreline"):
        TubeObject(
            object_id="target",
            centreline=centreline,
            cross_section=CircularCrossSection(radius_mm=1.0),
            profile=LinearRadialProfile(),
        )


def test_tube_render_rejects_2d_grid() -> None:
    grid = GridSpec(shape=(8, 8), spacing=(1.0, 1.0))
    centreline = LineCurve(
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(1.0, 0.0, 0.0),
    ).sample(n_samples=3)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=1.0),
        profile=LinearRadialProfile(),
    )

    with pytest.raises(ValueError, match="3D grid"):
        tube.render(grid)


def test_top_level_exports_tube_object_and_profiles() -> None:
    import synthworkshop

    assert synthworkshop.TubeObject is TubeObject
    assert synthworkshop.LinearRadialProfile is LinearRadialProfile


def test_tube_object_renders_variable_radius_cross_section() -> None:
    from synthworkshop.cross_sections import VariableCircularCrossSection

    grid = GridSpec(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 10.0, 10.0),
        end_mm=(14.0, 10.0, 10.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=VariableCircularCrossSection(
            radius_start_mm=1.0,
            radius_end_mm=3.0,
            length_mm=centreline.length_mm,
        ),
        profile=LinearRadialProfile(centre_value=1.0, edge_value=0.2),
        map_name="scalar",
    )

    scene = tube.render(grid)

    start_slice_voxels = scene.object_masks["target"][4].sum()
    end_slice_voxels = scene.object_masks["target"][14].sum()

    assert end_slice_voxels > start_slice_voxels
    assert scene.truth.geometric["target"]["cross_section"]["kind"] == (
        "variable_circle_linear"
    )


def test_tube_object_records_variable_radius_offset_maps() -> None:
    from synthworkshop.cross_sections import VariableCircularCrossSection

    grid = GridSpec(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 10.0, 10.0),
        end_mm=(14.0, 10.0, 10.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=VariableCircularCrossSection(
            radius_start_mm=1.0,
            radius_end_mm=3.0,
            length_mm=centreline.length_mm,
        ),
        profile=LinearRadialProfile(),
    )

    scene = tube.render(grid)

    longitudinal = scene.signed_offset_maps["target:longitudinal_mm"]
    assert np.isclose(longitudinal[4, 10, 10], 0.0)
    assert np.isclose(longitudinal[14, 10, 10], centreline.length_mm)


def test_tube_object_renders_superellipse_cross_section() -> None:
    from synthworkshop.cross_sections import SuperellipseCrossSection

    grid = GridSpec(shape=(24, 24, 24), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 12.0, 12.0),
        end_mm=(18.0, 12.0, 12.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=SuperellipseCrossSection(
            semi_axis_u_mm=4.0,
            semi_axis_v_mm=1.0,
            exponent=6.0,
        ),
        profile=LinearRadialProfile(),
    )

    scene = tube.render(grid)

    assert scene.object_masks["target"].sum() > 0
    assert scene.truth.geometric["target"]["cross_section"]["kind"] == "superellipse"


def test_tube_object_renders_ribbon_cross_section() -> None:
    from synthworkshop.cross_sections import RibbonCrossSection

    grid = GridSpec(shape=(24, 24, 24), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 12.0, 12.0),
        end_mm=(18.0, 12.0, 12.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=RibbonCrossSection(
            width_mm=8.0,
            thickness_mm=2.0,
            exponent=8.0,
        ),
        profile=LinearRadialProfile(),
    )

    scene = tube.render(grid)

    assert scene.object_masks["target"].sum() > 0
    assert scene.truth.geometric["target"]["cross_section"]["kind"] == "ribbon"


def test_tube_object_renders_hollow_core_profile() -> None:
    from synthworkshop.profiles import HollowCoreProfile

    grid = GridSpec(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 10.0, 10.0),
        end_mm=(14.0, 10.0, 10.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=3.0),
        profile=HollowCoreProfile(
            core_value=0.1,
            shell_value=1.0,
            edge_value=0.2,
        ),
    )

    scene = tube.render(grid)

    centre_value = scene.scalar_maps["scalar"][9, 10, 10]
    shell_value = scene.scalar_maps["scalar"][9, 12, 10]

    assert shell_value > centre_value
    assert scene.truth.scalar_fields["scalar"]["profile"]["kind"] == "hollow_core"


def test_tube_object_renders_longitudinal_gradient_profile() -> None:
    from synthworkshop.profiles import LongitudinalGradientProfile

    grid = GridSpec(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 10.0, 10.0),
        end_mm=(14.0, 10.0, 10.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LongitudinalGradientProfile(
            start_value=0.2,
            end_value=1.0,
            length_mm=centreline.length_mm,
        ),
    )

    scene = tube.render(grid)

    start_value = scene.scalar_maps["scalar"][4, 10, 10]
    end_value = scene.scalar_maps["scalar"][14, 10, 10]

    assert np.isclose(start_value, 0.2)
    assert np.isclose(end_value, 1.0)
    assert scene.truth.scalar_fields["scalar"]["profile"]["kind"] == (
        "longitudinal_gradient"
    )


def test_tube_object_renders_one_sided_lesion_profile() -> None:
    from synthworkshop.profiles import OneSidedLesionProfile

    grid = GridSpec(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 10.0, 10.0),
        end_mm=(14.0, 10.0, 10.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=3.0),
        profile=OneSidedLesionProfile(
            baseline_value=0.2,
            lesion_delta=1.0,
            lesion_side="positive",
            lesion_centre_mm=1.0,
            lesion_width_mm=0.4,
        ),
    )

    scene = tube.render(grid)

    positive_side = scene.scalar_maps["scalar"][9, 11, 10]
    negative_side = scene.scalar_maps["scalar"][9, 9, 10]

    assert positive_side > negative_side
    assert scene.truth.scalar_fields["scalar"]["profile"]["kind"] == (
        "one_sided_lesion"
    )


def test_tube_object_renders_periodic_longitudinal_profile() -> None:
    from synthworkshop.profiles import PeriodicLongitudinalProfile

    grid = GridSpec(shape=(24, 24, 24), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 12.0, 12.0),
        end_mm=(20.0, 12.0, 12.0),
    ).sample(step_mm=1.0)
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=PeriodicLongitudinalProfile(
            baseline_value=1.0,
            amplitude=0.5,
            length_mm=centreline.length_mm,
            periods=1.0,
        ),
    )

    scene = tube.render(grid)

    assert np.isclose(scene.scalar_maps["scalar"][4, 12, 12], 1.0)
    assert np.isclose(scene.scalar_maps["scalar"][8, 12, 12], 1.5)
    assert scene.truth.scalar_fields["scalar"]["profile"]["kind"] == (
        "periodic_longitudinal"
    )
