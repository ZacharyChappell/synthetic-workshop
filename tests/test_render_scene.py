from __future__ import annotations

import numpy as np

from synthworkshop import GridSpec
from synthworkshop.cross_sections import CircularCrossSection
from synthworkshop.primitives import LineCurve, TubeObject
from synthworkshop.profiles import ConstantProfile, LinearRadialProfile
from synthworkshop.scenes import CompositionRules, render_objects


def test_render_objects_composes_two_tubes() -> None:
    grid = GridSpec(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))

    target_centreline = LineCurve(
        start_mm=(4.0, 8.0, 8.0),
        end_mm=(15.0, 8.0, 8.0),
    ).sample(step_mm=1.0, object_id="target")

    support_centreline = LineCurve(
        start_mm=(4.0, 14.0, 8.0),
        end_mm=(15.0, 14.0, 8.0),
    ).sample(step_mm=1.0, object_id="support")

    target = TubeObject(
        object_id="target",
        centreline=target_centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(centre_value=1.0, edge_value=0.2),
        map_name="fa_like",
        role="target",
        label=1,
        priority=10,
    )
    support = TubeObject(
        object_id="support",
        centreline=support_centreline,
        cross_section=CircularCrossSection(radius_mm=1.5),
        profile=ConstantProfile(value=0.5),
        map_name="fa_like",
        role="analysis_support",
        label=2,
        priority=1,
    )

    scene = render_objects(
        grid,
        [target, support],
        composition=CompositionRules(overlap_policy="allow"),
    )

    assert set(scene.object_masks) == {"target", "support"}
    assert scene.scalar_maps["fa_like"].shape == grid.shape
    assert scene.target_masks["target"].sum() == scene.object_masks["target"].sum()
    assert scene.analysis_masks["analysis"].sum() == (
        scene.object_masks["target"].sum() + scene.object_masks["support"].sum()
    )
    assert np.isclose(scene.scalar_maps["fa_like"][8, 8, 8], 1.0)
    assert np.isclose(scene.scalar_maps["fa_like"][8, 14, 8], 0.5)
    assert "objects" in scene.truth.tables
    assert "overlaps" in scene.truth.tables


def test_render_objects_preserves_source_centreline_tables() -> None:
    grid = GridSpec(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(4.0, 8.0, 8.0),
        end_mm=(12.0, 8.0, 8.0),
    ).sample(step_mm=1.0, object_id="target")
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(),
        map_name="scalar",
        role="target",
        label=1,
    )

    scene = render_objects(grid, [tube])

    assert "centrelines" in scene.truth.tables
    assert scene.truth.tables["centrelines"].shape[0] == centreline.n_points
    assert scene.centrelines["target"].shape[0] == centreline.n_points


def test_top_level_exports_render_objects() -> None:
    import synthworkshop

    assert synthworkshop.render_objects is render_objects
