from __future__ import annotations

import numpy as np
import pytest

from synthworkshop import GridSpec
from synthworkshop.scenes import (
    CompositionRules,
    MaskRules,
    ObjectRole,
    RenderedScene,
    SceneObjectMetadata,
    compose_rendered_scenes,
)


def _manual_object_scene(
    *,
    object_id: str,
    role: str,
    label: int,
    priority: int,
    value: float,
    mask: np.ndarray,
    map_name: str = "scalar",
) -> RenderedScene:
    grid = GridSpec(shape=mask.shape, spacing=(1.0, 1.0, 1.0))
    scalar = np.zeros(grid.shape, dtype=float)
    scalar[mask] = value
    label_map = np.where(mask, label, 0).astype(np.int32)

    return RenderedScene(
        grid=grid,
        scalar_maps={map_name: scalar},
        label_map=label_map,
        object_masks={object_id: mask},
        object_metadata={
            object_id: SceneObjectMetadata(
                object_id=object_id,
                role=role,
                label=label,
                priority=priority,
            )
        },
        composition=CompositionRules(overlap_policy="allow"),
    )


def _mask(shape: tuple[int, int, int], slices) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[slices] = True
    return out


def test_compose_rendered_scenes_combines_non_overlapping_objects() -> None:
    shape = (5, 5, 5)
    target = _manual_object_scene(
        object_id="target",
        role="target",
        label=1,
        priority=10,
        value=1.0,
        mask=_mask(shape, np.s_[1:3, 1:3, 1:3]),
    )
    support = _manual_object_scene(
        object_id="support",
        role="analysis_support",
        label=2,
        priority=1,
        value=0.5,
        mask=_mask(shape, np.s_[3:5, 3:5, 3:5]),
    )

    scene = compose_rendered_scenes(
        [target, support],
        composition=CompositionRules(overlap_policy="allow"),
    )

    assert set(scene.object_masks) == {"target", "support"}
    assert scene.target_masks["target"].sum() == 8
    assert scene.analysis_masks["analysis"].sum() == 16
    assert np.all(scene.label_map[target.object_masks["target"]] == 1)
    assert np.all(scene.label_map[support.object_masks["support"]] == 2)


def test_default_analysis_mask_excludes_environment_role() -> None:
    shape = (5, 5, 5)
    target = _manual_object_scene(
        object_id="target",
        role="target",
        label=1,
        priority=10,
        value=1.0,
        mask=_mask(shape, np.s_[1:3, 1:3, 1:3]),
    )
    environment = _manual_object_scene(
        object_id="environment",
        role="environment",
        label=2,
        priority=1,
        value=0.5,
        mask=_mask(shape, np.s_[3:5, 3:5, 3:5]),
    )

    scene = compose_rendered_scenes(
        [target, environment],
        composition=CompositionRules(overlap_policy="allow"),
    )

    assert scene.analysis_masks["analysis"].sum() == 8
    assert scene.object_ids_by_role(ObjectRole.ENVIRONMENT) == ("environment",)


def test_custom_mask_rules_can_include_environment_in_analysis() -> None:
    shape = (5, 5, 5)
    target = _manual_object_scene(
        object_id="target",
        role="target",
        label=1,
        priority=10,
        value=1.0,
        mask=_mask(shape, np.s_[1:3, 1:3, 1:3]),
    )
    environment = _manual_object_scene(
        object_id="environment",
        role="environment",
        label=2,
        priority=1,
        value=0.5,
        mask=_mask(shape, np.s_[3:5, 3:5, 3:5]),
    )

    scene = compose_rendered_scenes(
        [target, environment],
        composition=CompositionRules(overlap_policy="allow"),
        mask_rules=MaskRules(analysis_roles=("target", "environment")),
    )

    assert scene.analysis_masks["analysis"].sum() == 16


def test_overlap_policy_error_raises_for_composed_scene() -> None:
    shape = (5, 5, 5)
    left = _manual_object_scene(
        object_id="left",
        role="target",
        label=1,
        priority=1,
        value=1.0,
        mask=_mask(shape, np.s_[1:4, 1:4, 1:4]),
    )
    right = _manual_object_scene(
        object_id="right",
        role="environment",
        label=2,
        priority=2,
        value=2.0,
        mask=_mask(shape, np.s_[2:5, 2:5, 2:5]),
    )

    with pytest.raises(ValueError, match="overlap_policy='error'"):
        compose_rendered_scenes(
            [left, right],
            composition=CompositionRules(overlap_policy="error"),
        )


def test_overlap_policy_warn_emits_warning_for_composed_scene() -> None:
    shape = (5, 5, 5)
    left = _manual_object_scene(
        object_id="left",
        role="target",
        label=1,
        priority=1,
        value=1.0,
        mask=_mask(shape, np.s_[1:4, 1:4, 1:4]),
    )
    right = _manual_object_scene(
        object_id="right",
        role="environment",
        label=2,
        priority=2,
        value=2.0,
        mask=_mask(shape, np.s_[2:5, 2:5, 2:5]),
    )

    with pytest.warns(UserWarning, match="Object masks overlap"):
        scene = compose_rendered_scenes(
            [left, right],
            composition=CompositionRules(overlap_policy="warn"),
        )

    assert scene.overlap_report.has_overlap
    assert scene.truth.tables["overlaps"].shape[0] == 1


def test_label_mode_priority_uses_highest_priority() -> None:
    shape = (5, 5, 5)
    low = _manual_object_scene(
        object_id="low",
        role="target",
        label=1,
        priority=1,
        value=1.0,
        mask=_mask(shape, np.s_[1:4, 1:4, 1:4]),
    )
    high = _manual_object_scene(
        object_id="high",
        role="environment",
        label=9,
        priority=10,
        value=2.0,
        mask=_mask(shape, np.s_[2:5, 2:5, 2:5]),
    )

    scene = compose_rendered_scenes(
        [low, high],
        composition=CompositionRules(label_mode="priority", overlap_policy="allow"),
    )

    assert scene.label_map[2, 2, 2] == 9


def test_label_mode_first_and_last_are_order_dependent() -> None:
    shape = (5, 5, 5)
    first = _manual_object_scene(
        object_id="first",
        role="target",
        label=1,
        priority=1,
        value=1.0,
        mask=_mask(shape, np.s_[1:4, 1:4, 1:4]),
    )
    second = _manual_object_scene(
        object_id="second",
        role="environment",
        label=2,
        priority=1,
        value=2.0,
        mask=_mask(shape, np.s_[2:5, 2:5, 2:5]),
    )

    first_scene = compose_rendered_scenes(
        [first, second],
        composition=CompositionRules(label_mode="first", overlap_policy="allow"),
    )
    last_scene = compose_rendered_scenes(
        [first, second],
        composition=CompositionRules(label_mode="last", overlap_policy="allow"),
    )

    assert first_scene.label_map[2, 2, 2] == 1
    assert last_scene.label_map[2, 2, 2] == 2


def test_scalar_blend_modes_for_overlap() -> None:
    shape = (5, 5, 5)
    left = _manual_object_scene(
        object_id="left",
        role="target",
        label=1,
        priority=1,
        value=1.0,
        mask=_mask(shape, np.s_[1:4, 1:4, 1:4]),
    )
    right = _manual_object_scene(
        object_id="right",
        role="environment",
        label=2,
        priority=2,
        value=2.0,
        mask=_mask(shape, np.s_[2:5, 2:5, 2:5]),
    )

    overwrite = compose_rendered_scenes(
        [left, right],
        composition=CompositionRules(scalar_blend="overwrite", overlap_policy="allow"),
    )
    max_scene = compose_rendered_scenes(
        [left, right],
        composition=CompositionRules(scalar_blend="max", overlap_policy="allow"),
    )
    sum_scene = compose_rendered_scenes(
        [left, right],
        composition=CompositionRules(scalar_blend="sum", overlap_policy="allow"),
    )
    mean_scene = compose_rendered_scenes(
        [left, right],
        composition=CompositionRules(
            scalar_blend="weighted_mean",
            overlap_policy="allow",
        ),
        scalar_weights={"left": 1.0, "right": 3.0},
    )

    assert np.isclose(overwrite.scalar_maps["scalar"][2, 2, 2], 2.0)
    assert np.isclose(max_scene.scalar_maps["scalar"][2, 2, 2], 2.0)
    assert np.isclose(sum_scene.scalar_maps["scalar"][2, 2, 2], 3.0)
    assert np.isclose(mean_scene.scalar_maps["scalar"][2, 2, 2], 1.75)


def test_duplicate_object_ids_are_rejected() -> None:
    shape = (5, 5, 5)
    first = _manual_object_scene(
        object_id="same",
        role="target",
        label=1,
        priority=1,
        value=1.0,
        mask=_mask(shape, np.s_[1:3, 1:3, 1:3]),
    )
    second = _manual_object_scene(
        object_id="same",
        role="environment",
        label=2,
        priority=2,
        value=2.0,
        mask=_mask(shape, np.s_[3:5, 3:5, 3:5]),
    )

    with pytest.raises(ValueError, match="Duplicate object_id"):
        compose_rendered_scenes([first, second])


def test_top_level_exports_composition_function() -> None:
    import synthworkshop

    assert synthworkshop.compose_rendered_scenes is compose_rendered_scenes
