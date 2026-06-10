import numpy as np
import pytest

from synthworkshop.grid import GridSpec
from synthworkshop.scenes.model import (
    CompositionRules,
    LabelMode,
    MaskRules,
    ObjectRole,
    OverlapPolicy,
    RenderedScene,
    ScalarBlend,
    SceneObjectMetadata,
)


def _base_scene(
    *,
    overlap_policy: str = "allow",
    target_overlap: bool = False,
) -> RenderedScene:
    grid = GridSpec(shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
    scalar = np.zeros(grid.shape, dtype=float)
    labels = np.zeros(grid.shape, dtype=int)

    target = np.zeros(grid.shape, dtype=bool)
    target[1:3, 1:3, 1:3] = True

    support = np.zeros(grid.shape, dtype=bool)
    if target_overlap:
        support[2, 2, 2] = True
    support[0, 0, 0] = True

    object_masks = {
        "target": target,
        "support": support,
    }
    object_metadata = {
        "target": SceneObjectMetadata(
            object_id="target", role="target", label=1, priority=10
        ),
        "support": SceneObjectMetadata(
            object_id="support", role="analysis_support", label=2, priority=1
        ),
    }

    return RenderedScene(
        grid=grid,
        scalar_maps={"scalar": scalar},
        label_map=labels,
        object_masks=object_masks,
        object_metadata=object_metadata,
        composition=CompositionRules(overlap_policy=overlap_policy),
    )


def test_enum_coercion() -> None:
    rules = CompositionRules(
        label_mode="priority", scalar_blend="max", overlap_policy="error"
    )

    assert rules.label_mode is LabelMode.PRIORITY
    assert rules.scalar_blend is ScalarBlend.MAX
    assert rules.overlap_policy is OverlapPolicy.ERROR


def test_mask_rules_coerce_roles() -> None:
    rules = MaskRules(
        target_roles=("target",), analysis_roles=("target", "environment")
    )

    assert rules.target_roles == (ObjectRole.TARGET,)
    assert rules.analysis_roles == (ObjectRole.TARGET, ObjectRole.ENVIRONMENT)


def test_object_metadata_validates_label() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SceneObjectMetadata(object_id="bad", role="target", label=-1)


def test_rendered_scene_derives_target_and_analysis_masks() -> None:
    scene = _base_scene()

    assert scene.target_masks["target"].sum() == 8
    assert scene.analysis_masks["analysis"].sum() == 9
    assert scene.object_ids_by_role("target") == ("target",)
    assert scene.object_ids_by_role(ObjectRole.ANALYSIS_SUPPORT) == ("support",)


def test_rendered_scene_rejects_scalar_shape_mismatch() -> None:
    grid = GridSpec(shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
    labels = np.zeros(grid.shape, dtype=int)
    mask = np.zeros(grid.shape, dtype=bool)

    with pytest.raises(ValueError, match="scalar_maps"):
        RenderedScene(
            grid=grid,
            scalar_maps={"bad": np.zeros((3, 4, 4), dtype=float)},
            label_map=labels,
            object_masks={"target": mask},
            object_metadata={
                "target": SceneObjectMetadata(
                    object_id="target", role="target", label=1
                )
            },
        )


def test_overlap_policy_allow_records_overlap() -> None:
    scene = _base_scene(overlap_policy="allow", target_overlap=True)

    assert scene.overlap_report.has_overlap
    assert scene.overlap_report.n_overlap_voxels == 1
    assert scene.summary()["overlap"]["has_overlap"] is True


def test_overlap_policy_warn_emits_warning() -> None:
    with pytest.warns(UserWarning, match="Object masks overlap"):
        _base_scene(overlap_policy="warn", target_overlap=True)


def test_overlap_policy_error_raises() -> None:
    with pytest.raises(ValueError, match="overlap_policy='error'"):
        _base_scene(overlap_policy="error", target_overlap=True)


def test_combined_object_mask() -> None:
    scene = _base_scene()
    combined = scene.combined_object_mask(["target", "support"])

    assert combined.sum() == 9
