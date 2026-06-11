from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from synthworkshop.scenes.config import (
    SceneSpec,
    load_scene_spec,
    render_scene_from_dict,
    render_scene_from_path,
    render_scene_from_spec,
    scene_spec_from_dict,
    scene_spec_to_dict,
)


def _basic_payload() -> dict:
    return {
        "schema_version": "0.1",
        "scene": {
            "id": "unit_basic",
            "description": "Unit-test scene.",
        },
        "grid": {
            "shape": [18, 18, 18],
            "spacing": [1.0, 1.0, 1.0],
        },
        "composition": {
            "label_mode": "priority",
            "scalar_blend": "overwrite",
            "overlap_policy": "allow",
        },
        "objects": [
            {
                "id": "target",
                "kind": "tube",
                "role": "target",
                "label": 1,
                "priority": 10,
                "map_name": "fa_like",
                "curve": {
                    "kind": "line",
                    "start_mm": [4.0, 9.0, 9.0],
                    "end_mm": [14.0, 9.0, 9.0],
                    "step_mm": 1.0,
                },
                "cross_section": {
                    "kind": "circle",
                    "radius_mm": 2.0,
                },
                "profile": {
                    "kind": "linear_radial",
                    "centre_value": 1.0,
                    "edge_value": 0.2,
                    "background_value": 0.0,
                },
            }
        ],
    }


def test_scene_spec_from_dict_parses_basic_payload() -> None:
    spec = scene_spec_from_dict(_basic_payload())

    assert isinstance(spec, SceneSpec)
    assert spec.scene_id == "unit_basic"
    assert spec.grid.shape == (18, 18, 18)
    assert len(spec.objects) == 1
    assert spec.objects[0].object_id == "target"
    assert spec.composition.label_mode.value == "priority"


def test_scene_spec_to_dict_is_serialisable_summary() -> None:
    spec = scene_spec_from_dict(_basic_payload())
    summary = scene_spec_to_dict(spec)

    assert summary["scene"]["id"] == "unit_basic"
    assert summary["grid"]["shape"] == (18, 18, 18)
    assert summary["objects"][0]["id"] == "target"


def test_render_scene_from_dict_renders_basic_tube() -> None:
    scene = render_scene_from_dict(_basic_payload())

    assert scene.grid.shape == (18, 18, 18)
    assert "fa_like" in scene.scalar_maps
    assert "target" in scene.object_masks
    assert scene.target_masks["target"].sum() == scene.object_masks["target"].sum()
    assert np.isclose(scene.scalar_maps["fa_like"][9, 9, 9], 1.0)
    assert scene.metadata["scene_id"] == "unit_basic"


def test_render_scene_from_spec_renders_sinusoidal_ellipse() -> None:
    payload = _basic_payload()
    payload["objects"][0]["curve"] = {
        "kind": "sinusoidal",
        "start_mm": [4.0, 9.0, 9.0],
        "end_mm": [14.0, 9.0, 9.0],
        "amplitude_mm": [0.0, 2.0, 0.0],
        "periods": 1.0,
        "step_mm": 1.0,
    }
    payload["objects"][0]["cross_section"] = {
        "kind": "ellipse",
        "semi_axis_u_mm": 3.0,
        "semi_axis_v_mm": 1.0,
    }
    payload["objects"][0]["profile"] = {
        "kind": "gaussian_radial",
        "centre_value": 1.0,
        "edge_value": 0.2,
        "sigma_fraction": 0.5,
    }

    spec = scene_spec_from_dict(payload)
    scene = render_scene_from_spec(spec)

    assert scene.object_masks["target"].sum() > 0
    assert scene.truth.geometric["target"]["centreline_length_mm"] > 10.0


def test_load_scene_spec_json(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(_basic_payload()), encoding="utf-8")

    spec = load_scene_spec(path)

    assert spec.scene_id == "unit_basic"


def test_render_scene_from_path_json(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(_basic_payload()), encoding="utf-8")

    scene = render_scene_from_path(path)

    assert scene.scalar_maps["fa_like"].shape == (18, 18, 18)


def test_load_scene_spec_yaml(tmp_path: Path) -> None:
    path = tmp_path / "scene.yml"
    path.write_text(
        """
schema_version: "0.1"
scene:
  id: yaml_scene
grid:
  shape: [18, 18, 18]
  spacing: [1.0, 1.0, 1.0]
objects:
  - id: target
    kind: tube
    role: target
    label: 1
    priority: 10
    map_name: fa_like
    curve:
      kind: line
      start_mm: [4.0, 9.0, 9.0]
      end_mm: [14.0, 9.0, 9.0]
    cross_section:
      kind: circle
      radius_mm: 2.0
    profile:
      kind: constant
      value: 1.0
""",
        encoding="utf-8",
    )

    spec = load_scene_spec(path)

    assert spec.scene_id == "yaml_scene"
    assert spec.objects[0].profile["kind"] == "constant"


def test_invalid_curve_kind_raises() -> None:
    payload = _basic_payload()
    payload["objects"][0]["curve"]["kind"] = "unknown_curve"

    with pytest.raises(ValueError, match="Unknown curve kind"):
        render_scene_from_dict(payload)


def test_invalid_cross_section_kind_raises() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"]["kind"] = "triangle"

    with pytest.raises(ValueError, match="Unknown cross_section kind"):
        render_scene_from_dict(payload)


def test_invalid_profile_kind_raises() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"]["kind"] = "mystery_profile"

    with pytest.raises(ValueError, match="Unknown profile kind"):
        render_scene_from_dict(payload)


def test_unknown_render_option_raises() -> None:
    payload = _basic_payload()
    payload["render"] = {"not_an_option": 1}

    with pytest.raises(ValueError, match="Unknown render option"):
        scene_spec_from_dict(payload)


def test_example_basic_tube_yaml_renders() -> None:
    scene = render_scene_from_path("examples/basic_tube.yml")

    assert scene.metadata["scene_id"] == "basic_tube"
    assert "fa_like" in scene.scalar_maps
    assert scene.object_masks["target"].sum() > 0


def test_example_curved_elliptic_tube_yaml_renders() -> None:
    scene = render_scene_from_path("examples/curved_elliptic_tube.yml")

    assert scene.metadata["scene_id"] == "curved_elliptic_tube"
    assert set(scene.object_masks) == {"target", "support_neighbour"}
    assert scene.analysis_masks["analysis"].sum() >= scene.target_masks["target"].sum()


def test_top_level_exports_scene_config_helpers() -> None:
    import synthworkshop

    assert synthworkshop.scene_spec_from_dict is scene_spec_from_dict
    assert synthworkshop.render_scene_from_path is render_scene_from_path


def test_scene_config_supports_variable_circular_cross_section() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"] = {
        "kind": "variable_circle_linear",
        "radius_start_mm": 1.0,
        "radius_end_mm": 3.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.object_masks["target"].sum() > 0
    assert scene.truth.geometric["target"]["cross_section"]["kind"] == (
        "variable_circle_linear"
    )

    start_slice_voxels = scene.object_masks["target"][4].sum()
    end_slice_voxels = scene.object_masks["target"][14].sum()
    assert end_slice_voxels > start_slice_voxels


def test_scene_config_variable_circular_cross_section_accepts_aliases() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"] = {
        "kind": "tapered_circle",
        "start_radius_mm": 1.0,
        "end_radius_mm": 3.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.truth.geometric["target"]["cross_section"]["kind"] == (
        "variable_circle_linear"
    )


def test_scene_config_variable_circular_cross_section_requires_radii() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"] = {
        "kind": "variable_circle_linear",
        "radius_start_mm": 1.0,
    }

    with pytest.raises(ValueError, match="radius_end_mm"):
        render_scene_from_dict(payload)


def test_example_variable_radius_tube_yaml_renders() -> None:
    scene = render_scene_from_path("examples/variable_radius_tube.yml")

    assert scene.metadata["scene_id"] == "variable_radius_tube"
    assert "fa_like" in scene.scalar_maps
    assert scene.object_masks["target"].sum() > 0
    assert scene.truth.geometric["target"]["cross_section"]["radius_start_mm"] == 1.0
    assert scene.truth.geometric["target"]["cross_section"]["radius_end_mm"] == 4.0


def test_scene_config_supports_variable_elliptic_cross_section() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"] = {
        "kind": "variable_ellipse_linear",
        "semi_axis_u_start_mm": 2.0,
        "semi_axis_u_end_mm": 4.0,
        "semi_axis_v_start_mm": 1.0,
        "semi_axis_v_end_mm": 2.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.truth.geometric["target"]["cross_section"]["kind"] == (
        "variable_ellipse_linear"
    )
    assert scene.object_masks["target"].sum() > 0


def test_scene_config_supports_rotating_elliptic_cross_section() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"] = {
        "kind": "rotating_ellipse_linear",
        "semi_axis_u_mm": 4.0,
        "semi_axis_v_mm": 1.0,
        "angle_start_radians": 0.0,
        "angle_end_radians": 1.5707963267948966,
    }

    scene = render_scene_from_dict(payload)

    assert scene.truth.geometric["target"]["cross_section"]["kind"] == (
        "rotating_ellipse_linear"
    )
    assert scene.object_masks["target"].sum() > 0


def test_scene_config_supports_superellipse_cross_section() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"] = {
        "kind": "superellipse",
        "semi_axis_u_mm": 4.0,
        "semi_axis_v_mm": 1.0,
        "exponent": 6.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.truth.geometric["target"]["cross_section"]["kind"] == "superellipse"
    assert scene.truth.geometric["target"]["cross_section"]["exponent"] == 6.0
    assert scene.object_masks["target"].sum() > 0


def test_scene_config_supports_ribbon_cross_section() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"] = {
        "kind": "ribbon",
        "width_mm": 8.0,
        "thickness_mm": 2.0,
        "exponent": 8.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.truth.geometric["target"]["cross_section"]["kind"] == "ribbon"
    assert scene.truth.geometric["target"]["cross_section"]["width_mm"] == 8.0
    assert scene.truth.geometric["target"]["cross_section"]["thickness_mm"] == 2.0
    assert scene.object_masks["target"].sum() > 0


def test_scene_config_supports_ribbon_aliases() -> None:
    payload = _basic_payload()
    payload["objects"][0]["cross_section"] = {
        "kind": "flattened_ribbon",
        "ribbon_width_mm": 8.0,
        "ribbon_thickness_mm": 2.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.truth.geometric["target"]["cross_section"]["kind"] == "ribbon"


def test_example_ribbon_tube_yaml_renders() -> None:
    scene = render_scene_from_path("examples/ribbon_tube.yml")

    assert scene.metadata["scene_id"] == "ribbon_tube"
    assert scene.truth.geometric["target"]["cross_section"]["kind"] == "ribbon"
    assert scene.object_masks["target"].sum() > 0


def test_scene_config_supports_slab_object() -> None:
    payload = {
        "scene": {"id": "slab_scene"},
        "grid": {
            "shape": [20, 20, 20],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "gm_like_sheet",
                "kind": "slab",
                "role": "environment",
                "label": 4,
                "priority": 1,
                "map_name": "wm_pve_like",
                "centre_mm": [10.0, 10.0, 10.0],
                "normal_axis": "j",
                "thickness_mm": 4.0,
                "half_extent_mm": [5.0, 3.0],
                "profile": {
                    "kind": "constant",
                    "value": 0.5,
                    "background_value": 0.0,
                },
            }
        ],
    }

    scene = render_scene_from_dict(payload)

    assert scene.metadata["scene_id"] == "slab_scene"
    assert scene.object_masks["gm_like_sheet"][10, 10, 10]
    assert scene.truth.geometric["gm_like_sheet"]["kind"] == "slab"
    assert scene.truth.geometric["gm_like_sheet"]["normal_axis_name"] == "j"
    assert scene.scalar_maps["wm_pve_like"][10, 10, 10] == 0.5


def test_scene_config_supports_slab_aliases_and_extent() -> None:
    payload = {
        "grid": {
            "shape": [20, 20, 20],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "sheet",
                "kind": "sheet",
                "role": "environment",
                "label": 4,
                "center": [10.0, 10.0, 10.0],
                "axis": "k",
                "thickness": 4.0,
                "extent_mm": [10.0, 6.0],
                "profile": {
                    "kind": "constant",
                    "value": 0.5,
                },
            }
        ],
    }

    scene = render_scene_from_dict(payload)

    assert scene.truth.geometric["sheet"]["kind"] == "slab"
    assert scene.truth.geometric["sheet"]["half_extent_mm"] == [5.0, 3.0]


def test_scene_config_slab_requires_geometry_parameters() -> None:
    payload = {
        "grid": {
            "shape": [20, 20, 20],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "sheet",
                "kind": "slab",
                "role": "environment",
                "label": 4,
                "centre_mm": [10.0, 10.0, 10.0],
                "normal_axis": "k",
                "profile": {
                    "kind": "constant",
                    "value": 0.5,
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="slab thickness"):
        render_scene_from_dict(payload)


def test_example_tube_with_slab_environment_yaml_renders() -> None:
    scene = render_scene_from_path("examples/tube_with_slab_environment.yml")

    assert scene.metadata["scene_id"] == "tube_with_slab_environment"
    assert set(scene.object_masks) == {"target", "gm_like_sheet"}
    assert scene.truth.geometric["gm_like_sheet"]["kind"] == "slab"
    assert set(scene.scalar_maps) == {"fa_like", "wm_pve_like"}


def test_scene_config_supports_cone_object() -> None:
    payload = {
        "scene": {"id": "cone_scene"},
        "grid": {
            "shape": [21, 21, 21],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "cone",
                "kind": "cone",
                "role": "inclusion",
                "label": 5,
                "priority": 10,
                "map_name": "qsm_like",
                "apex_mm": [5.0, 10.0, 10.0],
                "axis": "i",
                "height_mm": 10.0,
                "base_radius_mm": 4.0,
                "profile": {
                    "kind": "constant",
                    "value": 2.0,
                    "background_value": 0.0,
                },
            }
        ],
    }

    scene = render_scene_from_dict(payload)

    assert scene.metadata["scene_id"] == "cone_scene"
    assert scene.object_masks["cone"][5, 10, 10]
    assert scene.truth.geometric["cone"]["kind"] == "cone"
    assert scene.scalar_maps["qsm_like"][5, 10, 10] == 2.0


def test_scene_config_supports_frustum_object() -> None:
    payload = {
        "scene": {"id": "frustum_scene"},
        "grid": {
            "shape": [21, 21, 21],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "frustum",
                "kind": "frustum",
                "role": "environment",
                "label": 6,
                "priority": 1,
                "map_name": "wm_pve_like",
                "start_mm": [5.0, 10.0, 10.0],
                "axis": "i",
                "height_mm": 10.0,
                "radius_start_mm": 2.0,
                "radius_end_mm": 4.0,
                "profile": {
                    "kind": "constant",
                    "value": 1.5,
                    "background_value": 0.0,
                },
            }
        ],
    }

    scene = render_scene_from_dict(payload)

    assert scene.metadata["scene_id"] == "frustum_scene"
    assert scene.object_masks["frustum"][5, 10, 10]
    assert scene.truth.geometric["frustum"]["kind"] == "frustum"
    assert scene.scalar_maps["wm_pve_like"][5, 10, 10] == 1.5


def test_scene_config_supports_cone_and_frustum_aliases() -> None:
    cone_payload = {
        "grid": {
            "shape": [21, 21, 21],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "cone",
                "kind": "cone_object",
                "role": "inclusion",
                "label": 5,
                "apex": [5.0, 10.0, 10.0],
                "axis": "i",
                "height": 10.0,
                "radius": 4.0,
                "profile": {
                    "kind": "constant",
                    "value": 2.0,
                },
            }
        ],
    }
    frustum_payload = {
        "grid": {
            "shape": [21, 21, 21],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "frustum",
                "kind": "truncated_cone",
                "role": "environment",
                "label": 6,
                "base_centre_mm": [5.0, 10.0, 10.0],
                "axis": "i",
                "height": 10.0,
                "start_radius_mm": 2.0,
                "end_radius_mm": 4.0,
                "profile": {
                    "kind": "constant",
                    "value": 1.5,
                },
            }
        ],
    }

    cone_scene = render_scene_from_dict(cone_payload)
    frustum_scene = render_scene_from_dict(frustum_payload)

    assert cone_scene.truth.geometric["cone"]["kind"] == "cone"
    assert frustum_scene.truth.geometric["frustum"]["kind"] == "frustum"


def test_scene_config_cone_requires_geometry_parameters() -> None:
    payload = {
        "grid": {
            "shape": [21, 21, 21],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "cone",
                "kind": "cone",
                "role": "inclusion",
                "label": 5,
                "apex_mm": [5.0, 10.0, 10.0],
                "axis": "i",
                "height_mm": 10.0,
                "profile": {
                    "kind": "constant",
                    "value": 2.0,
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="cone base radius"):
        render_scene_from_dict(payload)


def test_scene_config_frustum_requires_geometry_parameters() -> None:
    payload = {
        "grid": {
            "shape": [21, 21, 21],
            "spacing": [1.0, 1.0, 1.0],
        },
        "objects": [
            {
                "id": "frustum",
                "kind": "frustum",
                "role": "environment",
                "label": 6,
                "start_mm": [5.0, 10.0, 10.0],
                "axis": "i",
                "height_mm": 10.0,
                "radius_start_mm": 2.0,
                "profile": {
                    "kind": "constant",
                    "value": 1.5,
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="frustum end radius"):
        render_scene_from_dict(payload)


def test_example_cone_frustum_scene_yaml_renders() -> None:
    scene = render_scene_from_path("examples/cone_frustum_scene.yml")

    assert scene.metadata["scene_id"] == "cone_frustum_scene"
    assert set(scene.object_masks) == {"cone_inclusion", "tapered_environment"}
    assert scene.truth.geometric["cone_inclusion"]["kind"] == "cone"
    assert scene.truth.geometric["tapered_environment"]["kind"] == "frustum"
    assert set(scene.scalar_maps) == {"qsm_like", "wm_pve_like"}


def test_scene_config_supports_hollow_core_profile() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "hollow_core",
        "core_value": 0.1,
        "shell_value": 1.0,
        "edge_value": 0.2,
        "core_radius_fraction": 0.25,
        "shell_radius_fraction": 0.75,
        "background_value": 0.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.object_metadata["target"].metadata["profile"]["kind"] == "hollow_core"
    assert scene.object_masks["target"].sum() > 0


def test_scene_config_supports_sigmoid_boundary_profile() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "sigmoid_boundary",
        "centre_value": 1.0,
        "edge_value": 0.2,
        "boundary_fraction": 0.65,
        "width_fraction": 0.08,
        "background_value": 0.0,
    }

    scene = render_scene_from_dict(payload)

    assert (
        scene.object_metadata["target"].metadata["profile"]["kind"]
        == "sigmoid_boundary"
    )
    assert scene.object_masks["target"].sum() > 0


def test_scene_config_supports_longitudinal_gradient_profile_with_inferred_length() -> (
    None
):
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "longitudinal_gradient",
        "start_value": 0.2,
        "end_value": 1.0,
        "background_value": 0.0,
    }

    scene = render_scene_from_dict(payload)

    profile = scene.object_metadata["target"].metadata["profile"]
    assert profile["kind"] == "longitudinal_gradient"
    assert profile["length_mm"] > 0.0
    assert scene.object_masks["target"].sum() > 0


def test_scene_config_supports_multi_peak_radial_profile() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "multi_peak_radial",
        "base_value": 0.0,
        "peak_centres_fraction": [0.25, 0.75],
        "peak_amplitudes": [1.0, 0.5],
        "peak_widths_fraction": [0.05, 0.05],
        "background_value": 0.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.object_metadata["target"].metadata["profile"]["kind"] == (
        "multi_peak_radial"
    )


def test_scene_config_supports_one_sided_lesion_profile() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "one_sided_lesion",
        "baseline_value": 0.2,
        "lesion_delta": 1.0,
        "lesion_side": "positive",
        "lesion_centre_mm": 1.0,
        "lesion_width_mm": 0.5,
        "background_value": 0.0,
    }

    scene = render_scene_from_dict(payload)

    assert scene.object_metadata["target"].metadata["profile"]["kind"] == (
        "one_sided_lesion"
    )


def test_scene_config_supports_radial_longitudinal_gradient_profile() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "radial_longitudinal_gradient",
        "centre_start_value": 1.0,
        "centre_end_value": 2.0,
        "edge_start_value": 0.2,
        "edge_end_value": 0.8,
        "background_value": 0.0,
    }

    scene = render_scene_from_dict(payload)

    profile = scene.object_metadata["target"].metadata["profile"]
    assert profile["kind"] == "radial_longitudinal_gradient"
    assert profile["length_mm"] > 0.0


def test_scene_config_supports_periodic_longitudinal_profile() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "periodic_longitudinal",
        "baseline_value": 1.0,
        "amplitude": 0.2,
        "periods": 2.0,
        "phase_radians": 0.0,
        "background_value": 0.0,
    }

    scene = render_scene_from_dict(payload)

    profile = scene.object_metadata["target"].metadata["profile"]
    assert profile["kind"] == "periodic_longitudinal"
    assert profile["length_mm"] > 0.0


def test_scene_spec_from_dict_parses_perturbations() -> None:
    payload = _basic_payload()
    payload["perturbations"] = [
        {
            "kind": "intensity_scaling",
            "factor": 2.0,
            "map_names": ["fa_like"],
        }
    ]

    spec = scene_spec_from_dict(payload)
    summary = scene_spec_to_dict(spec)

    assert spec.perturbations == (
        {
            "kind": "intensity_scaling",
            "factor": 2.0,
            "map_names": ["fa_like"],
        },
    )
    assert summary["perturbations"][0]["kind"] == "intensity_scaling"


def test_render_scene_from_dict_applies_perturbations() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "constant",
        "value": 1.0,
        "background_value": 0.0,
    }
    payload["perturbations"] = [
        {
            "kind": "intensity_scaling",
            "factor": 2.0,
            "map_names": ["fa_like"],
        }
    ]

    scene = render_scene_from_dict(payload)

    assert np.isclose(scene.scalar_maps["fa_like"][9, 9, 9], 2.0)
    assert "001_intensity_scaling" in scene.truth.perturbations
    assert scene.metadata["perturbations"][0]["name"] == "intensity_scaling"
    assert scene.metadata["scene_spec"]["perturbations"][0]["kind"] == (
        "intensity_scaling"
    )


def test_render_scene_from_path_yaml_applies_perturbations(tmp_path: Path) -> None:
    path = tmp_path / "perturbed_scene.yml"
    path.write_text(
        """
schema_version: "0.1"
scene:
  id: yaml_perturbed_scene
grid:
  shape: [18, 18, 18]
  spacing: [1.0, 1.0, 1.0]
objects:
  - id: target
    kind: tube
    role: target
    label: 1
    priority: 10
    map_name: fa_like
    curve:
      kind: line
      start_mm: [4.0, 9.0, 9.0]
      end_mm: [14.0, 9.0, 9.0]
    cross_section:
      kind: circle
      radius_mm: 2.0
    profile:
      kind: constant
      value: 1.0
      background_value: 0.0
perturbations:
  - kind: intensity_scaling
    factor: 2.0
    map_names: [fa_like]
""",
        encoding="utf-8",
    )

    scene = render_scene_from_path(path)

    assert scene.metadata["scene_id"] == "yaml_perturbed_scene"
    assert np.isclose(scene.scalar_maps["fa_like"][9, 9, 9], 2.0)
    assert "001_intensity_scaling" in scene.truth.perturbations


def test_invalid_perturbation_section_raises() -> None:
    payload = _basic_payload()
    payload["perturbations"] = {"kind": "intensity_scaling"}

    with pytest.raises(ValueError, match="perturbations"):
        scene_spec_from_dict(payload)


def test_unknown_perturbation_kind_from_config_raises() -> None:
    payload = _basic_payload()
    payload["perturbations"] = [{"kind": "does_not_exist"}]

    with pytest.raises(ValueError, match="Unknown perturbation kind"):
        render_scene_from_dict(payload)


def test_example_perturbed_tube_yaml_renders() -> None:
    scene = render_scene_from_path("examples/perturbed_tube.yml")

    assert scene.metadata["scene_id"] == "perturbed_tube"
    assert "scalar" in scene.scalar_maps
    assert scene.object_masks["target"].sum() > 0
    assert list(scene.truth.perturbations) == [
        "001_intensity_scaling",
        "002_gaussian_noise",
    ]
    assert scene.metadata["perturbations"][0]["name"] == "intensity_scaling"
    assert scene.metadata["perturbations"][1]["seed"] == 123


def test_scene_spec_from_dict_parses_effects() -> None:
    payload = _basic_payload()
    payload["effects"] = [
        {
            "kind": "object_value_shift",
            "object_id": "target",
            "map_name": "fa_like",
            "delta": 0.5,
        }
    ]

    spec = scene_spec_from_dict(payload)
    summary = scene_spec_to_dict(spec)

    assert spec.effects == (
        {
            "kind": "object_value_shift",
            "object_id": "target",
            "map_name": "fa_like",
            "delta": 0.5,
        },
    )
    assert summary["effects"][0]["kind"] == "object_value_shift"


def test_render_scene_from_dict_applies_effects() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "constant",
        "value": 1.0,
        "background_value": 0.0,
    }
    payload["effects"] = [
        {
            "kind": "object_value_shift",
            "object_id": "target",
            "map_name": "fa_like",
            "delta": 0.5,
        }
    ]

    scene = render_scene_from_dict(payload)

    assert np.isclose(scene.scalar_maps["fa_like"][9, 9, 9], 1.5)
    assert "001_object_value_shift" in scene.truth.metadata["effects"]
    assert scene.metadata["effects"][0]["name"] == "object_value_shift"
    assert scene.metadata["scene_spec"]["effects"][0]["kind"] == "object_value_shift"


def test_effects_are_applied_before_perturbations() -> None:
    payload = _basic_payload()
    payload["objects"][0]["profile"] = {
        "kind": "constant",
        "value": 1.0,
        "background_value": 0.0,
    }
    payload["effects"] = [
        {
            "kind": "object_value_shift",
            "object_id": "target",
            "map_name": "fa_like",
            "delta": 0.5,
        }
    ]
    payload["perturbations"] = [
        {
            "kind": "intensity_scaling",
            "factor": 2.0,
            "map_names": ["fa_like"],
        }
    ]

    scene = render_scene_from_dict(payload)

    assert np.isclose(scene.scalar_maps["fa_like"][9, 9, 9], 3.0)
    assert "001_object_value_shift" in scene.truth.metadata["effects"]
    assert "001_intensity_scaling" in scene.truth.perturbations


def test_render_scene_from_path_yaml_applies_effects(tmp_path: Path) -> None:
    path = tmp_path / "known_effect_scene.yml"
    path.write_text(
        """
schema_version: "0.1"
scene:
  id: yaml_known_effect_scene
grid:
  shape: [18, 18, 18]
  spacing: [1.0, 1.0, 1.0]
objects:
  - id: target
    kind: tube
    role: target
    label: 1
    priority: 10
    map_name: fa_like
    curve:
      kind: line
      start_mm: [4.0, 9.0, 9.0]
      end_mm: [14.0, 9.0, 9.0]
    cross_section:
      kind: circle
      radius_mm: 2.0
    profile:
      kind: constant
      value: 1.0
      background_value: 0.0
effects:
  - kind: object_value_shift
    object_id: target
    map_name: fa_like
    delta: 0.5
""",
        encoding="utf-8",
    )

    scene = render_scene_from_path(path)

    assert scene.metadata["scene_id"] == "yaml_known_effect_scene"
    assert np.isclose(scene.scalar_maps["fa_like"][9, 9, 9], 1.5)
    assert "001_object_value_shift" in scene.truth.metadata["effects"]


def test_invalid_effects_section_raises() -> None:
    payload = _basic_payload()
    payload["effects"] = {"kind": "object_value_shift"}

    with pytest.raises(ValueError, match="effects"):
        scene_spec_from_dict(payload)


def test_unknown_effect_kind_from_config_raises() -> None:
    payload = _basic_payload()
    payload["effects"] = [{"kind": "does_not_exist"}]

    with pytest.raises(ValueError, match="Unknown effect kind"):
        render_scene_from_dict(payload)


def test_example_known_effect_tube_yaml_renders() -> None:
    scene = render_scene_from_path("examples/known_effect_tube.yml")

    assert scene.metadata["scene_id"] == "known_effect_tube"
    assert "scalar" in scene.scalar_maps
    assert scene.object_masks["target"].sum() > 0

    assert np.isclose(scene.scalar_maps["scalar"][16, 16, 16], 1.4)
    assert np.isclose(scene.scalar_maps["scalar"][10, 16, 16], 1.0)

    assert list(scene.truth.metadata["effects"]) == [
        "001_axis_interval_value_shift",
    ]
    record = scene.truth.metadata["effects"]["001_axis_interval_value_shift"]
    assert record["affected_objects"] == ["target"]
    assert record["affected_maps"] == ["scalar"]
    assert record["expected_direction"] == "increase"
    assert record["clean_null"] is False
    assert scene.metadata["effects"][0]["name"] == "axis_interval_value_shift"
