# Examples

The `examples/` folder contains small YAML scenes that can be rendered directly.

## `basic_tube.yml`

A simple straight tube with a circular cross-section and radial scalar profile.

This is the best first example for checking installation, rendering, export, and gallery creation.

## `curved_elliptic_tube.yml`

A curved tube with an elliptic cross-section.

This example is useful for inspecting curved centreline geometry, local frames, and anisotropic cross-sectional support.

## `variable_radius_tube.yml`

A tube with changing radius along its length.

This example tests longitudinal variation in object width and width-related ground truth.

## `ribbon_tube.yml`

A flattened ribbon-like tube.

This example is useful for sheet-like or highly anisotropic cross-sectional support.

## `tube_with_implicit_objects.yml`

A tube rendered with additional implicit objects.

This example illustrates how inclusions or neighbouring analytic structures can be represented in the same scene.

## `tube_with_slab_environment.yml`

A tube rendered with a slab-like environment.

This example is useful for neighbouring-compartment and boundary-contamination scenarios.

## `cone_frustum_scene.yml`

A scene containing cone or frustum-like implicit objects.

This example illustrates non-tube analytic primitives and simple tapered compartments.

## Running an example

```python
from pathlib import Path

from synthworkshop.io.scene import export_scene
from synthworkshop.plotting.galleries import write_scene_gallery
from synthworkshop.scenes.config import render_scene_config_from_path

scene = render_scene_config_from_path("examples/basic_tube.yml")

export_scene(scene, Path("outputs/basic_tube/export"), overwrite=True)
write_scene_gallery(scene, Path("outputs/basic_tube/gallery"), overwrite=True)
```
