# Examples

This folder contains small YAML scene specifications for `synthetic-workshop`.

Each example can be rendered with:

```python
from pathlib import Path

from synthworkshop.io.scene import export_scene
from synthworkshop.plotting.galleries import write_scene_gallery
from synthworkshop.scenes.config import render_scene_config_from_path

scene_name = "basic_tube"
scene = render_scene_config_from_path(f"examples/{scene_name}.yml")

output_dir = Path("outputs") / scene_name
export_scene(scene, output_dir / "export", overwrite=True)
write_scene_gallery(scene, output_dir / "gallery", overwrite=True)
```

## Included scenes

- `basic_tube.yml`: straight circular tube.
- `curved_elliptic_tube.yml`: curved tube with elliptic cross-section.
- `variable_radius_tube.yml`: tube with changing radius.
- `ribbon_tube.yml`: flattened ribbon-like tube.
- `tube_with_implicit_objects.yml`: tube with additional implicit objects.
- `tube_with_slab_environment.yml`: tube with slab-like environment.
- `cone_frustum_scene.yml`: cone and frustum-style implicit geometry.

The examples are intentionally small so that they can be rendered quickly and inspected directly.
