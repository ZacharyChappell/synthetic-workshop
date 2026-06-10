# Python API

The public Python interface centres on scene loading, rendering, export, and plotting.

## Render a YAML scene

```python
from synthworkshop.scenes.config import render_scene_config_from_path

scene = render_scene_config_from_path("examples/basic_tube.yml")
```

The returned object is a `RenderedScene`.

## Export a scene

```python
from pathlib import Path

from synthworkshop.io.scene import export_scene
from synthworkshop.scenes.config import render_scene_config_from_path

scene = render_scene_config_from_path("examples/basic_tube.yml")
export_scene(scene, Path("outputs/basic_tube/export"), overwrite=True)
```

## Write a gallery

```python
from pathlib import Path

from synthworkshop.plotting.galleries import write_scene_gallery
from synthworkshop.scenes.config import render_scene_config_from_path

scene = render_scene_config_from_path("examples/basic_tube.yml")
write_scene_gallery(scene, Path("outputs/basic_tube/gallery"), overwrite=True)
```

## Inspect scene contents

A rendered scene contains arrays, masks, object metadata, composition metadata, and truth objects where available.

Common attributes include:

- `grid`;
- `scalar_maps`;
- `label_map`;
- `object_masks`;
- `target_masks`;
- `analysis_masks`;
- `skeleton_masks`;
- `centrelines`;
- `frames`;
- `distance_maps`;
- `signed_offset_maps`;
- `truth`;
- `overlap_report`;
- `metadata`;
- `provenance`.

The exact attribute set depends on the rendered objects and the scene configuration.

## Build scenes programmatically

YAML/JSON is the most compact route for reproducible examples. Programmatic scene construction is also available through the lower-level modules:

- `synthworkshop.grid`;
- `synthworkshop.primitives`;
- `synthworkshop.cross_sections`;
- `synthworkshop.profiles`;
- `synthworkshop.scenes`.

For new examples, YAML files are usually the clearest representation because they keep geometry, scalar profiles, object roles, and composition rules visible in one place.
