# Plotting

`synthetic-workshop` includes plotting helpers for inspecting rendered scenes.

The plotting layer is intended for quality control and visual communication. It is not required for rendering or export.

## Galleries

The gallery writer creates a compact visual summary of a rendered scene:

```python
from pathlib import Path

from synthworkshop.plotting.galleries import write_scene_gallery
from synthworkshop.scenes.config import render_scene_config_from_path

scene = render_scene_config_from_path("examples/basic_tube.yml")
write_scene_gallery(scene, Path("outputs/basic_tube/gallery"), overwrite=True)
```

## Plot types

The plotting package includes helpers for:

- slice views;
- projections;
- skeleton and centreline overlays;
- legends;
- gallery figures;
- shared plotting style.

Projections are often more informative than a single central slice for curved, branched, or oblique structures.

## Figure outputs

Gallery functions write image files to the chosen output directory. Available formats depend on the plotting function and Matplotlib backend.

For publication or reports, vector formats such as PDF or SVG are useful where supported. For quick inspection, PNG is usually sufficient.
