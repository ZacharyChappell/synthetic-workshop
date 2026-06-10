# synthetic-workshop

`synthetic-workshop` is a Python package for generating analytic synthetic 2D/3D image scenes for image-method validation.

The package renders reproducible scenes with known geometry, object masks, scalar fields, centrelines, frames, topology, distance-like maps, composition metadata, plotting outputs, and export-ready 
arrays and tables. It is motivated by tract-centred neuroimaging validation, but the core package is intentionally general.

`synthetic-workshop` is not an MRI acquisition simulator. MRI-like map names, when used, refer to analytic validation fields rather than simulated acquisition physics.

## Design
The package uses a compositional scene model. A scene is built from a grid, analytic objects, curves or centrelines, cross-sections, scalar profiles, semantic object roles, and explicit composition rules.

The intended workflow is:

```text
GridSpec
  -> analytic primitives
  -> curves / centrelines / graph structures
  -> cross-section models
  -> scalar profiles
  -> objects and roles
  -> scene composition
  -> rendered scene
  -> ground-truth tables
  -> export / plotting / downstream adapters / GUI
```

Named phantoms such as straight tubes, curved tubes, slab environments, inclusions, crossings, and stress cases are expressed as scene configurations rather than as the primary internal API. This keeps the package extensible and makes example scenes inspectable.

## Current capabilities

The repository currently supports a working analytic scene loop:

```text
YAML/JSON scene specification
  -> GridSpec
  -> analytic scene objects
  -> render_objects(...)
  -> RenderedScene
  -> export_scene(...)
  -> write_scene_gallery(...)
```

Implemented or actively represented in the current codebase are:

* grid and coordinate helpers;
* rendered-scene and scene-truth containers;
* object roles and mask semantics;
* explicit composition and overlap metadata;
* line and sinusoidal curves;
* tube rendering;
* circular and elliptic cross-sections;
* variable, ribbon, superellipse, and rotating-ellipse cross-section modules;
* scalar profile modules, including radial and structured profiles;
* implicit object support, including sphere, ellipsoid, slab, cone, and frusta;
* topology modules for graphs, graph centrelines, and graph tubes;
* YAML/JSON scene loading;
* NumPy/TSV/JSON export;
* projection, slice, skeleton, legend, and gallery plotting;
* public synthetic examples;
* tests covering the core geometry, scene, IO, plotting, topology, cross-section, profile, implicit-object, GUI-helper, and workflow modules.

The public API is still under active development. The examples and tests are the most reliable executable reference for the current interface.

## Installation

Install the package in editable mode:

```bash
python -m pip install -e ".[dev,io,plot]"
```

The optional GUI dependency is separated from the core package:

```bash
python -m pip install -e ".[gui]"
```

## Minimal usage

Render a scene from a YAML specification, export the arrays and tables, and write a gallery:

```python
from pathlib import Path

from synthworkshop.io.scene import export_scene
from synthworkshop.plotting.galleries import write_scene_gallery
from synthworkshop.scenes.config import render_scene_config_from_path

scene = render_scene_config_from_path("examples/basic_tube.yml")

output_dir = Path("outputs/basic_tube")
export_scene(scene, output_dir / "export", overwrite=True)
write_scene_gallery(scene, output_dir / "gallery", overwrite=True)
```

The exact public API may change while the package is under active development. Prefer the examples and tests as the current executable reference.

## Example scenes

Current example scene specifications include:

```text
examples/basic_tube.yml
examples/curved_elliptic_tube.yml
examples/variable_radius_tube.yml
examples/ribbon_tube.yml
examples/tube_with_implicit_objects.yml
examples/tube_with_slab_environment.yml
examples/cone_frustum_scene.yml
```

These examples are designed to be small, reproducible, and suitable for tests or manual inspection.

## Outputs

Rendered scenes may contain:

* scalar maps;
* label maps;
* object masks;
* target masks;
* analysis masks;
* skeleton masks;
* centrelines;
* frames;
* distance maps;
* signed offset maps;
* scene truth;
* composition metadata;
* overlap reports;
* provenance.

Export layout is organised into predictable directories:

```text
arrays/
tables/
metadata/
```

This layout is intended to be easy to inspect directly and straightforward for downstream packages to consume.

## Documentation

Core documentation lives under `docs/`.

It details:

* scene schema;
* modelling concepts;
* example scenes;
* exported outputs. 

```text
docs/README.md
docs/concepts.md
docs/scene_schema.md
docs/python_api.md
docs/examples.md
docs/outputs.md
docs/plotting.md
```

## Relationship to TraCSS

TraCSS is a downstream consumer. The workshop generates:

* synthetic scenes;
* arrays;
* masks;
* labels;
* centrelines;
* skeletons;
* frames;
* distance maps;
* scalar maps;
* truth tables;
* case metadata.

The workshop does not own:

* TraCSS edge strategies;
* TraCSS feature extraction;
* TraCSS benchmark claims;
* TraCSS paper figures;
* TraCSS statistical conclusions.

## GUI direction

A GUI workbench has been implemented as a user-facing layer.

The GUI lets users build, visualise, test, validate, export, and print scenes visually as well as through Python and YAML. It is essentially a thin interface over the scene schema and renderer.

The GUI supports:

* loading example scenes;
* editing scene YAML/JSON;
* editing common object parameters through forms;
* placing and orienting objects;
* previewing scalar maps, labels, masks, skeletons, and overlaps;
* validating scene configurations;
* exporting arrays, tables, metadata, and figures.

## Development checks

Formatting/linting and tests were run at each stage of development.

```bash
ruff format .
ruff check .
ruff check . --fix
python -m pytest -q
```

## Licence

MIT Licence.

