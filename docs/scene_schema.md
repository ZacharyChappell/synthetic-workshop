# Scene schema

Scenes can be specified in YAML or JSON. The same structure is used by the Python loader.

A typical scene contains:

```yaml
scene:
  id: basic_tube
  description: A simple circular tube.

grid:
  shape: [32, 32, 32]
  spacing: [1.0, 1.0, 1.0]

composition:
  label_mode: priority
  scalar_blend: overwrite
  overlap_policy: warn

masks:
  target_roles: [target]
  analysis_roles: [target, analysis_support]

objects:
  - id: tube
    kind: tube
    role: target
    label: 1
    map_name: scalar
    priority: 10
    curve:
      kind: line
      start_mm: [6.0, 16.0, 16.0]
      end_mm: [25.0, 16.0, 16.0]
      n_points: 20
    cross_section:
      kind: circular
      radius_mm: 4.0
    profile:
      kind: linear_radial
      centre_value: 1.0
      edge_value: 0.2
      background_value: 0.0
```

## Top-level sections

### `scene`

Scene-level metadata.

Common fields:

- `id`: short scene identifier;
- `description`: human-readable description;
- additional metadata fields may be included by examples or higher-level workflows.

### `grid`

Grid definition.

Common fields:

- `shape`: array shape, usually three integers;
- `spacing`: voxel spacing in synthetic millimetres;
- `origin`: optional world-coordinate origin;
- `axis_names`: optional axis names.

### `composition`

Rules for combining multiple rendered objects.

Current values include:

- `label_mode`: `priority`, `first`, or `last`;
- `scalar_blend`: `overwrite`, `max`, `sum`, or `weighted_mean`;
- `overlap_policy`: `allow`, `warn`, or `error`.

### `masks`

Rules for deriving target and analysis masks from object roles.

Common fields:

- `target_roles`: roles included in default target masks;
- `analysis_roles`: roles included in default analysis masks.

### `objects`

List of objects rendered into the scene. Every object has an identifier, kind, role, label, scalar map name, geometry, and scalar profile.

Common fields:

- `id` or `object_id`;
- `kind`;
- `role`;
- `label`;
- `priority`;
- `map_name`;
- `name`;
- `description`;
- `metadata`;
- `profile`;
- geometry fields.

## Object kinds

Current scene specifications support:

- `tube`;
- `sphere`;
- `ellipsoid`;
- `slab` or `sheet`;
- `cone`;
- `frustum` or `truncated_cone`.

Tube objects use a `curve` and `cross_section`. Implicit objects use object-specific parameters.

## Curves

Current scene specifications support:

### `line`

A straight centreline.

Common fields:

```yaml
curve:
  kind: line
  start_mm: [6.0, 16.0, 16.0]
  end_mm: [25.0, 16.0, 16.0]
  n_points: 20
```

### `sinusoidal`

A sinusoidal centreline.

Common fields include start and end coordinates, amplitude, phase or period-like parameters, and sampling density. See `examples/curved_elliptic_tube.yml` for an executable example.

## Cross-sections

Tube cross-sections describe the object support around a centreline.

Current kinds include:

- `circular`;
- `elliptic`;
- `superellipse`;
- `ribbon`;
- `variable_circle`;
- `variable_ellipse`;
- `rotating_ellipse`.

Examples:

```yaml
cross_section:
  kind: circular
  radius_mm: 4.0
```

```yaml
cross_section:
  kind: elliptic
  semi_axis_u_mm: 5.0
  semi_axis_v_mm: 2.5
```

```yaml
cross_section:
  kind: ribbon
  width_mm: 8.0
  thickness_mm: 2.0
```

```yaml
cross_section:
  kind: variable_circle
  radius_start_mm: 3.0
  radius_end_mm: 6.0
```

## Scalar profiles

Profiles define scalar values inside rendered objects.

Current profile kinds include:

- `constant`;
- `linear_radial` or `linear_radial_decay`;
- `gaussian_radial`;
- `edge_enhanced`;
- `asymmetric_linear`;
- `hollow_core`;
- `sigmoid_boundary` or `soft_boundary`;
- `longitudinal_gradient` or `linear_longitudinal`;
- `radial_longitudinal_gradient`;
- `multi_peak_radial`;
- `one_sided_lesion`;
- `periodic_longitudinal`.

Examples:

```yaml
profile:
  kind: constant
  value: 1.0
  background_value: 0.0
```

```yaml
profile:
  kind: linear_radial
  centre_value: 1.0
  edge_value: 0.2
  background_value: 0.0
```

```yaml
profile:
  kind: asymmetric_linear
  centre_value: 1.0
  edge_value: 0.2
  asymmetry: 0.15
  background_value: 0.0
```

## Implicit-object parameters

### Sphere

```yaml
kind: sphere
parameters:
  centre_mm: [16.0, 16.0, 16.0]
  radius_mm: 4.0
```

### Ellipsoid

```yaml
kind: ellipsoid
parameters:
  centre_mm: [16.0, 16.0, 16.0]
  axes_mm: [5.0, 3.0, 2.0]
```

### Slab or sheet

```yaml
kind: slab
parameters:
  centre_mm: [16.0, 16.0, 16.0]
  normal: [0.0, 1.0, 0.0]
  thickness_mm: 2.0
```

### Cone

```yaml
kind: cone
parameters:
  apex_mm: [16.0, 16.0, 8.0]
  axis: [0.0, 0.0, 1.0]
  height_mm: 12.0
  base_radius_mm: 4.0
```

### Frustum

```yaml
kind: frustum
parameters:
  start_mm: [16.0, 16.0, 8.0]
  axis: [0.0, 0.0, 1.0]
  height_mm: 12.0
  radius_start_mm: 2.0
  radius_end_mm: 5.0
```

Some examples place implicit-object parameters directly on the object rather than under `parameters`; the loader normalises both forms where supported.

## Rendering

A scene dictionary can be rendered with:

```python
from synthworkshop.scenes.config import render_scene_from_dict

scene = render_scene_from_dict(payload)
```

A YAML or JSON file can be rendered with:

```python
from synthworkshop.scenes.config import render_scene_config_from_path

scene = render_scene_config_from_path("examples/basic_tube.yml")
```
