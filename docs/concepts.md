# Concepts

`synthetic-workshop` represents synthetic images as analytic scenes rendered onto regular 2D or 3D grids. The main aim is to make validation data with known mathematical structure, rather than visually realistic images.

## Grid

A grid defines the image domain. It stores the array shape, voxel spacing, origin, axis names, and coordinate transforms used when evaluating analytic objects.

The default coordinate system is synthetic. Axis names such as `i`, `j`, and `k` describe array axes; they do not imply anatomical orientation.

## Scene

A scene is a collection of analytic objects rendered onto the same grid. Each object has a stable identifier, a semantic role, a label value, a scalar map name, geometry, a scalar profile, and optional metadata.

Scenes are usually specified in YAML or JSON, then rendered into a `RenderedScene`.

## Objects

Objects are analytic entities that can be evaluated on a grid. Current scene specifications support tube objects and several implicit objects.

Tube objects combine a centreline curve, a cross-section model, and a scalar profile. They are useful for tract-like, vessel-like, fibre-like, or elongated validation structures.

Implicit objects are defined directly by spatial support. Current examples include spheres, ellipsoids, slabs, cones, and frusta. These are useful for inclusions, neighbouring structures, sheet-like compartments, and simple analytic environments.

## Roles

Object roles describe how rendered masks are interpreted. The common roles are:

- `target`: the main object of interest;
- `analysis_support`: an object included in the default analysis mask;
- `environment`: neighbouring structure or background compartment;
- `distractor`: an object present in the scene but not part of the target;
- `inclusion`: a focal compartment or embedded object;
- `background`: a background object or compartment.

Target masks are derived from target objects. Analysis masks are derived from target and analysis-support objects unless the scene specifies different mask rules.

## Scalar profiles

Scalar profiles define the value field inside an object. Examples include constant, linear radial, Gaussian radial, edge-enhanced, asymmetric, hollow-core, sigmoid-boundary, longitudinal-gradient, multi-peak, one-sided lesion-like, and periodic longitudinal profiles.

These profiles are analytic validation fields. MRI-like names can be used for convenience, but they do not imply acquisition simulation.

## Composition

When multiple objects occupy the same voxel, composition rules determine how labels and scalar values are combined. The scene model records overlap information so that multi-object interactions remain inspectable.

Common composition settings are:

- `label_mode`: how labels are chosen in overlapping regions;
- `scalar_blend`: how scalar values are combined;
- `overlap_policy`: whether overlaps are allowed, warned about, or rejected.

## Ground truth

Rendered scenes can contain method-agnostic truth: object masks, labels, scalar maps, centrelines, local frames, distances, signed offsets, composition metadata, and provenance.

The package does not define downstream method-specific success criteria. Downstream packages can compare their own estimates with the exported truth and metadata.
