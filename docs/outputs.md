# Outputs

Rendered scenes can be exported as arrays, tables, and metadata. The export layout is designed to be predictable and easy to inspect.

A typical export has three directories:

```text
arrays/
tables/
metadata/
```

## Arrays

The `arrays/` directory contains NumPy arrays.

Common array outputs include:

- scalar maps;
- label maps;
- object masks;
- target masks;
- analysis masks;
- skeleton masks;
- distance maps;
- signed offset maps.

Array names are derived from the scene content. For example, a scalar map named `scalar` may be exported as a NumPy array under `arrays/`.

## Tables

The `tables/` directory contains tabular metadata in TSV format.

Common table outputs include:

- object tables;
- scene manifests;
- centrelines;
- frames;
- scalar-map manifests;
- distance-map manifests;
- export manifests.

The exact set of tables depends on the rendered scene.

## Metadata

The `metadata/` directory contains JSON files.

Common metadata outputs include:

- grid metadata;
- scene summaries;
- truth summaries;
- render metadata;
- composition metadata;
- overlap reports;
- provenance;
- export manifests.

## Masks

Object masks describe the support of each rendered object. Target and analysis masks are derived from object roles and mask rules.

By default, target masks represent primary objects of interest. Analysis masks can include both target objects and analysis-support objects.

## Labels

The label map is an integer array. Label values come from object specifications. When objects overlap, the scene composition rules determine which label is visible in the final label map.

The overlap report records object-mask intersections so that overlapping regions can be inspected.

## Current export format

The current public export format is NumPy for arrays, TSV for tables, and JSON for metadata.

NIfTI export is not part of the current documented public interface.
