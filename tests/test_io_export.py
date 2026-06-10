from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synthworkshop import GridSpec
from synthworkshop.cross_sections import CircularCrossSection
from synthworkshop.io import (
    export_scene,
    read_array,
    read_json,
    read_table,
    safe_name,
    write_json,
    write_table,
)
from synthworkshop.primitives import LineCurve, TubeObject
from synthworkshop.profiles import LinearRadialProfile


def _small_scene():
    grid = GridSpec(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
    centreline = LineCurve(
        start_mm=(3.0, 6.0, 6.0),
        end_mm=(8.0, 6.0, 6.0),
    ).sample(step_mm=1.0, object_id="target")
    tube = TubeObject(
        object_id="target",
        centreline=centreline,
        cross_section=CircularCrossSection(radius_mm=2.0),
        profile=LinearRadialProfile(centre_value=1.0, edge_value=0.2),
        map_name="fa_like",
        role="target",
        label=1,
        priority=10,
    )
    return tube.render(grid)


def test_safe_name_sanitises_identifiers() -> None:
    assert safe_name("target:u mm") == "target_u_mm"
    assert safe_name("fa/like") == "fa_like"

    with pytest.raises(ValueError, match="empty"):
        safe_name("///")


def test_write_json_round_trip(tmp_path: Path) -> None:
    path = write_json(
        {"array": np.array([1, 2, 3]), "nonfinite": float("nan")},
        tmp_path / "metadata.json",
    )
    payload = read_json(path)

    assert payload["array"] == [1, 2, 3]
    assert payload["nonfinite"] is None


def test_write_table_round_trip_and_overwrite_guard(tmp_path: Path) -> None:
    import pandas as pd

    table = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = write_table(table, tmp_path / "table.tsv")
    loaded = read_table(path)

    assert loaded.shape == (2, 2)
    with pytest.raises(FileExistsError):
        write_table(table, path)


def test_export_scene_writes_standard_layout(tmp_path: Path) -> None:
    scene = _small_scene()
    manifest = export_scene(scene, tmp_path / "scene", overwrite=False)

    assert manifest.output_root == tmp_path / "scene"
    assert (tmp_path / "scene" / "arrays").is_dir()
    assert (tmp_path / "scene" / "tables").is_dir()
    assert (tmp_path / "scene" / "metadata").is_dir()

    assert "scalar__fa_like" in manifest.arrays
    assert "labels" in manifest.arrays
    assert "object_mask__target" in manifest.arrays
    assert "target_mask__target" in manifest.arrays
    assert "analysis_mask__analysis" in manifest.arrays
    assert "skeleton_mask__target" in manifest.arrays
    assert "distance__target" in manifest.arrays
    assert "signed_offset__target_u_mm" in manifest.arrays

    assert "centrelines" in manifest.tables
    assert "frames" in manifest.tables
    assert "export_manifest" in manifest.tables

    assert "grid" in manifest.metadata
    assert "scene_summary" in manifest.metadata
    assert "truth_summary" in manifest.metadata
    assert "export_manifest" in manifest.metadata


def test_exported_arrays_can_be_loaded(tmp_path: Path) -> None:
    scene = _small_scene()
    manifest = export_scene(scene, tmp_path / "scene")

    scalar = read_array(manifest.arrays["scalar__fa_like"])
    labels = read_array(manifest.arrays["labels"])
    mask = read_array(manifest.arrays["object_mask__target"])

    assert scalar.shape == scene.grid.shape
    assert labels.shape == scene.grid.shape
    assert mask.shape == scene.grid.shape
    assert np.isclose(scalar[6, 6, 6], 1.0)


def test_exported_metadata_and_manifest_tables_can_be_read(tmp_path: Path) -> None:
    scene = _small_scene()
    manifest = export_scene(
        scene,
        tmp_path / "scene",
        extra_metadata={"purpose": "unit_test"},
    )

    grid = read_json(manifest.metadata["grid"])
    summary = read_json(manifest.metadata["scene_summary"])
    export_manifest = read_table(manifest.tables["export_manifest"])

    assert grid["shape"] == [12, 12, 12]
    assert summary["scalar_maps"] == ["fa_like"]
    assert {"group", "role", "path"}.issubset(export_manifest.columns)


def test_export_scene_respects_overwrite_guard(tmp_path: Path) -> None:
    scene = _small_scene()
    export_scene(scene, tmp_path / "scene")

    with pytest.raises(FileExistsError):
        export_scene(scene, tmp_path / "scene")


def test_export_scene_can_skip_arrays_and_metadata(tmp_path: Path) -> None:
    scene = _small_scene()
    manifest = export_scene(
        scene,
        tmp_path / "scene",
        include_arrays=False,
        include_metadata=False,
    )

    assert manifest.arrays == {}
    assert manifest.metadata == {}
    assert "export_manifest" in manifest.tables
    assert not (tmp_path / "scene" / "arrays").exists()
    assert not (tmp_path / "scene" / "metadata").exists()


def test_top_level_exports_export_scene() -> None:
    import synthworkshop

    assert synthworkshop.export_scene is export_scene
