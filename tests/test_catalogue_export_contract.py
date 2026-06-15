from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from synthworkshop.datasets import list_catalogue_entries, render_catalogue_scene
from synthworkshop.io import export_scene, read_json, read_table

CATALOGUE_SCENE_IDS = tuple(entry.scene_id for entry in list_catalogue_entries())

EXPECTED_CONTRACT_TABLES = {
    "object_table",
    "scene_manifest",
    "scalar_map_manifest",
    "distance_map_manifest",
    "perturbation_table",
    "effect_table",
    "export_manifest",
}

EXPECTED_METADATA_FILES = {
    "grid",
    "scene_summary",
    "truth_summary",
    "render_metadata",
    "composition_metadata",
    "overlap_report",
    "provenance",
    "export_manifest",
}


@pytest.mark.parametrize("scene_id", CATALOGUE_SCENE_IDS)
def test_catalogue_scene_exports_standard_contract(
    scene_id: str,
    tmp_path: Path,
) -> None:
    scene = render_catalogue_scene(scene_id)

    manifest = export_scene(
        scene,
        tmp_path / scene_id,
        overwrite=False,
    )

    assert "labels" in manifest.arrays
    assert any(role.startswith("scalar__") for role in manifest.arrays)
    assert any(role.startswith("object_mask__") for role in manifest.arrays)
    assert any(role.startswith("target_mask__") for role in manifest.arrays)
    assert any(role.startswith("analysis_mask__") for role in manifest.arrays)

    assert EXPECTED_CONTRACT_TABLES.issubset(manifest.tables)
    assert EXPECTED_METADATA_FILES.issubset(manifest.metadata)

    for path in (
        tuple(manifest.arrays.values())
        + tuple(manifest.tables.values())
        + tuple(manifest.metadata.values())
    ):
        assert path.exists(), path


@pytest.mark.parametrize("scene_id", CATALOGUE_SCENE_IDS)
def test_catalogue_export_manifest_paths_are_relative_and_existing(
    scene_id: str,
    tmp_path: Path,
) -> None:
    scene = render_catalogue_scene(scene_id)
    manifest = export_scene(scene, tmp_path / scene_id)

    export_manifest = read_table(manifest.tables["export_manifest"])

    assert {"group", "role", "path", "relative_path", "format"}.issubset(
        export_manifest.columns
    )
    assert export_manifest["relative_path"].notna().all()

    for relative_path in export_manifest["relative_path"]:
        path = Path(str(relative_path))
        assert not path.is_absolute()
        assert (manifest.output_root / path).exists(), relative_path


@pytest.mark.parametrize("scene_id", CATALOGUE_SCENE_IDS)
def test_catalogue_export_contract_tables_are_consistent(
    scene_id: str,
    tmp_path: Path,
) -> None:
    scene = render_catalogue_scene(scene_id)
    manifest = export_scene(scene, tmp_path / scene_id)

    object_table = read_table(manifest.tables["object_table"])
    scene_manifest = read_table(manifest.tables["scene_manifest"])
    scalar_manifest = read_table(manifest.tables["scalar_map_manifest"])
    distance_manifest = read_table(manifest.tables["distance_map_manifest"])

    assert len(object_table) == len(scene.object_masks)
    assert set(object_table["object_id"]) == set(scene.object_masks)

    assert set(scene_manifest["field"]).issuperset(
        {
            "scene_id",
            "shape",
            "spacing",
            "scalar_maps",
            "object_ids",
            "n_objects",
            "n_scalar_maps",
            "overlap_voxels",
            "overlap_policy",
        }
    )

    assert len(scalar_manifest) == len(scene.scalar_maps)
    assert set(scalar_manifest["map_name"]) == set(scene.scalar_maps)

    expected_distance_rows = len(scene.distance_maps) + len(scene.signed_offset_maps)
    assert len(distance_manifest) == expected_distance_rows


def test_export_manifest_json_includes_relative_paths(tmp_path: Path) -> None:
    scene = render_catalogue_scene("basic_tube")
    manifest = export_scene(scene, tmp_path / "basic_tube")

    payload = read_json(manifest.metadata["export_manifest"])

    assert payload["relative_paths"]["arrays"]["labels"] == "arrays/labels.npy"
    assert (
        payload["relative_paths"]["tables"]["object_table"] == "tables/object_table.tsv"
    )
    assert payload["relative_paths"]["metadata"]["grid"] == "metadata/grid.json"

    files = pd.DataFrame(payload["files"])
    assert {"group", "role", "path", "relative_path", "format"}.issubset(files.columns)
    assert "arrays/labels.npy" in set(files["relative_path"])
    assert "tables/export_manifest.tsv" in set(files["relative_path"])
