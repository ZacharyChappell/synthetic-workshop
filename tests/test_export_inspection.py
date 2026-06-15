from __future__ import annotations

from pathlib import Path

import pytest

from synthworkshop.datasets import render_catalogue_scene
from synthworkshop.io import (
    export_scene,
    inspect_export_contract,
    require_export_contract,
)


def test_inspect_export_contract_passes_for_basic_export(tmp_path: Path) -> None:
    scene = render_catalogue_scene("basic_tube")
    manifest = export_scene(scene, tmp_path / "basic_tube")

    report = inspect_export_contract(manifest.output_root)

    assert report.passed
    assert report.summary_counts()["error"] == 0
    assert report.manifest_rows


def test_require_export_contract_returns_report_for_valid_export(
    tmp_path: Path,
) -> None:
    scene = render_catalogue_scene("known_effect_tube")
    manifest = export_scene(scene, tmp_path / "known_effect_tube")

    report = require_export_contract(manifest.output_root)

    assert report.passed


def test_inspect_export_contract_reports_missing_export_root(tmp_path: Path) -> None:
    report = inspect_export_contract(tmp_path / "missing_export")

    assert not report.passed
    assert report.errors()[0].location.endswith("missing_export")
    assert "does not exist" in report.errors()[0].message


def test_inspect_export_contract_reports_missing_manifest_target(
    tmp_path: Path,
) -> None:
    scene = render_catalogue_scene("basic_tube")
    manifest = export_scene(scene, tmp_path / "basic_tube")

    manifest.arrays["labels"].unlink()

    report = inspect_export_contract(manifest.output_root)

    assert not report.passed
    assert any("does not exist" in issue.message for issue in report.errors())


def test_require_export_contract_raises_for_invalid_export(tmp_path: Path) -> None:
    scene = render_catalogue_scene("basic_tube")
    manifest = export_scene(scene, tmp_path / "basic_tube")

    (manifest.output_root / "metadata" / "grid.json").unlink()

    with pytest.raises(ValueError, match="Export contract inspection failed"):
        require_export_contract(manifest.output_root)


def test_top_level_exports_export_inspection_api() -> None:
    import synthworkshop.io as io

    assert io.inspect_export_contract is inspect_export_contract
    assert io.require_export_contract is require_export_contract
