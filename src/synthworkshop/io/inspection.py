"""Inspect exported scene contract files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from synthworkshop.io.json import read_json
from synthworkshop.io.tables import read_table

IssueSeverity = Literal["error", "warning", "info"]

EXPECTED_EXPORT_DIRS = ("arrays", "tables", "metadata")

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

EXPECTED_MANIFEST_COLUMNS = {
    "group",
    "role",
    "path",
    "relative_path",
    "format",
}


@dataclass(frozen=True)
class ExportInspectionIssue:
    """One export-contract inspection issue."""

    severity: IssueSeverity
    location: str
    message: str


@dataclass(frozen=True)
class ExportInspectionReport:
    """Result of inspecting an exported scene directory."""

    export_root: Path
    issues: tuple[ExportInspectionIssue, ...] = field(default_factory=tuple)
    manifest_rows: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Whether the inspected export has no errors."""

        return not self.errors()

    def errors(self) -> tuple[ExportInspectionIssue, ...]:
        """Return error issues."""

        return tuple(issue for issue in self.issues if issue.severity == "error")

    def warnings(self) -> tuple[ExportInspectionIssue, ...]:
        """Return warning issues."""

        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def summary_counts(self) -> dict[str, int]:
        """Return counts by severity."""

        return {
            "error": len(self.errors()),
            "warning": len(self.warnings()),
            "info": sum(issue.severity == "info" for issue in self.issues),
        }


def inspect_export_contract(
    export_root: str | Path,
    *,
    require_arrays: bool = True,
    require_tables: bool = True,
    require_metadata: bool = True,
) -> ExportInspectionReport:
    """Inspect a scene export directory for the standard contract."""

    root = Path(export_root)
    issues: list[ExportInspectionIssue] = []
    manifest_rows: tuple[Mapping[str, object], ...] = ()

    if not root.exists():
        return ExportInspectionReport(
            export_root=root,
            issues=(
                ExportInspectionIssue(
                    severity="error",
                    location=str(root),
                    message="Export root does not exist.",
                ),
            ),
        )

    if not root.is_dir():
        return ExportInspectionReport(
            export_root=root,
            issues=(
                ExportInspectionIssue(
                    severity="error",
                    location=str(root),
                    message="Export root is not a directory.",
                ),
            ),
        )

    for dirname in EXPECTED_EXPORT_DIRS:
        if not (root / dirname).is_dir():
            issues.append(
                ExportInspectionIssue(
                    severity="error",
                    location=dirname,
                    message="Expected export subdirectory is missing.",
                )
            )

    tsv_path = root / "tables" / "export_manifest.tsv"
    json_path = root / "metadata" / "export_manifest.json"

    if not tsv_path.exists():
        issues.append(
            ExportInspectionIssue(
                severity="error",
                location="tables/export_manifest.tsv",
                message="Export manifest TSV is missing.",
            )
        )
        manifest_table = pd.DataFrame()
    else:
        try:
            manifest_table = read_table(tsv_path)
        except Exception as error:  # pragma: no cover - defensive
            issues.append(
                ExportInspectionIssue(
                    severity="error",
                    location="tables/export_manifest.tsv",
                    message=f"Could not read export manifest TSV: {error}",
                )
            )
            manifest_table = pd.DataFrame()

    if not json_path.exists():
        issues.append(
            ExportInspectionIssue(
                severity="error",
                location="metadata/export_manifest.json",
                message="Export manifest JSON is missing.",
            )
        )
        manifest_json: Mapping[str, object] = {}
    else:
        try:
            payload = read_json(json_path)
            manifest_json = payload if isinstance(payload, Mapping) else {}
        except Exception as error:  # pragma: no cover - defensive
            issues.append(
                ExportInspectionIssue(
                    severity="error",
                    location="metadata/export_manifest.json",
                    message=f"Could not read export manifest JSON: {error}",
                )
            )
            manifest_json = {}

    if not manifest_table.empty:
        issues.extend(_inspect_manifest_table(root, manifest_table))
        manifest_rows = tuple(manifest_table.to_dict(orient="records"))

    if manifest_json:
        issues.extend(_inspect_manifest_json(root, manifest_json))

    if require_arrays:
        issues.extend(_inspect_required_array_entries(root, manifest_table))

    if require_tables:
        issues.extend(_inspect_required_table_entries(root, manifest_table))

    if require_metadata:
        issues.extend(_inspect_required_metadata_entries(root, manifest_table))

    return ExportInspectionReport(
        export_root=root,
        issues=tuple(issues),
        manifest_rows=manifest_rows,
    )


def require_export_contract(export_root: str | Path) -> ExportInspectionReport:
    """Inspect an export directory and raise ValueError if the contract fails."""

    report = inspect_export_contract(export_root)
    if report.passed:
        return report

    rendered = "\n".join(
        f"[{issue.severity}] {issue.location}: {issue.message}"
        for issue in report.errors()
    )
    raise ValueError(f"Export contract inspection failed:\n{rendered}")


def _inspect_manifest_table(
    root: Path,
    manifest_table: pd.DataFrame,
) -> list[ExportInspectionIssue]:
    issues: list[ExportInspectionIssue] = []

    missing_columns = EXPECTED_MANIFEST_COLUMNS - set(manifest_table.columns)
    for column in sorted(missing_columns):
        issues.append(
            ExportInspectionIssue(
                severity="error",
                location="tables/export_manifest.tsv",
                message=f"Manifest column is missing: {column}",
            )
        )

    if missing_columns:
        return issues

    for index, row in manifest_table.iterrows():
        location = f"tables/export_manifest.tsv:{index + 2}"
        relative_path = Path(str(row["relative_path"]))
        path = Path(str(row["path"]))

        if relative_path.is_absolute():
            issues.append(
                ExportInspectionIssue(
                    severity="error",
                    location=location,
                    message="relative_path must not be absolute.",
                )
            )
            continue

        if path.is_absolute():
            issues.append(
                ExportInspectionIssue(
                    severity="error",
                    location=location,
                    message="path must not be absolute in exported manifests.",
                )
            )

        if path != relative_path:
            issues.append(
                ExportInspectionIssue(
                    severity="warning",
                    location=location,
                    message="path and relative_path differ.",
                )
            )

        if not (root / relative_path).exists():
            issues.append(
                ExportInspectionIssue(
                    severity="error",
                    location=location,
                    message=f"Manifest target does not exist: {relative_path}",
                )
            )

    return issues


def _inspect_manifest_json(
    root: Path,
    manifest_json: Mapping[str, object],
) -> list[ExportInspectionIssue]:
    issues: list[ExportInspectionIssue] = []

    if manifest_json.get("path_mode") != "relative_to_export_root":
        issues.append(
            ExportInspectionIssue(
                severity="error",
                location="metadata/export_manifest.json",
                message="path_mode must be 'relative_to_export_root'.",
            )
        )

    if manifest_json.get("output_root") != ".":
        issues.append(
            ExportInspectionIssue(
                severity="error",
                location="metadata/export_manifest.json",
                message="output_root must be '.'.",
            )
        )

    for group_name in ("arrays", "tables", "metadata"):
        group = manifest_json.get(group_name)
        if not isinstance(group, Mapping):
            issues.append(
                ExportInspectionIssue(
                    severity="error",
                    location=f"metadata/export_manifest.json:{group_name}",
                    message="Manifest group must be a mapping.",
                )
            )
            continue

        for role, value in group.items():
            relative_path = Path(str(value))
            location = f"metadata/export_manifest.json:{group_name}.{role}"

            if relative_path.is_absolute():
                issues.append(
                    ExportInspectionIssue(
                        severity="error",
                        location=location,
                        message="Manifest path must not be absolute.",
                    )
                )
                continue

            if not (root / relative_path).exists():
                issues.append(
                    ExportInspectionIssue(
                        severity="error",
                        location=location,
                        message=f"Manifest target does not exist: {relative_path}",
                    )
                )

    return issues


def _inspect_required_array_entries(
    root: Path,
    manifest_table: pd.DataFrame,
) -> list[ExportInspectionIssue]:
    if manifest_table.empty or "group" not in manifest_table:
        return []

    rows = manifest_table[manifest_table["group"] == "array"]
    roles = {str(role) for role in rows["role"]}

    issues: list[ExportInspectionIssue] = []
    if "labels" not in roles:
        issues.append(_missing_manifest_entry("array", "labels"))

    if not any(role.startswith("scalar__") for role in roles):
        issues.append(_missing_manifest_prefix("array", "scalar__"))

    if not any(role.startswith("object_mask__") for role in roles):
        issues.append(_missing_manifest_prefix("array", "object_mask__"))

    if not any(role.startswith("target_mask__") for role in roles):
        issues.append(_missing_manifest_prefix("array", "target_mask__"))

    if not any(role.startswith("analysis_mask__") for role in roles):
        issues.append(_missing_manifest_prefix("array", "analysis_mask__"))

    return issues


def _inspect_required_table_entries(
    root: Path,
    manifest_table: pd.DataFrame,
) -> list[ExportInspectionIssue]:
    if manifest_table.empty or "group" not in manifest_table:
        return []

    rows = manifest_table[manifest_table["group"] == "table"]
    roles = {str(role) for role in rows["role"]}

    expected_roles = EXPECTED_CONTRACT_TABLES - {"export_manifest"}
    return [
        _missing_manifest_entry("table", role)
        for role in sorted(expected_roles - roles)
    ]


def _inspect_required_metadata_entries(
    root: Path,
    manifest_table: pd.DataFrame,
) -> list[ExportInspectionIssue]:
    if manifest_table.empty or "group" not in manifest_table:
        return []

    rows = manifest_table[manifest_table["group"] == "metadata"]
    roles = {str(role) for role in rows["role"]}

    expected_roles = EXPECTED_METADATA_FILES - {"export_manifest"}
    return [
        _missing_manifest_entry("metadata", role)
        for role in sorted(expected_roles - roles)
    ]


def _missing_manifest_entry(group: str, role: str) -> ExportInspectionIssue:
    return ExportInspectionIssue(
        severity="error",
        location=f"export_manifest:{group}.{role}",
        message="Expected manifest entry is missing.",
    )


def _missing_manifest_prefix(group: str, prefix: str) -> ExportInspectionIssue:
    return ExportInspectionIssue(
        severity="error",
        location=f"export_manifest:{group}.{prefix}",
        message="Expected manifest entry prefix is missing.",
    )
