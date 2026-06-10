"""Validation helpers for scene configuration files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from synthworkshop.scenes.config import (
    SceneSpec,
    load_scene_spec,
    render_scene_from_spec,
)

IssueSeverity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class SceneValidationIssue:
    """One scene-validation issue."""

    severity: IssueSeverity
    location: str
    message: str


@dataclass(frozen=True)
class SceneValidationReport:
    """Structured validation report for one scene configuration."""

    path: Path
    scene_id: str | None
    issues: tuple[SceneValidationIssue, ...]
    rendered: bool = False

    @property
    def passed(self) -> bool:
        """Return True if no error-level issues are present."""

        return not any(issue.severity == "error" for issue in self.issues)

    def errors(self) -> tuple[SceneValidationIssue, ...]:
        """Return error-level issues."""

        return tuple(issue for issue in self.issues if issue.severity == "error")

    def warnings(self) -> tuple[SceneValidationIssue, ...]:
        """Return warning-level issues."""

        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def infos(self) -> tuple[SceneValidationIssue, ...]:
        """Return info-level messages."""

        return tuple(issue for issue in self.issues if issue.severity == "info")

    def summary_counts(self) -> dict[str, int]:
        """Return issue counts by severity."""

        counts = Counter(issue.severity for issue in self.issues)
        return {
            "error": int(counts.get("error", 0)),
            "warning": int(counts.get("warning", 0)),
            "info": int(counts.get("info", 0)),
        }

    def raise_for_errors(self) -> None:
        """Raise ValueError if the report contains errors."""

        errors = self.errors()
        if not errors:
            return

        rendered = "\n".join(
            f"[{issue.severity}] {issue.location}: {issue.message}" for issue in errors
        )
        raise ValueError(f"Scene validation failed:\n{rendered}")

    def to_rows(self) -> list[dict[str, str]]:
        """Return issue rows suitable for TSV/JSON export."""

        return [
            {
                "path": str(self.path),
                "scene_id": "" if self.scene_id is None else self.scene_id,
                "rendered": str(self.rendered).lower(),
                "severity": issue.severity,
                "location": issue.location,
                "message": issue.message,
            }
            for issue in self.issues
        ]


def validate_scene_config(
    path: str | Path,
    *,
    render: bool = False,
) -> SceneValidationReport:
    """Validate a YAML/JSON scene configuration.

    Validation has two layers:

    1. load/spec validation, which checks that the file can be parsed into a
       SceneSpec and passes dataclass/schema coercion;
    2. optional render validation, which checks that the spec can actually
       produce a RenderedScene with internally consistent arrays.
    """

    scene_path = Path(path)
    issues: list[SceneValidationIssue] = []

    if not scene_path.exists():
        return SceneValidationReport(
            path=scene_path,
            scene_id=None,
            issues=(
                SceneValidationIssue(
                    severity="error",
                    location="path",
                    message=f"Scene config does not exist: {scene_path}",
                ),
            ),
            rendered=False,
        )

    try:
        spec = load_scene_spec(scene_path)
    except Exception as exc:
        return SceneValidationReport(
            path=scene_path,
            scene_id=None,
            issues=(
                SceneValidationIssue(
                    severity="error",
                    location="load",
                    message=f"{type(exc).__name__}: {exc}",
                ),
            ),
            rendered=False,
        )

    issues.extend(_validate_loaded_spec(spec))

    rendered = False
    if render:
        try:
            scene = render_scene_from_spec(spec)
            rendered = True
        except Exception as exc:
            issues.append(
                SceneValidationIssue(
                    severity="error",
                    location="render",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            issues.extend(_validate_rendered_scene(spec, scene))

    if not issues:
        issues.append(
            SceneValidationIssue(
                severity="info",
                location="scene",
                message="Scene config passed validation.",
            )
        )

    return SceneValidationReport(
        path=scene_path,
        scene_id=spec.scene_id,
        issues=tuple(issues),
        rendered=rendered,
    )


def _validate_loaded_spec(spec: SceneSpec) -> list[SceneValidationIssue]:
    """Validate a successfully loaded SceneSpec."""

    issues: list[SceneValidationIssue] = []

    if spec.schema_version != "0.1":
        issues.append(
            SceneValidationIssue(
                severity="warning",
                location="schema_version",
                message=(
                    f"Scene uses schema_version={spec.schema_version!r}; "
                    "current documented version is '0.1'."
                ),
            )
        )

    object_ids = [obj.object_id for obj in spec.objects]
    duplicate_ids = sorted(
        object_id for object_id, count in Counter(object_ids).items() if count > 1
    )
    for object_id in duplicate_ids:
        issues.append(
            SceneValidationIssue(
                severity="error",
                location="objects",
                message=f"Duplicate object id: {object_id!r}.",
            )
        )

    labels = [obj.label for obj in spec.objects]
    duplicate_labels = sorted(
        label for label, count in Counter(labels).items() if count > 1
    )
    for label in duplicate_labels:
        issues.append(
            SceneValidationIssue(
                severity="warning",
                location="objects",
                message=(
                    f"Label {label} is used by multiple objects. This may be "
                    "intentional, but distinct objects usually benefit from "
                    "distinct labels."
                ),
            )
        )

    roles = {str(obj.role) for obj in spec.objects}
    if "target" not in roles:
        issues.append(
            SceneValidationIssue(
                severity="warning",
                location="objects.role",
                message="Scene contains no object with role 'target'.",
            )
        )

    map_names = sorted({obj.map_name for obj in spec.objects})
    issues.append(
        SceneValidationIssue(
            severity="info",
            location="objects.map_name",
            message=f"Scene defines scalar map(s): {', '.join(map_names)}.",
        )
    )

    if len(spec.objects) > 1 and str(spec.composition.overlap_policy.value) == "allow":
        issues.append(
            SceneValidationIssue(
                severity="warning",
                location="composition.overlap_policy",
                message=(
                    "overlap_policy is 'allow' for a multi-object scene. "
                    "Use 'warn' or 'error' if overlaps should be surfaced."
                ),
            )
        )

    return issues


def _validate_rendered_scene(spec: SceneSpec, scene) -> list[SceneValidationIssue]:
    """Validate a RenderedScene produced from a SceneSpec."""

    issues: list[SceneValidationIssue] = []
    expected_shape = scene.grid.shape

    if scene.label_map.shape != expected_shape:
        issues.append(
            SceneValidationIssue(
                severity="error",
                location="render.label_map",
                message=(
                    f"label_map shape {scene.label_map.shape} does not match "
                    f"grid shape {expected_shape}."
                ),
            )
        )

    if not scene.scalar_maps:
        issues.append(
            SceneValidationIssue(
                severity="error",
                location="render.scalar_maps",
                message="Rendered scene contains no scalar maps.",
            )
        )

    for map_name, scalar in scene.scalar_maps.items():
        if scalar.shape != expected_shape:
            issues.append(
                SceneValidationIssue(
                    severity="error",
                    location=f"render.scalar_maps.{map_name}",
                    message=(
                        f"Scalar map shape {scalar.shape} does not match "
                        f"grid shape {expected_shape}."
                    ),
                )
            )

    expected_object_ids = {obj.object_id for obj in spec.objects}
    rendered_object_ids = set(scene.object_masks)
    missing_masks = sorted(expected_object_ids - rendered_object_ids)
    for object_id in missing_masks:
        issues.append(
            SceneValidationIssue(
                severity="error",
                location="render.object_masks",
                message=f"Rendered scene is missing object mask for {object_id!r}.",
            )
        )

    for object_id, mask in scene.object_masks.items():
        if mask.shape != expected_shape:
            issues.append(
                SceneValidationIssue(
                    severity="error",
                    location=f"render.object_masks.{object_id}",
                    message=(
                        f"Object mask shape {mask.shape} does not match "
                        f"grid shape {expected_shape}."
                    ),
                )
            )

    issues.append(
        SceneValidationIssue(
            severity="info",
            location="render",
            message=(
                f"Rendered scene has {len(scene.scalar_maps)} scalar map(s), "
                f"{len(scene.object_masks)} object mask(s), and grid shape "
                f"{expected_shape}."
            ),
        )
    )

    return issues
