"""Curated scene catalogue for synthetic-workshop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SceneFamily = Literal[
    "control",
    "morphology",
    "environment",
    "implicit_object",
    "topology",
    "scalar_profile",
    "stress",
]


@dataclass(frozen=True)
class CatalogueEntry:
    """Metadata for one curated scene configuration."""

    scene_id: str
    title: str
    family: SceneFamily
    config_path: Path
    purpose: str
    expected_appearance: str
    validation_focus: tuple[str, ...] = ()
    notes: str = ""

    def to_row(self) -> dict[str, str]:
        """Return a compact row suitable for CLI display or TSV export."""

        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "family": self.family,
            "config_path": str(self.config_path),
            "purpose": self.purpose,
            "expected_appearance": self.expected_appearance,
            "validation_focus": "; ".join(self.validation_focus),
            "notes": self.notes,
        }


def repository_root() -> Path:
    """Return the repository root for an editable/source checkout."""

    return Path(__file__).resolve().parents[3]


def examples_dir() -> Path:
    """Return the default examples directory."""

    return repository_root() / "examples"


def _example_path(name: str) -> Path:
    """Return the path to a bundled example scene config."""

    return examples_dir() / name


CATALOGUE: tuple[CatalogueEntry, ...] = (
    CatalogueEntry(
        scene_id="basic_tube",
        title="Basic straight circular tube",
        family="control",
        config_path=_example_path("basic_tube.yml"),
        purpose=(
            "Baseline sanity-check scene with a single target tube, known support, "
            "known label, and simple radial scalar field."
        ),
        expected_appearance=(
            "A straight cylindrical object with a smooth radial scalar profile."
        ),
        validation_focus=(
            "grid handling",
            "tube rendering",
            "target mask",
            "scalar profile",
            "export smoke test",
        ),
    ),
    CatalogueEntry(
        scene_id="curved_elliptic_tube",
        title="Curved elliptic target tube",
        family="morphology",
        config_path=_example_path("curved_elliptic_tube.yml"),
        purpose=(
            "Tests curved centreline geometry, elliptic cross-sections, and "
            "non-circular support."
        ),
        expected_appearance=(
            "A curved tube with an elongated cross-section and smooth scalar "
            "variation within the target object."
        ),
        validation_focus=(
            "curved centreline",
            "elliptic cross-section",
            "local frame behaviour",
            "projection visualisation",
        ),
    ),
    CatalogueEntry(
        scene_id="variable_radius_tube",
        title="Variable-radius tube",
        family="morphology",
        config_path=_example_path("variable_radius_tube.yml"),
        purpose=(
            "Tests changing tube width along the centreline and width-sensitive "
            "downstream estimands."
        ),
        expected_appearance=(
            "A tube whose apparent radius expands and contracts along its length."
        ),
        validation_focus=(
            "variable width",
            "radius metadata",
            "profile support",
            "width recovery",
        ),
    ),
    CatalogueEntry(
        scene_id="ribbon_tube",
        title="Ribbon-like flattened tube",
        family="morphology",
        config_path=_example_path("ribbon_tube.yml"),
        purpose=(
            "Tests flattened or sheet-like tract morphology where circular "
            "sampling assumptions may be inadequate."
        ),
        expected_appearance=(
            "A flattened ribbon-shaped object rather than a circular tube."
        ),
        validation_focus=(
            "ribbon cross-section",
            "flattened morphology",
            "edge ambiguity",
            "projection QC",
        ),
    ),
    CatalogueEntry(
        scene_id="tube_with_implicit_objects",
        title="Tube with implicit inclusions",
        family="implicit_object",
        config_path=_example_path("tube_with_implicit_objects.yml"),
        purpose=(
            "Tests composition of a target tube with independent implicit objects "
            "such as inclusions or neighbouring structures."
        ),
        expected_appearance=(
            "A target tube with additional compact analytic objects visible in "
            "the scalar or label maps."
        ),
        validation_focus=(
            "implicit objects",
            "object labels",
            "composition",
            "overlap reporting",
        ),
    ),
    CatalogueEntry(
        scene_id="tube_with_slab_environment",
        title="Tube with slab environment",
        family="environment",
        config_path=_example_path("tube_with_slab_environment.yml"),
        purpose=(
            "Tests target/environment separation and analysis-mask semantics in "
            "the presence of an adjacent sheet-like compartment."
        ),
        expected_appearance=(
            "A target tube close to or intersecting a slab-like environmental "
            "structure."
        ),
        validation_focus=(
            "environment role",
            "analysis mask",
            "partial-volume-like contamination",
            "object overlap",
        ),
    ),
    CatalogueEntry(
        scene_id="cone_frustum_scene",
        title="Cone and frustum implicit objects",
        family="implicit_object",
        config_path=_example_path("cone_frustum_scene.yml"),
        purpose=(
            "Tests non-tube implicit geometry through cone and frustum objects "
            "with known analytic support."
        ),
        expected_appearance=(
            "One or more tapered implicit objects with clear geometric boundaries."
        ),
        validation_focus=(
            "cone geometry",
            "frustum geometry",
            "implicit support",
            "label composition",
        ),
    ),
)


def iter_catalogue_entries() -> tuple[CatalogueEntry, ...]:
    """Return all built-in catalogue entries."""

    return CATALOGUE


def list_catalogue_entries(
    *, require_exists: bool = True
) -> tuple[CatalogueEntry, ...]:
    """Return catalogue entries, optionally checking that config files exist."""

    entries = iter_catalogue_entries()

    if require_exists:
        missing = [
            entry.config_path for entry in entries if not entry.config_path.exists()
        ]
        if missing:
            rendered = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Catalogue config file(s) are missing:\n{rendered}"
            )

    return entries


def get_catalogue_entry(
    scene_id: str, *, require_exists: bool = True
) -> CatalogueEntry:
    """Return one catalogue entry by scene ID."""

    matches = [
        entry for entry in iter_catalogue_entries() if entry.scene_id == scene_id
    ]
    if not matches:
        available = ", ".join(entry.scene_id for entry in iter_catalogue_entries())
        raise KeyError(
            f"Unknown catalogue scene_id {scene_id!r}. Available: {available}"
        )

    entry = matches[0]
    if require_exists and not entry.config_path.exists():
        raise FileNotFoundError(
            f"Catalogue config file is missing: {entry.config_path}"
        )

    return entry
