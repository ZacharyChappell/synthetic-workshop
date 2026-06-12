"""Curated scene catalogue for synthetic-workshop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SceneFamily = Literal[
    "control",
    "morphology",
    "environment",
    "implicit_object",
    "topology",
    "scalar_profile",
    "perturbation",
    "known_effect",
    "null_case",
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
    expected_failure_mode: str = ""
    recommended_use: str = ""
    tags: tuple[str, ...] = ()
    default_output_name: str | None = None
    seed: int | None = None
    sweep_parameters: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id)
        title = str(self.title)
        purpose = str(self.purpose)
        expected_appearance = str(self.expected_appearance)
        default_output_name = self.default_output_name or scene_id

        if not scene_id:
            raise ValueError("scene_id must be a non-empty string.")
        if not title:
            raise ValueError("title must be a non-empty string.")
        if not purpose:
            raise ValueError("purpose must be a non-empty string.")
        if not expected_appearance:
            raise ValueError("expected_appearance must be a non-empty string.")
        if not default_output_name:
            raise ValueError("default_output_name must be a non-empty string.")

        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "expected_appearance", expected_appearance)
        object.__setattr__(
            self,
            "validation_focus",
            tuple(str(item) for item in self.validation_focus),
        )
        object.__setattr__(
            self,
            "tags",
            tuple(str(item) for item in self.tags),
        )
        object.__setattr__(self, "default_output_name", str(default_output_name))
        object.__setattr__(self, "sweep_parameters", dict(self.sweep_parameters))

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
            "expected_failure_mode": self.expected_failure_mode,
            "recommended_use": self.recommended_use,
            "tags": "; ".join(self.tags),
            "default_output_name": str(self.default_output_name),
            "seed": "" if self.seed is None else str(self.seed),
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
        expected_failure_mode=(
            "Unexpected mask, label, scalar-profile, or export failure in a simple "
            "single-object scene."
        ),
        recommended_use="Use as smoke test for rendering and export workflows.",
        tags=("control", "tube", "straight", "scalar-profile"),
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
        expected_failure_mode=(
            "Frame, projection, or support-estimation errors caused by curvature "
            "and non-circular cross-sections."
        ),
        recommended_use="Use when checking curve-aware rendering and "
        "centreline/frame outputs.",
        tags=("morphology", "curved", "ellipse", "frame"),
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
        expected_failure_mode=(
            "Biased profile or width estimates when local support changes along "
            "the object."
        ),
        recommended_use="Use for width-sensitive sampling and profile-support checks.",
        tags=("morphology", "variable-width", "tube"),
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
        expected_failure_mode=(
            "Circular or isotropic sampling assumptions blur or misrepresent "
            "flattened support."
        ),
        recommended_use="Use when testing methods on anisotropic or sheet-like support",
        tags=("morphology", "ribbon", "flattened", "edge"),
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
        expected_failure_mode=(
            "Incorrect object labelling or overlap handling when target and "
            "implicit objects coexist."
        ),
        recommended_use="Use for composition, overlap-reporting, and label-map checks.",
        tags=("implicit-object", "composition", "overlap"),
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
        expected_failure_mode=(
            "Target/environment confusion or analysis-mask leakage near adjacent "
            "compartments."
        ),
        recommended_use="Use for environment, analysis-mask, and contamination checks.",
        tags=("environment", "slab", "analysis-mask", "contamination"),
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
        expected_failure_mode=(
            "Boundary or support errors for tapered non-tube analytic objects."
        ),
        recommended_use="Use when checking implicit-object rendering beyond "
        "spheres and slabs.",
        tags=("implicit-object", "cone", "frustum", "tapered"),
    ),
    CatalogueEntry(
        scene_id="perturbed_tube",
        title="Perturbed straight tube",
        family="perturbation",
        config_path=_example_path("perturbed_tube.yml"),
        purpose=(
            "Tests ordered observation-level perturbations applied to a simple "
            "straight tube scene."
        ),
        expected_appearance=(
            "A straight tube with intensity scaling and low-amplitude scalar noise."
        ),
        validation_focus=(
            "perturbation metadata",
            "ordered perturbations",
            "observation degradation",
            "seeded noise",
        ),
        expected_failure_mode=(
            "Missing perturbation provenance or incorrect ordering of observation "
            "degradation steps."
        ),
        recommended_use="Use as the minimal perturbation workflow example.",
        tags=("perturbation", "noise", "intensity-scaling", "metadata"),
        seed=123,
    ),
    CatalogueEntry(
        scene_id="known_effect_tube",
        title="Known-effect straight tube",
        family="known_effect",
        config_path=_example_path("known_effect_tube.yml"),
        purpose=(
            "Tests a localised known scalar effect inside a simple straight tube."
        ),
        expected_appearance=(
            "A straight tube with a higher-valued interval along one section of "
            "the target object."
        ),
        validation_focus=(
            "effect metadata",
            "localised scalar effect",
            "known support",
            "expected direction",
        ),
        expected_failure_mode=(
            "Failure to recover or record a known localised scalar change."
        ),
        recommended_use="Use as the minimal known-effect workflow example.",
        tags=("known-effect", "localised", "scalar-shift", "metadata"),
    ),
    CatalogueEntry(
        scene_id="near_crossing_tubes",
        title="Near-crossing target and neighbour tubes",
        family="environment",
        config_path=_example_path("near_crossing_tubes.yml"),
        purpose=(
            "Tests local proximity between a target tube and a nearby "
            "non-touching environmental tube."
        ),
        expected_appearance=(
            "Two tubes pass close to one another without forming a graph junction "
            "or shared topology."
        ),
        validation_focus=(
            "near crossing",
            "environment role",
            "local contamination",
            "target separation",
        ),
        expected_failure_mode=(
            "Sampling or support-estimation methods may leak signal from the "
            "nearby environmental tube into the target profile."
        ),
        recommended_use=(
            "Use for proximity-contamination checks where objects are close but "
            "topologically separate."
        ),
        tags=("environment", "near-crossing", "contamination", "two-tube"),
    ),
    CatalogueEntry(
        scene_id="physical_crossing_tubes",
        title="Physical crossing target and neighbour tubes",
        family="environment",
        config_path=_example_path("physical_crossing_tubes.yml"),
        purpose=(
            "Tests physically overlapping tubes that remain separate analytic "
            "objects rather than a shared graph topology."
        ),
        expected_appearance=(
            "Two perpendicular tubes cross through the same local region, with "
            "target and environmental roles kept distinct."
        ),
        validation_focus=(
            "physical crossing",
            "object overlap",
            "environment role",
            "partial-volume-like ambiguity",
        ),
        expected_failure_mode=(
            "Methods may treat physical overlap as shared topology or mix target "
            "and environmental signal at the crossing."
        ),
        recommended_use=(
            "Use for crossing-contamination checks where objects physically "
            "overlap but are not graph-connected."
        ),
        tags=("environment", "physical-crossing", "overlap", "two-tube"),
    ),
    CatalogueEntry(
        scene_id="simple_graph_tube",
        title="Simple graph tube",
        family="topology",
        config_path=_example_path("simple_graph_tube.yml"),
        purpose=(
            "Tests a small graph-defined tube with explicit edge objects, "
            "junction metadata, and graph-level truth tables."
        ),
        expected_appearance=(
            "A T-shaped target object with one trunk and two branches meeting at "
            "a central junction."
        ),
        validation_focus=(
            "graph topology",
            "junction handling",
            "edge masks",
            "graph truth tables",
        ),
        expected_failure_mode=(
            "Methods may lose branch-specific support or conflate junction "
            "overlap with a single ordinary tube."
        ),
        recommended_use=(
            "Use as the minimal topology scene for graph, branch, and junction "
            "handling checks."
        ),
        tags=("topology", "graph-tube", "junction", "branch"),
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


def catalogue_scene_ids(*, require_exists: bool = True) -> tuple[str, ...]:
    """Return built-in catalogue scene IDs."""

    return tuple(
        entry.scene_id
        for entry in list_catalogue_entries(require_exists=require_exists)
    )


def catalogue_rows(*, require_exists: bool = True) -> tuple[dict[str, str], ...]:
    """Return catalogue entries as display-friendly rows."""

    return tuple(
        entry.to_row()
        for entry in list_catalogue_entries(require_exists=require_exists)
    )


def render_catalogue_scene(scene_id: str):
    """Render one built-in catalogue scene."""

    from synthworkshop.scenes.config import render_scene_from_path

    entry = get_catalogue_entry(scene_id)
    return render_scene_from_path(entry.config_path)
