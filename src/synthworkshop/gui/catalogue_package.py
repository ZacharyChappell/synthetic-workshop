"""Export/import helpers for GUI-authored catalogue scene packages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthworkshop.gui.summary import build_scene_summary, summary_to_json
from synthworkshop.gui.yaml_editor import parse_scene_text

CATALOGUE_FAMILIES: tuple[str, ...] = (
    "control",
    "morphology",
    "environment",
    "implicit_object",
    "topology",
    "scalar_profile",
    "stress",
)


@dataclass(frozen=True)
class CataloguePackage:
    """A reusable GUI-authored scene package."""

    package_dir: Path
    scene_path: Path
    readme_path: Path
    entry_path: Path
    summary_path: Path
    scene_text: str
    entry: dict[str, Any]
    summary: dict[str, Any]

    def path_rows(self) -> list[dict[str, str]]:
        """Return package paths as display rows."""

        return [
            {"file": "scene", "path": str(self.scene_path)},
            {"file": "README", "path": str(self.readme_path)},
            {"file": "catalogue_entry", "path": str(self.entry_path)},
            {"file": "scene_summary", "path": str(self.summary_path)},
        ]


def export_catalogue_package(
    *,
    text: str,
    package_dir: str | Path,
    title: str,
    family: str,
    purpose: str,
    expected_appearance: str,
    validation_focus: str | list[str] | tuple[str, ...] = (),
    notes: str = "",
    overwrite: bool = False,
    render_summary: bool = False,
) -> CataloguePackage:
    """Export current scene YAML as a reusable catalogue-entry package."""

    if family not in CATALOGUE_FAMILIES:
        raise ValueError(f"Unknown catalogue family: {family!r}.")

    payload = parse_scene_text(text)
    scene_id = scene_id_from_payload(payload)
    if not scene_id:
        raise ValueError("Scene must have scene.id before catalogue export.")

    target_dir = Path(package_dir)
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty package directory: {target_dir}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = target_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    scene_path = target_dir / "scene.yml"
    readme_path = target_dir / "README.md"
    entry_path = metadata_dir / "catalogue_entry.json"
    summary_path = metadata_dir / "scene_summary.json"

    focus_items = parse_focus_items(validation_focus)
    summary = build_scene_summary(text, render=render_summary)

    entry = {
        "scene_id": scene_id,
        "title": title.strip() or humanise_scene_id(scene_id),
        "family": family,
        "config_path": "scene.yml",
        "purpose": purpose.strip(),
        "expected_appearance": expected_appearance.strip(),
        "validation_focus": focus_items,
        "notes": notes.strip(),
    }

    scene_path.write_text(text, encoding="utf-8")
    entry_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(summary_to_json(summary) + "\n", encoding="utf-8")
    readme_path.write_text(
        catalogue_package_readme(entry=entry, summary=summary),
        encoding="utf-8",
    )

    return CataloguePackage(
        package_dir=target_dir,
        scene_path=scene_path,
        readme_path=readme_path,
        entry_path=entry_path,
        summary_path=summary_path,
        scene_text=text,
        entry=entry,
        summary=summary,
    )


def read_catalogue_package(package_dir: str | Path) -> CataloguePackage:
    """Read a previously exported catalogue scene package."""

    root = Path(package_dir)
    scene_path = root / "scene.yml"
    readme_path = root / "README.md"
    entry_path = root / "metadata" / "catalogue_entry.json"
    summary_path = root / "metadata" / "scene_summary.json"

    if not scene_path.exists():
        raise FileNotFoundError(f"Package scene file is missing: {scene_path}")

    scene_text = scene_path.read_text(encoding="utf-8")
    parse_scene_text(scene_text)

    if entry_path.exists():
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
    else:
        payload = parse_scene_text(scene_text)
        scene_id = scene_id_from_payload(payload)
        entry = {
            "scene_id": scene_id,
            "title": humanise_scene_id(scene_id),
            "family": "control",
            "config_path": "scene.yml",
            "purpose": "",
            "expected_appearance": "",
            "validation_focus": [],
            "notes": "",
        }

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = build_scene_summary(scene_text, render=False)

    return CataloguePackage(
        package_dir=root,
        scene_path=scene_path,
        readme_path=readme_path,
        entry_path=entry_path,
        summary_path=summary_path,
        scene_text=scene_text,
        entry=entry,
        summary=summary,
    )


def default_catalogue_package_dir(
    *,
    output_root: str | Path,
    scene_id: str,
) -> Path:
    """Return the default output directory for a GUI catalogue package."""

    return Path(output_root) / "catalogue_entries" / safe_slug(scene_id)


def scene_id_from_text(text: str) -> str:
    """Return scene.id from YAML text."""

    return scene_id_from_payload(parse_scene_text(text))


def scene_id_from_payload(payload: dict[str, Any]) -> str:
    """Return scene.id from a parsed scene payload."""

    scene = payload.get("scene", {})
    if not isinstance(scene, dict):
        return ""
    value = scene.get("id", "")
    return str(value).strip()


def humanise_scene_id(scene_id: str) -> str:
    """Create a readable title from a scene ID."""

    words = scene_id.replace("-", "_").split("_")
    return " ".join(word.capitalize() for word in words if word)


def safe_slug(value: str) -> str:
    """Return a filesystem-safe slug."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or "scene"


def parse_focus_items(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse validation-focus items from text or a sequence."""

    if isinstance(value, str):
        raw = value.replace("\n", ",").split(",")
        return [item.strip() for item in raw if item.strip()]

    return [str(item).strip() for item in value if str(item).strip()]


def format_focus_items(items: list[str] | tuple[str, ...]) -> str:
    """Format focus items for GUI text input."""

    return ", ".join(str(item) for item in items)


def catalogue_package_readme(
    *,
    entry: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    """Return README text for a catalogue package."""

    focus = entry.get("validation_focus", [])
    focus_lines = (
        "\n".join(f"- {item}" for item in focus) if focus else "- Not specified."
    )

    scene_id = str(entry.get("scene_id", ""))
    title = str(entry.get("title", humanise_scene_id(scene_id)))

    n_objects = summary.get("objects", {}).get("n_objects", "unknown")
    map_names = summary.get("objects", {}).get("map_names", [])
    map_text = ", ".join(map_names) if map_names else "none recorded"

    return f"""# {title}

## Catalogue metadata

- Scene ID: `{scene_id}`
- Family: `{entry.get("family", "")}`
- Config path: `{entry.get("config_path", "scene.yml")}`
- Objects: `{n_objects}`
- Scalar maps: `{map_text}`

## Purpose

{entry.get("purpose", "") or "Not specified."}

## Expected appearance

{entry.get("expected_appearance", "") or "Not specified."}

## Validation focus

{focus_lines}

## Notes

{entry.get("notes", "") or "None."}

## Files

```text
scene.yml
README.md
metadata/catalogue_entry.json
metadata/scene_summary.json
```
This package was exported from the optional synthetic-workshop GUI. Review it before
promoting it into the built-in examples or catalogue.
"""
