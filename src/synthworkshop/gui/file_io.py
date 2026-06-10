"""Scene file load/save helpers for the optional GUI workbench."""

from __future__ import annotations

from pathlib import Path

from synthworkshop.gui.yaml_editor import parse_scene_text


def decode_uploaded_scene_bytes(data: bytes) -> str:
    """Decode uploaded scene bytes as UTF-8 text and validate top-level YAML."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Uploaded scene file must be UTF-8 text.") from exc

    parse_scene_text(text)
    return text


def read_scene_text_file(path: str | Path) -> str:
    """Read a scene YAML/JSON text file from disk."""

    scene_path = Path(path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file does not exist: {scene_path}")
    if not scene_path.is_file():
        raise ValueError(f"Scene path is not a file: {scene_path}")

    text = scene_path.read_text(encoding="utf-8")
    parse_scene_text(text)
    return text


def save_scene_text_file(
    text: str,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save scene YAML/JSON text to disk after lightweight validation."""

    parse_scene_text(text)

    scene_path = Path(path)
    if scene_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing scene file: {scene_path}"
        )

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(text, encoding="utf-8")
    return scene_path


def default_saved_scene_path(
    *,
    output_root: str | Path,
    scene_id: str,
) -> Path:
    """Return the default GUI save path for the current scene text."""

    return Path(output_root) / "scenes" / f"{scene_id}.yml"
