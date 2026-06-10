"""Plotting style and figure-saving helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_figure(
    fig: Figure,
    output_dir: str | Path,
    *,
    stem: str,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    overwrite: bool = False,
) -> list[Path]:
    """Save a Matplotlib figure in one or more formats."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for fmt in formats:
        clean_fmt = str(fmt).lower().lstrip(".")
        if not clean_fmt:
            raise ValueError("Figure format cannot be empty.")
        path = out_dir / f"{stem}.{clean_fmt}"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing figure: {path}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)

    return paths


def close_figure(fig: Figure) -> None:
    """Close a Matplotlib figure."""

    plt.close(fig)


def clean_title(value: object) -> str:
    """Create a readable title from an identifier."""

    text = str(value).replace("_", " ").strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:]
