"""Tabular I/O helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def write_table(
    table: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
    index: bool = False,
) -> Path:
    """Write a TSV table with conservative overwrite behaviour."""

    out_path = Path(path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, sep="\t", index=index)
    return out_path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a TSV table."""

    return pd.read_csv(Path(path), sep="\t")


def write_table_mapping(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write a mapping of named tables as TSV files."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, table in tables.items():
        paths[str(name)] = write_table(
            table,
            out_dir / f"{name}.tsv",
            overwrite=overwrite,
        )
    return paths
