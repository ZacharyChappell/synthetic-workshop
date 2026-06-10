"""Array I/O helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


def write_array(
    array: ArrayLike,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one NumPy array with conservative overwrite behaviour."""

    out_path = Path(path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.asarray(array))
    return out_path


def read_array(path: str | Path) -> np.ndarray:
    """Read a NumPy array."""

    return np.load(Path(path), allow_pickle=False)


def write_array_mapping(
    arrays: Mapping[str, ArrayLike],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write a mapping of named arrays as .npy files."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, array in arrays.items():
        paths[str(name)] = write_array(
            array,
            out_dir / f"{name}.npy",
            overwrite=overwrite,
        )
    return paths
