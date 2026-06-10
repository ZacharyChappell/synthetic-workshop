"""JSON serialisation helpers for scene metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-compatible values."""

    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return to_jsonable(value.item())

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())

    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=str)]

    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]

    return value


def write_json(
    payload: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
    indent: int = 2,
) -> Path:
    """Write a JSON file with conservative overwrite behaviour."""

    out_path = Path(path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(
        to_jsonable(payload),
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )
    out_path.write_text(text + "\n", encoding="utf-8")
    return out_path


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
