"""Legend-table helpers for rendered scenes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthworkshop.io.tables import write_table
from synthworkshop.scenes import RenderedScene


def scene_legend_table(scene: RenderedScene) -> pd.DataFrame:
    """Create a tabular object legend for a rendered scene."""

    rows: list[dict[str, object]] = []
    for object_id, metadata in scene.object_metadata.items():
        mask = scene.object_masks[object_id]
        rows.append(
            {
                "object_id": object_id,
                "name": metadata.name,
                "role": metadata.role.value,
                "label": metadata.label,
                "priority": metadata.priority,
                "mask_voxels": int(mask.sum()),
                "description": metadata.description,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "object_id",
            "name",
            "role",
            "label",
            "priority",
            "mask_voxels",
            "description",
        ],
    )


def write_scene_legend(
    scene: RenderedScene,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a scene legend table."""

    return write_table(scene_legend_table(scene), path, overwrite=overwrite)
