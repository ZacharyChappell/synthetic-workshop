from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.state import read_scene_text
from synthworkshop.gui.summary import (
    build_scene_summary,
    default_summary_path,
    save_scene_summary_json,
    summary_to_json,
)


def _basic_tube_text() -> str:
    return read_scene_text(get_catalogue_entry("basic_tube"))


def test_build_scene_summary_without_render() -> None:
    summary = build_scene_summary(_basic_tube_text(), render=False)

    assert summary["scene"]["id"] == "basic_tube"
    assert summary["grid"]["shape"] == [32, 32, 32]
    assert summary["objects"]["n_objects"] == 1
    assert summary["objects"]["kind_counts"]["tube"] == 1
    assert "fa_like" in summary["objects"]["map_names"]
    assert not summary["render"]["attempted"]


def test_build_scene_summary_with_render() -> None:
    summary = build_scene_summary(_basic_tube_text(), render=True)

    assert summary["render"]["attempted"]
    assert summary["render"]["passed"]
    assert "fa_like" in summary["render"]["scalar_maps"]
    assert "target" in summary["render"]["object_masks"]


def test_summary_to_json_is_valid_json() -> None:
    summary = build_scene_summary(_basic_tube_text(), render=False)
    text = summary_to_json(summary)
    decoded = json.loads(text)

    assert decoded["scene"]["id"] == "basic_tube"


def test_save_scene_summary_json_writes_file(tmp_path: Path) -> None:
    summary = build_scene_summary(_basic_tube_text(), render=False)
    out = tmp_path / "metadata" / "scene_summary.json"

    written = save_scene_summary_json(summary, out)

    assert written == out
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["scene"]["id"] == "basic_tube"


def test_save_scene_summary_json_refuses_overwrite(tmp_path: Path) -> None:
    summary = build_scene_summary(_basic_tube_text(), render=False)
    out = tmp_path / "scene_summary.json"
    out.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_scene_summary_json(summary, out, overwrite=False)


def test_default_summary_path() -> None:
    path = default_summary_path(
        output_root="outputs/gui/basic_tube",
        scene_id="basic_tube",
    )

    assert path == Path("outputs/gui/basic_tube/metadata/basic_tube_scene_summary.json")
