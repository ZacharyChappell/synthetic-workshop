from __future__ import annotations

from pathlib import Path

import pytest

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.file_io import (
    decode_uploaded_scene_bytes,
    default_saved_scene_path,
    read_scene_text_file,
    save_scene_text_file,
)
from synthworkshop.gui.state import read_scene_text


def _basic_tube_text() -> str:
    return read_scene_text(get_catalogue_entry("basic_tube"))


def test_decode_uploaded_scene_bytes_accepts_valid_yaml() -> None:
    text = _basic_tube_text()

    decoded = decode_uploaded_scene_bytes(text.encode("utf-8"))

    assert "basic_tube" in decoded


def test_decode_uploaded_scene_bytes_rejects_non_utf8() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        decode_uploaded_scene_bytes(b"\xff\xfe\x00")


def test_decode_uploaded_scene_bytes_rejects_non_mapping_yaml() -> None:
    with pytest.raises(ValueError, match="top level"):
        decode_uploaded_scene_bytes(b"- one\n- two\n")


def test_read_scene_text_file_reads_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "scene.yml"
    path.write_text(_basic_tube_text(), encoding="utf-8")

    text = read_scene_text_file(path)

    assert "basic_tube" in text


def test_read_scene_text_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_scene_text_file(tmp_path / "missing.yml")


def test_save_scene_text_file_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "saved" / "scene.yml"

    written = save_scene_text_file(
        _basic_tube_text(),
        path,
        overwrite=False,
    )

    assert written == path
    assert path.exists()
    assert "basic_tube" in path.read_text(encoding="utf-8")


def test_save_scene_text_file_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "scene.yml"
    path.write_text(_basic_tube_text(), encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_scene_text_file(
            _basic_tube_text(),
            path,
            overwrite=False,
        )


def test_save_scene_text_file_allows_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "scene.yml"
    path.write_text(_basic_tube_text(), encoding="utf-8")

    written = save_scene_text_file(
        _basic_tube_text(),
        path,
        overwrite=True,
    )

    assert written == path


def test_default_saved_scene_path() -> None:
    path = default_saved_scene_path(
        output_root="outputs/gui/basic_tube",
        scene_id="basic_tube",
    )

    assert path == Path("outputs/gui/basic_tube/scenes/basic_tube.yml")
