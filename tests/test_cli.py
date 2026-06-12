from __future__ import annotations

from pathlib import Path

import pytest

from synthworkshop.cli import build_parser, main


def test_cli_render_writes_outputs(tmp_path: Path) -> None:
    exit_code = main(
        [
            "render",
            "--config",
            "examples/basic_tube.yml",
            "--output-root",
            str(tmp_path),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "export" / "metadata" / "export_manifest.json").exists()
    assert (tmp_path / "export" / "tables" / "export_manifest.tsv").exists()
    assert (tmp_path / "gallery").exists()


def test_cli_render_can_skip_export_and_gallery(tmp_path: Path) -> None:
    exit_code = main(
        [
            "render",
            "--config",
            "examples/basic_tube.yml",
            "--output-root",
            str(tmp_path),
            "--no-export",
            "--no-gallery",
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "export").exists()
    assert not (tmp_path / "gallery").exists()


def test_cli_render_rejects_unknown_map_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown map_name"):
        main(
            [
                "render",
                "--config",
                "examples/basic_tube.yml",
                "--output-root",
                str(tmp_path),
                "--map-name",
                "not_a_map",
            ]
        )


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_validate_config_passes() -> None:
    exit_code = main(
        [
            "validate-config",
            "--config",
            "examples/basic_tube.yml",
        ]
    )

    assert exit_code == 0


def test_cli_validate_config_can_render() -> None:
    exit_code = main(
        [
            "validate-config",
            "--config",
            "examples/basic_tube.yml",
            "--render",
        ]
    )

    assert exit_code == 0


def test_cli_validate_config_fails_for_missing_file(tmp_path: Path) -> None:
    exit_code = main(
        [
            "validate-config",
            "--config",
            str(tmp_path / "missing.yml"),
        ]
    )

    assert exit_code == 1


def test_cli_catalogue_list() -> None:
    exit_code = main(["catalogue", "--list"])

    assert exit_code == 0


def test_cli_catalogue_show() -> None:
    exit_code = main(["catalogue", "--show", "basic_tube"])

    assert exit_code == 0


def test_cli_catalogue_render_requires_output_root() -> None:
    with pytest.raises(ValueError, match="--output-root is required"):
        main(["catalogue", "--render", "basic_tube"])


def test_cli_catalogue_render_writes_outputs(tmp_path: Path) -> None:
    exit_code = main(
        [
            "catalogue",
            "--render",
            "basic_tube",
            "--output-root",
            str(tmp_path),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "export" / "metadata" / "export_manifest.json").exists()
    assert (tmp_path / "gallery").exists()


def test_cli_gui_parser_accepts_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "gui",
            "--host",
            "127.0.0.1",
            "--port",
            "8502",
            "--no-browser",
        ]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 8502
    assert args.no_browser


def test_cli_catalogue_show_includes_validation_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["catalogue", "--show", "known_effect_tube"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Expected failure mode:" in captured.out
    assert "Recommended use:" in captured.out
    assert "Tags:" in captured.out
    assert "known-effect" in captured.out
